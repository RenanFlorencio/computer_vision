import cv2
import numpy as np
import os
import time
import sys
from detector import ObjectDetector
from depth import DepthEstimator
from tts import tts
import requests
from threading import Thread, Lock
import queue
import logging
from pathlib import Path

# --- CONFIGURATION ---
DETECTION_INTERVAL = 10          # Run YOLO every N frames
TTS_COOLDOWN_TIME = 5           # seconds between TTS warnings
TARGET_FPS = 30

# Criar pasta de logs se não existir
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)
# Nome do arquivo com timestamp
log_file = log_dir / f'navigation_{time.strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # Log para arquivo
        logging.StreamHandler(sys.stdout)              # Log para console também
    ]
)

FRAME_DURATION = 1.0 / TARGET_FPS
DEPTH_SMOOTH_WINDOW = 10         # number of frames for depth smoothing
SCORE_WINDOW = 10                # number of frames for score smoothing
# IP_CELULAR_TAILSCALE = "100.118.7.80" #Renan
IP_CELULAR_TAILSCALE = "100.79.114.120" # Igor
PORTA_DROIDCAM = "4747"

MODELO_PROFUNDIDADE = 'DPT_Hybrid'  # 'DPT_Hybrid', 'DPT_Large', 'MiDaS', 'depth_anything_v2_vits', 'depth_anything_v2_vitb', 'depth_anything_v2_vitl'
logging.info(f"Using depth model: {MODELO_PROFUNDIDADE}")

# --- SHARED DATA ---
frame_queue = queue.Queue(maxsize=1)
depth_map = None

# Keeps the most recent relevant object for TTS
relevant_obj_queue = queue.Queue(maxsize=1)
last_warned_object_id = None
last_warning_time = 100
last_warned_score = -float('inf')

# Shared data structure for the latest frame
latest_frame = {"frame": None, "lock": Lock(), "stopped": False}

# Depth smoothing per object ID
depth_history = {}  # {obj_id: [distances]}
score_history = {}  # {obj_id: [scores]}

# --- DEPTH THREAD ---


def depth_worker(estimator, frame_queue):
    global depth_map
    while True:
        frame = frame_queue.get()
        if frame is not None:
            st = time.time()
            depth_map = estimator.estimate(frame)
            logging.info(f"Depth estimation time: {time.time() - st:.5f}s")
        frame_queue.task_done()


def relevance_monitor():
    """Thread that continuously checks for the most relevant object."""
    global last_warned_object_id, last_warned_score, last_warning_time

    while True:
        try:
            obj = relevant_obj_queue.get(timeout=1)
        except queue.Empty:
            continue

        if obj is None:
            # Signal to stop
            break

        highest_score = obj["score"]

        if (time.time() - last_warning_time > TTS_COOLDOWN_TIME):
            if (last_warned_object_id != obj["id"] or
                    highest_score > last_warned_score * 1.2):

                print(f"Warning most relevant object: {obj['name']} at "
                      f"{obj['distance']: .2f}m (score: {highest_score: .2f})")
                # Compute direction
                h, w, _ = obj["frame_shape"]
                cx = obj["cx"]
                pos = "direita" if cx > w * 0.66 else "esquerda" if cx < w * 0.33 else "frente"
                proximity = "perto" if obj["distance"] < 1.5 else "longe" if obj["distance"] < 2.5 else "distante"

                logging.info(f"Warning most relevant object: {obj['name']} at "
                             f"{obj['distance']: .2f}m (score: {highest_score: .2f})")
                st = time.time()
                make_warning_phone(
                    obj["distance"], obj["name"], pos, proximity)
                last_warned_object_id = obj["id"]
                last_warned_score = highest_score
                last_warning_time = time.time()
                logging.info(f"TTS generation and sending time: {time.time() - st:.5f}s")


def frame_reader(cap, latest_frame):
    """Continuously read frames from the capture device and store the most recent one."""
    while not latest_frame["stopped"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # Store the most recent frame (thread-safe)
        with latest_frame["lock"]:
            latest_frame["frame"] = frame


# --- TTS FUNCTIONS ---
def make_warning_phone(distance, obj_name, position, proximity):
    audio_path = os.path.join(
        os.path.dirname(
            __file__), f"tts/audios/{obj_name}_{position}_{proximity}.wav"
    )
    if not os.path.exists(audio_path):
        tts.synthesize_speech(
            f"{obj_name} {proximity} à {position}.", audio_path)

    with open(audio_path, "rb") as f:
        requests.post(f"http://{IP_CELULAR_TAILSCALE}:5000/play_audio",
                      files={"audio": f})


def calculate_score(obj_id, distance, bbox_area, object_name, center, frame_width):
    """Calculate relevance score."""
    current_score = (1.0 / (distance + 1e-3)) + 0.001 * bbox_area  # base score

    score_history.setdefault(obj_id, [])
    score_history[obj_id].append(current_score)
    # keep last SCORE_WINDOW frames
    score_history[obj_id] = score_history[obj_id][-SCORE_WINDOW:]

    smoothed_score = sum(score_history[obj_id]) / len(score_history[obj_id])

    if distance <= 0.6:
        smoothed_score += 5.0  # high bonus for very close objects

    if center[0] < frame_width * 0.33 or center[0] > frame_width * 0.66:
        smoothed_score *= 0.8  # penalize objects far from center

    if object_name in ["pessoa", "mesa", "cama", "cadeira", "sofá"]:
        smoothed_score *= 1.5  # prioritize certain objects

    return smoothed_score


# --- MAIN SCRIPT ---
if __name__ == "__main__":
    print("=" * 70)
    print("NAVIGATION SYSTEM - DETECTION + TRACKING + DEPTH")
    print("=" * 70)

    try:
        detector = ObjectDetector()
        estimator = DepthEstimator(temporal_filter='exponential', model = MODELO_PROFUNDIDADE)
        estimator.alpha = 0.35

        # RTSP URL from IP Webcam app
        # phone_url = f"rtsp://{IP_CELULAR_TAILSCALE}:8080/h264_ulaw.sdp"
        # cap = cv2.VideoCapture(phone_url)

        # Camera
        video_url = f"http://{IP_CELULAR_TAILSCALE}:{PORTA_DROIDCAM}/video"
        cap = cv2.VideoCapture(video_url)

        if not cap.isOpened():
            raise RuntimeError("Cannot open video stream")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Start the frame reader thread
        reader_thread = Thread(target=frame_reader, args=(
            cap, latest_frame), daemon=True)
        reader_thread.start()

        # Start depth thread
        depth_thread = Thread(target=depth_worker,
                              args=(estimator, frame_queue))
        depth_thread.daemon = True
        depth_thread.start()

        # Start relevance monitor thread
        monitor_thread = Thread(target=relevance_monitor, daemon=True)
        monitor_thread.start()

        frame_counter = 0

        while True:
            loop_start = time.time()

            with latest_frame["lock"]:
                frame = latest_frame["frame"].copy(
                ) if latest_frame["frame"] is not None else None

            if frame is None:
                time.sleep(0.01)
                continue

            # Push frame to depth queue
            try:
                frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

            frame_out = frame.copy()

            # --- Detection every N frames ---
            st = time.time()
            tracked_objects = detector.detect_and_track(frame)
            logging.info(f"Detection+Tracking time: {time.time() - st:.5f}s")


            # --- Process tracked objects ---
            if tracked_objects and depth_map is not None:
                highest_score = -float('inf')
                most_relevant_obj = None
                processed_objects = []

                for obj in tracked_objects:
                    x1, y1, x2, y2 = obj['bbox']
                    cx, cy = obj['center']
                    obj_name = obj['name_pt']

                    # distance = estimator.get_distance_at_point(
                    #     depth_map, cx, cy, use_buffer=True)

                    bbox_int = (int(x1), int(y1), int(x2), int(y2))

                    distance = estimator.get_distance_roi(
                        depth_map, bbox_int)

                    if distance is None:
                        continue

                    depth_history.setdefault(obj['id'], [])
                    depth_history[obj['id']].append(distance)
                    depth_history[obj['id']] = depth_history[obj['id']
                                                             ][-DEPTH_SMOOTH_WINDOW:]
                    smoothed_distance = sum(
                        depth_history[obj['id']]) / len(depth_history[obj['id']])

                    bbox_area = (x2 - x1) * (y2 - y1)
                    center = (cx, cy)
                    score = calculate_score(obj["id"],
                                            smoothed_distance, bbox_area, obj_name, center, frame.shape[1])

                    # Store all info in a dict
                    processed_objects.append({
                        "id": obj["id"],
                        "name": obj_name,
                        "distance": smoothed_distance,
                        "bbox": (x1, y1, x2, y2),
                        "cx": cx,
                        "score": score
                    })

                most_relevant_obj = max(
                    processed_objects, key=lambda o: o["score"], default=None)
                most_relevant_obj["frame_shape"] = frame.shape

                # Draw all objects
                for obj in processed_objects:
                    x1, y1, x2, y2 = obj["bbox"]
                    is_most_relevant = obj["id"] == most_relevant_obj["id"]
                    box_color = (0, 0, 255) if is_most_relevant else (
                        255, 0, 0)
                    distance_text = f"{obj['distance']:.2f}m"
                    label = f"RELEVANT | {obj['name']} | {distance_text}" if is_most_relevant else f"{obj['name']} | {distance_text}"

                    cv2.rectangle(frame_out, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame_out, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

                try:
                    relevant_obj_queue.put_nowait(most_relevant_obj)

                except queue.Full:  # Replace the old object if the queue is full
                    relevant_obj_queue.get_nowait()
                    relevant_obj_queue.put_nowait(most_relevant_obj)

            # FPS limiter & display
            elapsed = time.time() - loop_start
            fps = 1.0 / max(elapsed, 1e-6)
            cv2.putText(frame_out, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Navigation System (ESC=exit)", frame_out)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            sleep_time = FRAME_DURATION - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            frame_counter += 1

        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ System Stopped Successfully!")
        relevant_obj_queue.put(None)  # Stop monitor thread
        latest_frame["stopped"] = True
        reader_thread.join()
        monitor_thread.join()
        depth_thread.join()
        
        exit(0)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
