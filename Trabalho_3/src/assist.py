import cv2
import numpy as np
import os
import time
from detector import ObjectDetector
from depth import DepthEstimator
from tts import tts
import requests
from threading import Thread, Lock
import queue

# --- CONFIGURATION ---
DETECTION_INTERVAL = 10          # Run YOLO every N frames
TTS_COOLDOWN_TIME = 5           # seconds between TTS warnings
TARGET_FPS = 30
FRAME_DURATION = 1.0 / TARGET_FPS
DEPTH_SMOOTH_WINDOW = 5         # number of frames for depth smoothing
IP_CELULAR_TAILSCALE = "100.118.7.80"
PORTA_DROIDCAM = "4747"

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

# --- DEPTH THREAD ---


def depth_worker(estimator, frame_queue):
    global depth_map
    while True:
        frame = frame_queue.get()
        if frame is not None:
            depth_map = estimator.estimate(frame)
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

                make_warning_phone(obj["distance"], obj["name"], pos)
                last_warned_object_id = obj["id"]
                last_warned_score = highest_score
                last_warning_time = time.time()


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
def make_warning_phone(distance, obj_name, position):
    proximity = "próximo" if distance < 1.5 else "distante"
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


# --- MAIN SCRIPT ---
if __name__ == "__main__":
    print("=" * 70)
    print("NAVIGATION SYSTEM - DETECTION + TRACKING + DEPTH")
    print("=" * 70)

    try:
        detector = ObjectDetector()
        estimator = DepthEstimator(temporal_filter='exponential')
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
            tracked_objects = detector.detect_and_track(frame)

            # --- Process tracked objects ---
            if tracked_objects and depth_map is not None:
                highest_score = -float('inf')
                most_relevant_obj = None

                for obj in tracked_objects:
                    x1, y1, x2, y2 = obj['bbox']
                    cx, cy = obj['center']
                    obj_name = obj['name_pt']

                    distance = estimator.get_distance_at_point(
                        depth_map, cx, cy, use_buffer=True)
                    if distance is None:
                        continue

                    depth_history.setdefault(obj['id'], [])
                    depth_history[obj['id']].append(distance)
                    depth_history[obj['id']] = depth_history[obj['id']
                                                             ][-DEPTH_SMOOTH_WINDOW:]
                    smoothed_distance = sum(
                        depth_history[obj['id']]) / len(depth_history[obj['id']])

                    bbox_area = (x2 - x1) * (y2 - y1)
                    score = (1.0 / (smoothed_distance + 1e-3)) + \
                        0.001 * bbox_area

                    if score > highest_score:  # serving the queue that will be read by the monitor thread
                        highest_score = score
                        most_relevant_obj = {
                            "id": obj["id"],
                            "name": obj_name,
                            "distance": smoothed_distance,
                            "cx": cx,
                            "frame_shape": frame.shape,
                            "score": score
                        }

                    # Draw bbox + distance
                    cv2.rectangle(frame_out, (x1, y1),
                                  (x2, y2), (0, 255, 0), 2)
                    distance_text = f"{smoothed_distance:.2f}m" if smoothed_distance else "..."
                    label = f"{obj_name} | {distance_text}"
                    cv2.putText(frame_out, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                # Push most relevant object to the queue
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

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
