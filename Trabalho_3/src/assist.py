import cv2
import numpy as np
import os
import time
from detector import ObjectDetector
from depth import DepthEstimator
from tts import tts
import requests
from threading import Thread
import queue

# --- CONFIGURATION ---
DETECTION_INTERVAL = 10          # Run YOLO every N frames
TTS_COOLDOWN_TIME = 5           # seconds between TTS warnings
TARGET_FPS = 30
FRAME_DURATION = 1.0 / TARGET_FPS
DEPTH_SMOOTH_WINDOW = 5         # number of frames for depth smoothing
IP_CELULAR_TAILSCALE = "100.118.7.80"

# --- SHARED DATA ---
frame_queue = queue.Queue(maxsize=1)
depth_map = None

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

        # Camera
        phone_url = f"rtsp://{IP_CELULAR_TAILSCALE}:8080/h264_ulaw.sdp"
        cap = cv2.VideoCapture(phone_url)
        if not cap.isOpened():
            raise Exception("❌ Camera not available!")

        # Start depth thread
        depth_thread = Thread(target=depth_worker,
                              args=(estimator, frame_queue))
        depth_thread.daemon = True
        depth_thread.start()

        last_warning_time = 0
        frame_counter = 0

        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
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
                for obj in tracked_objects:
                    x1, y1, x2, y2 = obj['bbox']
                    cx, cy = obj['center']
                    obj_name = obj['name_pt']

                    # Depth smoothing
                    distance = estimator.get_distance_at_point(
                        depth_map, cx, cy, use_buffer=True)

                    if distance is not None:
                        depth_history.setdefault(obj['id'], [])
                        depth_history[obj['id']].append(distance)
                        depth_history[obj['id']
                                      ] = depth_history[obj['id']][-DEPTH_SMOOTH_WINDOW:]
                        smoothed_distance = sum(
                            depth_history[obj['id']]) / len(depth_history[obj['id']])
                    else:
                        smoothed_distance = None

                    # Draw bbox + distance
                    cv2.rectangle(frame_out, (x1, y1),
                                  (x2, y2), (0, 255, 0), 2)
                    distance_text = f"{smoothed_distance:.2f}m" if smoothed_distance else "..."
                    label = f"{obj_name} | {distance_text}"
                    cv2.putText(frame_out, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                    # TTS warning
                    if (smoothed_distance is not None and smoothed_distance < 1.5 and
                            time.time() - last_warning_time > TTS_COOLDOWN_TIME):
                        h, w, _ = frame.shape
                        pos = "direita" if cx > w * 0.66 else "esquerda" if cx < w * 0.33 else "frente"
                        make_warning_phone(smoothed_distance, obj_name, pos)
                        last_warning_time = time.time()

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

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
