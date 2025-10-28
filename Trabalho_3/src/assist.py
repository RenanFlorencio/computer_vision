import cv2
import numpy as np
import os
import time
from detector import ObjectDetector
from depth import DepthEstimator
from tts import tts
import subprocess
import requests
from threading import Thread
import queue

# --- CONFIGURATION ---
FRAME_INTERVAL_DEPTH = 1        # depth thread can skip frames internally if needed
TTS_COOLDOWN_TIME = 5           # seconds between TTS warnings
TARGET_FPS = 30
FRAME_DURATION = 1.0 / TARGET_FPS

# --- SHARED DATA ---
frame_queue = queue.Queue(maxsize=1)  # queue for depth estimation
# latest depth map (updated by depth thread)
depth_map = None


# --- DEPTH WORKER THREAD ---
def depth_worker(estimator, frame_queue):
    global depth_map
    while True:
        frame = frame_queue.get()  # wait for new frame
        if frame is not None:
            depth_map = estimator.estimate(frame)
        frame_queue.task_done()


# --- TTS / WARNING FUNCTIONS ---
def make_warning(distance, obj_name, position):
    proximity = "próximo" if distance < 1.5 else "distante"
    print(f"⚠️ Aviso TTS: {obj_name} {proximity} à {position}")

    audio_path = os.path.join(
        os.path.dirname(
            __file__), f"tts/audios/{obj_name}_{position}_{proximity}.wav"
    )

    if not os.path.exists(audio_path):
        tts.synthesize_speech(
            f"Atenção! {obj_name} {proximity} à {position}.", audio_path
        )

    subprocess.Popen(["aplay", audio_path])


def make_warning_phone(distance, obj_name, position):
    proximity = "próximo" if distance < 1.5 else "distante"

    audio_path = os.path.join(
        os.path.dirname(
            __file__), f"tts/audios/{obj_name}_{position}_{proximity}.wav"
    )

    if not os.path.exists(audio_path):
        text = f"Atenção! {obj_name} {proximity} à {position}."
        tts.synthesize_speech(text, audio_path)

    with open(audio_path, "rb") as f:
        requests.post("http://100.118.7.80:5000/play_audio",
                      files={"audio": f})


# --- MAIN SCRIPT ---
if __name__ == "__main__":
    print("=" * 70)
    print("MERGED SYSTEM - OBJECT DETECTION + DEPTH ESTIMATION (MiDaS)")
    print("=" * 70)

    try:
        # Initialize detectors
        detector = ObjectDetector()
        estimator = DepthEstimator(temporal_filter='exponential')
        estimator.alpha = 0.35

        # Camera setup (RTSP)
        phone_url = "rtsp://100.118.7.80:8080/h264_ulaw.sdp"
        cap = cv2.VideoCapture(phone_url)
        if not cap.isOpened():
            raise Exception("❌ Camera not available!")

        print("\n🎥 Running system (ESC to exit)")

        # Start depth thread
        depth_thread = Thread(target=depth_worker,
                              args=(estimator, frame_queue))
        depth_thread.daemon = True
        depth_thread.start()

        frame_counter = 0
        last_warning_time = 0

        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                continue

            # --- Object detection ---
            detections = detector.detect(frame)

            # --- Push frame to depth queue ---
            try:
                # drop frame if queue is full
                frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

            frame_out = frame.copy()
            min_distance = {"dist": float("inf"), "name": "", "cx": 0}

            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Use latest depth map
                distance_txt = "..."
                if depth_map is not None:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    distance = estimator.get_distance_at_point(
                        depth_map, cx, cy, use_buffer=True)
                    if distance and distance < min_distance["dist"]:
                        min_distance["dist"] = distance
                        min_distance["name"] = det['name_pt']
                        min_distance["cx"] = cx
                    distance_txt = f"{distance:.2f}m" if distance else "..."

                conf = det.get('confidence', 0)
                label = det.get('name_pt', 'objeto')
                text = f"{label}: {conf:.2f} | {distance_txt}"
                cv2.putText(frame_out, text, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # --- TTS Warning ---
            if min_distance["dist"] != float("inf") and (time.time() - last_warning_time > TTS_COOLDOWN_TIME):
                h, w, _ = frame.shape
                pos = "direita" if min_distance["cx"] > w * \
                    0.66 else "esquerda" if min_distance["cx"] < w * 0.33 else "frente"
                make_warning_phone(
                    min_distance["dist"], min_distance["name"], pos)
                last_warning_time = time.time()

            frame_counter += 1

            # --- FPS ---
            elapsed = time.time() - loop_start
            fps = 1.0 / max(elapsed, 1e-6)
            cv2.putText(frame_out, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- Display ---
            cv2.imshow("Detector + Depth (ESC = exit)", frame_out)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            # --- FPS limiter ---
            sleep_time = FRAME_DURATION - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ System Stopped Successfully!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
