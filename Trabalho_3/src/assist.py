import cv2
import numpy as np
import os
import time
from detector import ObjectDetector
from depth import DepthEstimator
import subprocess
from tts import tts
import requests

FRAME_INTERVAL_DEPTH = 20  # Depth every N frames
TTS_COOLDOWN_TIME = 5  # Cooldown time (seconds) between TTS warnings


def get_center_distance(det, depth_map, estimator):
    """Get depth from center of detection bounding box"""
    x1, y1, x2, y2, _, _ = det
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    if depth_map is None:
        return None

    return estimator.get_distance_at_point(depth_map, cx, cy, use_buffer=True)


def make_warning(distance, obj_name, position):

    proximity = "próximo" if (distance < 1.5) else "distante"

    print(f"⚠️ Aviso TTS: {obj_name} {proximity} à {position}")
    audio_path = os.path.join(
        os.path.dirname(__file__), f"tts/audios/{obj_name}_{position}_{proximity}.wav")

    if (not os.path.exists(audio_path)):
        start_time = time.time()
        tts.synthesize_speech(
            f"Atenção! {obj_name} {proximity} à {position}.", audio_path
        )
        print(
            f"Synthesis time: {time.time() - start_time:.2f}s")

    subprocess.Popen([
        "aplay",
        audio_path
    ])


def make_warning_phone(distance, obj_name, position):

    proximity = "próximo" if (distance < 1.5) else "distante"

    audio_path = os.path.join(
        os.path.dirname(__file__),
        f"tts/audios/{obj_name}_{position}_{proximity}.wav"
    )

    if not os.path.exists(audio_path):
        text = f"Atenção! {obj_name} {proximity} à {position}."
        tts.synthesize_speech(text, audio_path)

    with open(audio_path, 'rb') as f:
        requests.post(f"http://100.118.7.80:5000/play_audio",
                      files={"audio": f})


if __name__ == "__main__":

    print("=" * 70)
    print("MERGED SYSTEM - OBJECT DETECTION + DEPTH ESTIMATION (MiDaS)")
    print("=" * 70)

    try:
        # Initialize detectors
        detector = ObjectDetector()
        estimator = DepthEstimator(temporal_filter='exponential')
        estimator.alpha = 0.35  # smoothing filter

        # Conection through RTSP
        phone_url = "rtsp://100.118.7.80:8080/h264_ulaw.sdp"
        cap = cv2.VideoCapture(phone_url)

        # cap = cv2.VideoCapture(0)

        # Conection through IP camera
        # phone_url = "http://192.168.0.117:8080/video"
        # cap = cv2.VideoCapture(phone_url)

        if not cap.isOpened():
            raise Exception("❌ Camera not available!")

        print("\n🎥 Running system (ESC to exit)")

        frame_counter = 0
        last_warning_time = 0
        last_depth_map = None
        danger = False

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # --- Object detection (every frame)
            detections = detector.detect(frame)

            # --- Depth estimation every N frames (configurable)
            if frame_counter % FRAME_INTERVAL_DEPTH == 0:
                last_depth_map = estimator.estimate(frame)

            frame_out = frame.copy()
            # [distance, name, cx]
            min_distance = {"dist": float('inf'), "name": "", "cx": 0}

            for det in detections:
                # Bounding box
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Compute depth only if we have a depth map
                if last_depth_map is not None:
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    distance = estimator.get_distance_at_point(
                        last_depth_map, cx, cy, use_buffer=True
                    )
                    # Update minimum distance
                    if distance and distance < min_distance["dist"]:
                        min_distance["dist"] = distance
                        min_distance["name"] = det['name_pt']
                        min_distance["cx"] = cx

                    distance_txt = f"{distance:.2f}m" if distance else "..."

                else:
                    distance_txt = "..."

                # Label text (Portuguese name + confidence)
                conf = det.get('confidence', 0)
                label = det.get('name_pt', 'objeto')
                text = f"{label}: {conf:.2f} | {distance_txt}"

                # Draw text
                cv2.putText(frame_out, text, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0), 2)

            if min_distance["dist"] != float('inf') and time.time() - last_warning_time > TTS_COOLDOWN_TIME:

                h, w, _ = frame.shape
                position = "direita" if (
                    min_distance["cx"] > w * 0.66) else "esquerda" if (min_distance["cx"] < w * 0.33) else "frente"

                make_warning_phone(
                    min_distance["dist"], min_distance["name"], position)
                last_warning_time = time.time()

            frame_counter += 1

            # FPS computation
            inf_time = detector.get_avg_inference_time()
            fps = 1000 / inf_time if inf_time > 0 else 0

            cv2.putText(frame_out, f"FPS: {fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            # Display
            cv2.imshow("Detector + Depth (ESC = exit)", frame_out)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ System Stopped Successfully!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
