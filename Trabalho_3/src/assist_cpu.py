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
DETECTION_INTERVAL = 25          # Run YOLO every N frames
TTS_COOLDOWN_TIME = 4           # seconds between TTS warnings
TARGET_FPS = 20
FRAME_DURATION = 1.0 / TARGET_FPS
DEPTH_SMOOTH_WINDOW = 10         # number of frames for depth smoothing
SCORE_WINDOW = 10                # number of frames for score smoothing
# IP_CELULAR_TAILSCALE = "100.118.7.80" #Renan
IP_CELULAR_TAILSCALE = "100.64.18.21" # Igor
PORTA_DROIDCAM = "4747"

MODELO_PROFUNDIDADE = 'DPT_Hybrid'  # 'DPT_Hybrid', 'DPT_Large', 'MiDaS', 'depth_anything_v2_vits', 'depth_anything_v2_vitb', 'depth_anything_v2_vitl'

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

# Shared data structure for the latest frame shown on screen
latest_display_frame = {"frame": None, "lock": Lock()}

# Depth smoothing per object ID
depth_history = {}  # {obj_id: [distances]}
score_history = {}  # {obj_id: [scores]}

# --- DEPTH THREAD ---


def depth_worker(estimator, frame_queue):
    """Thread dedicada para estimar profundidade do frame mais recente."""
    global depth_map
    while True:
        try:
            # Pega o frame mais recente e descarta os antigos
            while True:
                frame = frame_queue.get(timeout=1)
                # se ainda há mais frames na fila, descarta este e pega o próximo
                if frame_queue.qsize() > 0:
                    frame_queue.task_done()
                    continue
                break
        except queue.Empty:
            continue

        if frame is None:
            break

        # Estima profundidade
        depth_map = estimator.estimate(frame)
        frame_queue.task_done()

def relevance_monitor():
    """Thread that continuously checks for the most relevant object and decides which warning to emit."""
    global last_warned_object_id, last_warned_score, last_warning_time

    warnings_dir = os.path.join(os.path.dirname(__file__), "warnings")
    os.makedirs(warnings_dir, exist_ok=True)

    # thresholds
    NEAR = 1.5
    MID = 2.5

    # fallback width if frame_shape missing
    FALLBACK_FRAME_WIDTH = 480

    while True:
        try:
            objs = relevant_obj_queue.get(timeout=1)  # agora recebo lista de objs
        except queue.Empty:
            continue

        if objs is None:
            break

        if not isinstance(objs, (list, tuple)) or len(objs) == 0:
            # nada para processar
            continue

        # tenta obter largura do frame a partir de qualquer objeto que tenha frame_shape
        frame_w = None
        for o in objs:
            fs = o.get("frame_shape")
            if fs and len(fs) >= 2:
                frame_w = fs[1]
                break
        if frame_w is None:
            frame_w = FALLBACK_FRAME_WIDTH

        # ============================== ANALISE GLOBAL ==============================
        any_near = any(o.get("distance", float('inf')) < NEAR for o in objs)

        # regiões (usa frame_w)
        left_objs = [o for o in objs if o.get("cx", 0) < frame_w * 0.33]
        right_objs = [o for o in objs if o.get("cx", 0) > frame_w * 0.66]
        front_objs = [o for o in objs if (o not in left_objs and o not in right_objs)]

        # considera "perto" ou "a distância média" usando MID threshold para as regras de virar
        near_left = any(o.get("distance", float('inf')) < MID for o in left_objs)
        near_right = any(o.get("distance", float('inf')) < MID for o in right_objs)
        near_front = any(o.get("distance", float('inf')) < MID for o in front_objs)

        # ============================== HIERARQUIA ==============================
        # Prioridade rígida: pare -> vire -> siga -> identificação

        # --- 1) PARE: se qualquer objeto está perto (NEAR)
        if any_near:
            if time.time() - last_warning_time > TTS_COOLDOWN_TIME:
                audio_path = os.path.join(os.path.dirname(__file__), "tts/audios/pare.wav")
                if not os.path.exists(audio_path):
                    tts.synthesize_speech("pare", audio_path)
                with open(audio_path, "rb") as f:
                    requests.post(f"http://{IP_CELULAR_TAILSCALE}:5000/play_audio", files={"audio": f})
                last_warning_time = time.time()

            # após emitir "pare" não emite outros avisos agora
            continue

        # --- 2) vire a esquerda: há objeto a distância média ou perto à frente e à direita
        if near_front and near_right:
            if time.time() - last_warning_time > TTS_COOLDOWN_TIME:
                audio_path = os.path.join(os.path.dirname(__file__), "tts/audios/vire_esquerda.wav")
                if not os.path.exists(audio_path):
                    tts.synthesize_speech("vire a esquerda", audio_path)
                with open(audio_path, "rb") as f:
                    requests.post(f"http://{IP_CELULAR_TAILSCALE}:5000/play_audio", files={"audio": f})
                last_warning_time = time.time()
            continue

        # --- 3) vire a direita: há objeto a distância média ou perto à frente e à esquerda
        if near_front and near_left:
            if time.time() - last_warning_time > TTS_COOLDOWN_TIME:
                audio_path = os.path.join(os.path.dirname(__file__), "tts/audios/vire_direita.wav")
                if not os.path.exists(audio_path):
                    tts.synthesize_speech("vire a direita", audio_path)
                with open(audio_path, "rb") as f:
                    requests.post(f"http://{IP_CELULAR_TAILSCALE}:5000/play_audio", files={"audio": f})
                last_warning_time = time.time()
            continue

        # --- 4) siga em frente: campo livre (nenhum objeto com distância < MID)
        if not any(o.get("distance", float('inf')) < MID for o in objs):
            if time.time() - last_warning_time > TTS_COOLDOWN_TIME:
                audio_path = os.path.join(os.path.dirname(__file__), "tts/audios/siga_em_frente.wav")
                if not os.path.exists(audio_path):
                    tts.synthesize_speech("siga em frente", audio_path)
                with open(audio_path, "rb") as f:
                    requests.post(f"http://{IP_CELULAR_TAILSCALE}:5000/play_audio", files={"audio": f})
                last_warning_time = time.time()
            continue

        # --- 5) fallback: aviso identificando o objeto mais relevante (se nada mais foi emitido)
        best = max(objs, key=lambda o: o.get("score", -float('inf')))
        if time.time() - last_warning_time > TTS_COOLDOWN_TIME:
            h_w = best.get("frame_shape", (0, frame_w))
            h, w = h_w[0], h_w[1] if len(h_w) >= 2 else (0, frame_w)
            cx = best.get("cx", 0)
            pos = "direita" if cx > w * 0.66 else "esquerda" if cx < w * 0.33 else "frente"
            proximity = "perto" if best.get("distance", float('inf')) < NEAR else \
                        "a distância média" if best.get("distance", float('inf')) < MID else "distante"
            make_warning_phone(best.get("distance", 0.0), best.get("name", "objeto"), pos, proximity)
            last_warning_time = time.time()

def frame_reader(cap, latest_frame):
    """Continuously read frames from the capture device, resize them, and store the most recent one."""
    TARGET_WIDTH = 480
    TARGET_HEIGHT = 360

    # 🔸 Garante que o buffer da câmera será mínimo
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while not latest_frame["stopped"]:
        # 🔸 Descarta frames antigos no buffer (mantém apenas o mais recente)
        for _ in range(3):
            cap.grab()  # lê o frame mas não o decodifica

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # 🔸 Reduz resolução para acelerar processamento
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT),
                           interpolation=cv2.INTER_AREA)

        # 🔸 Atualiza frame mais recente (thread-safe)
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
            # Dropa frames antigos antes de inserir o novo
            try:
                while not frame_queue.empty():
                    frame_queue.get_nowait()
                    frame_queue.task_done()
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass

            frame_out = frame.copy()

            # --- Detection every N frames ---
            tracked_objects = detector.detect_and_track(frame)

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
                        "score": score,
                        "frame_shape": frame.shape
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
                    relevant_obj_queue.put_nowait(processed_objects)

                except queue.Full:  # Replace the old object if the queue is full
                    relevant_obj_queue.get_nowait()
                    relevant_obj_queue.put_nowait(most_relevant_obj)

            # FPS limiter & display
            elapsed = time.time() - loop_start
            fps = 1.0 / max(elapsed, 1e-6)
            cv2.putText(frame_out, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Atualiza frame mostrado (com bounding boxes) para o monitor de relevância
            with latest_display_frame["lock"]:
                latest_display_frame["frame"] = frame_out.copy()

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
