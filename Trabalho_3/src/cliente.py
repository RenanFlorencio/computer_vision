import os
import cv2
import json
import time
import threading
import argparse
import signal
import sys
import numpy as np
import queue
import requests
import websockets
import asyncio
from assist_cpu import calculate_score, make_warning_phone, make_warning_phone_nav  # mantido

# -------------------- Argumentos --------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cliente WS + navegação com alertas TTS via make_warning_phone")
    parser.add_argument("--ws-infer", default=os.getenv("WS_INFER", "ws://127.0.0.1:8000/ws/infer"))
    parser.add_argument("--ws-depth", default=os.getenv("WS_DEPTH", "ws://127.0.0.1:8000/ws/depth"))
    parser.add_argument("--camera", type=str, default="0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--jpeg", type=int, default=80)
    parser.add_argument("--max-fps", type=float, default=15.0)
    parser.add_argument("--tts", action="store_true")
    parser.add_argument("--cooldown", type=float, default=5.0)
    parser.add_argument("--use-roi", action="store_true")
    parser.add_argument("--depth-model", default=os.getenv("DEPTH_MODEL", "MiDaS_small"))
    parser.add_argument("--temporal-filter", default=os.getenv("TEMPORAL_FILTER", "exponential"))
    return parser.parse_args()

# -------------------- Funções auxiliares --------------------
def draw_detections(frame, detections, fps=None):
    f = frame.copy()
    # distância vertical inicial para evitar overlap
    y_offset = 20

    for d in detections:
        bbox = d.get("bbox", [0, 0, 0, 0])
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except Exception:
            continue

        label = d.get("name_pt") or d.get("name", "objeto")
        conf = d.get("confidence", 0)

        # Desenha retângulo
        color = (0, 255, 0) if label != "RELEVANTE" else (0, 0, 255)
        cv2.rectangle(f, (x1, y1), (x2, y2), color, 2)

        # Ajuste de posição do texto para evitar overlap
        if label == "RELEVANTE":
            text_pos_y = y_offset
            y_offset += 25  # incrementa distância para próximo texto
        else:
            text_pos_y = y1 - 5 if y1 > 15 else y2 + 15

        # Desenha texto
        cv2.putText(f, f"{label} {conf:.2f}", (x1, text_pos_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Mostra FPS se fornecido
    if fps is not None:
        cv2.putText(f, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return f

# -------------------- Threads --------------------
class VideoCaptureThread(threading.Thread):
    def __init__(self, camera, width, height):
        super().__init__()
        self.cap = cv2.VideoCapture(camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False

    def run(self):
        while not self.stopped:
            ret, frm = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frm

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()

class WebSocketWorker(threading.Thread):
    def __init__(self, ws_infer_url, ws_depth_url, jpeg_quality, max_fps, depth_model, temporal_filter, use_roi):
        super().__init__()
        self.ws_infer_url = ws_infer_url
        self.ws_depth_url = ws_depth_url
        self.jpeg_quality = jpeg_quality
        self.max_fps = max_fps
        self.depth_model = depth_model
        self.temporal_filter = temporal_filter
        self.use_roi = use_roi
        self.frame_queue = queue.Queue(maxsize=1)
        self.detections = []
        self.distances_point = []
        self.distances_roi = []
        self.stopped = False

    def enqueue_frame(self, f):
        if not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(f)

    def run(self):
        asyncio.run(self._loop())

    async def _loop(self):
        min_period = 1.0 / max(self.max_fps, 1.0)
        last_send = 0.0
        try:
            async with websockets.connect(self.ws_infer_url, max_size=None) as ws_infer, \
                       websockets.connect(self.ws_depth_url, max_size=None) as ws_depth:
                while not self.stopped:
                    try:
                        frame = self.frame_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    now = time.time()
                    if now - last_send < min_period:
                        await asyncio.sleep(0.001)
                        continue
                    last_send = now

                    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                    if not ok:
                        continue

                    # --- Envia para inferência
                    await ws_infer.send(buf.tobytes())
                    try:
                        msg_det = await asyncio.wait_for(ws_infer.recv(), timeout=3)
                        data_det = json.loads(msg_det)
                        self.detections = data_det.get("objects", [])
                    except Exception:
                        self.detections = []

                    # --- Calcula pontos e ROIs
                    pts, rois = [], []
                    for d in self.detections:
                        cx, cy = d.get("center", [0, 0])
                        x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])
                        pts.append([int(cx), int(cy)])
                        rois.append([int(x1), int(y1), int(x2), int(y2)])
                    cfg = {"points": pts, "rois": rois, "model": self.depth_model, "temporal_filter": self.temporal_filter}
                    await ws_depth.send(json.dumps(cfg))

                    try:
                        _ack = await asyncio.wait_for(ws_depth.recv(), timeout=1)
                    except asyncio.TimeoutError:
                        pass

                    # --- Envia frame para profundidade
                    await ws_depth.send(buf.tobytes())
                    try:
                        msg_dep = await asyncio.wait_for(ws_depth.recv(), timeout=3)
                        data_dep = json.loads(msg_dep)
                        self.distances_point = [p.get("distance", float("inf")) for p in data_dep.get("points", [])]
                        self.distances_roi = [r.get("distance", float("inf")) for r in data_dep.get("rois", [])]
                    except Exception:
                        self.distances_point = []
                        self.distances_roi = []
        except Exception as e:
            print(f"❌ Erro no WebSocket: {e}")

# -------------------- Lógica de Navegação --------------------
relevant_obj_queue = queue.Queue(maxsize=1)
depth_history = {}
score_history = {}

def calculate_score_nav(obj_id, distance, bbox_area, name, center, frame_width):
    score = (1.0 / (distance + 1e-6)) + 0.001 * bbox_area
    score_history.setdefault(obj_id, [])
    score_history[obj_id].append(score)
    score_history[obj_id] = score_history[obj_id][-10:]
    s = sum(score_history[obj_id]) / len(score_history[obj_id])
    if distance <= 0.6:
        s += 5.0
    if center[0] < frame_width * 0.33 or center[0] > frame_width * 0.66:
        s *= 0.8
    if name in ["pessoa", "mesa", "cama", "cadeira", "sofá"]:
        s *= 1.5
    return s

def emit_navigation_warning(warning_text: str, tts_enabled: bool, last_warning_time: float, cooldown: float) -> float:
    """Emite warnings de navegação legíveis (pare, vire, siga)."""
    import time
    now = time.time()
    if tts_enabled and (now - last_warning_time > cooldown):
        # Usa a nova função make_warning_phone_nav
        make_warning_phone_nav(warning_text)
        last_warning_time = now
    return last_warning_time


def relevance_monitor_thread(stop_event: threading.Event, cooldown: float, tts_enabled: bool):
    last_warning_time = 0.0
    NEAR = 1.5
    MID = 2.5
    FALLBACK_W = 480

    while not stop_event.is_set():
        try:
            objs = relevant_obj_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if objs is None:
            break
        if not objs:
            continue

        frame_w = next((o.get("frame_shape")[1] for o in objs if o.get("frame_shape")), FALLBACK_W)

        any_near = any(o.get("distance", float("inf")) < NEAR for o in objs)
        left_objs = [o for o in objs if o.get("cx", 0) < frame_w * 0.33]
        right_objs = [o for o in objs if o.get("cx", 0) > frame_w * 0.66]
        front_objs = [o for o in objs if o not in left_objs and o not in right_objs]

        near_left = any(o.get("distance", float("inf")) < MID for o in left_objs)
        near_right = any(o.get("distance", float("inf")) < MID for o in right_objs)
        near_front = any(o.get("distance", float("inf")) < MID for o in front_objs)

        # --- 1) PARE
        if any_near:
            last_warning_time = emit_navigation_warning("pare", tts_enabled, last_warning_time, cooldown)
            continue

        # --- 2) VIRE À ESQUERDA
        if near_front and near_right:
            last_warning_time = emit_navigation_warning("vire à esquerda", tts_enabled, last_warning_time, cooldown)
            continue

        # --- 3) VIRE À DIREITA
        if near_front and near_left:
            last_warning_time = emit_navigation_warning("vire à direita", tts_enabled, last_warning_time, cooldown)
            continue

        # --- 4) SIGA EM FRENTE
        if not any(o.get("distance", float('inf')) < MID for o in objs):
            last_warning_time = emit_navigation_warning("siga em frente", tts_enabled, last_warning_time, cooldown)
            continue

        # --- 5) Fallback: identificação de objeto específico
        best = max(objs, key=lambda o: o.get("score", -float("inf")))
        if best and tts_enabled and time.time() - last_warning_time > cooldown:
            dist = best.get("distance", 0.0)
            pos = "direita" if best.get("cx", 0) > frame_w * 0.66 else "esquerda" if best.get("cx", 0) < frame_w * 0.33 else "frente"
            prox = "perto" if dist < NEAR else "a média distância" if dist < MID else "distante"
            make_warning_phone(dist, best.get("name", "objeto"), pos, prox)
            last_warning_time = time.time()

# -------------------- Main --------------------
def main():
    args = parse_args()
    cap_thread = VideoCaptureThread(args.camera, args.width, args.height)
    ws_thread = WebSocketWorker(args.ws_infer, args.ws_depth, args.jpeg, args.max_fps, args.depth_model, args.temporal_filter, args.use_roi)
    cap_thread.start()
    ws_thread.start()

    stop_event = threading.Event()
    nav_thread = threading.Thread(target=relevance_monitor_thread, args=(stop_event, args.cooldown, args.tts), daemon=True)
    nav_thread.start()

    stop_flag = False
    signal.signal(signal.SIGINT, lambda *_: setattr(sys.modules[__name__], "stop_flag", True))
    signal.signal(signal.SIGTERM, lambda *_: setattr(sys.modules[__name__], "stop_flag", True))

    while not stop_flag:
        frame = cap_thread.read()
        if frame is None:
            time.sleep(0.01)
            continue
        ws_thread.enqueue_frame(frame)

        detections = ws_thread.detections
        distances_p = ws_thread.distances_point
        distances_r = ws_thread.distances_roi

        h, w = frame.shape[:2]
        processed = []
        best = None
        best_score = -1e9

        for i, d in enumerate(detections):
            x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])
            cx, cy = d.get("center", [0, 0])
            name = d.get("name_pt") or d.get("name", "objeto")
            area = max(1, (x2 - x1) * (y2 - y1))
            dist_p = distances_p[i] if i < len(distances_p) else float("inf")
            dist_r = distances_r[i] if i < len(distances_r) else float("inf")
            dist = dist_r if args.use_roi else dist_p

            obj_id = d.get("id", f"idx_{i}")
            depth_history.setdefault(obj_id, [])
            depth_history[obj_id].append(dist)
            depth_history[obj_id] = depth_history[obj_id][-10:]
            smoothed_dist = sum(depth_history[obj_id]) / len(depth_history[obj_id])

            score = calculate_score_nav(obj_id, smoothed_dist, area, name, (cx, cy), w)
            processed.append({
                "id": obj_id,
                "idx": i,
                "score": score,
                "distance": smoothed_dist,
                "bbox": (x1, y1, x2, y2),
                "cx": cx,
                "frame_shape": frame.shape,
                "name": name
            })
            if score > best_score:
                best = processed[-1]
                best_score = score

        frame_out = draw_detections(frame, detections)
        if best:
            x1, y1, x2, y2 = best["bbox"]
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame_out, f"RELEVANTE: {best['name']} {best['distance']:.2f}m",
                        (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        try:
            relevant_obj_queue.put_nowait(processed)
        except queue.Full:
            try:
                relevant_obj_queue.get_nowait()
            except queue.Empty:
                pass
            relevant_obj_queue.put_nowait(processed)

        cv2.imshow("Cliente - ESC para sair", frame_out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap_thread.stop()
    ws_thread.stopped = True
    stop_event.set()
    relevant_obj_queue.put(None)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
