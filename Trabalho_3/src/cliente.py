import os
import cv2
import json
import time
import asyncio
import argparse
import signal
import sys
import numpy as np

import websockets
from assist import calculate_score, make_warning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cliente WS para enviar frames e visualizar detecções + profundidade")
    parser.add_argument("--ws-infer", default=os.getenv("WS_INFER", "ws://127.0.0.1:8000/ws/infer"), help="URL do WebSocket de detecção (/ws/infer)")
    parser.add_argument("--ws-depth", default=os.getenv("WS_DEPTH", "ws://127.0.0.1:8000/ws/depth"), help="URL do WebSocket de profundidade (/ws/depth)")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera do OpenCV")
    parser.add_argument("--width", type=int, default=640, help="Largura do frame de captura")
    parser.add_argument("--height", type=int, default=480, help="Altura do frame de captura")
    parser.add_argument("--jpeg", type=int, default=80, help="Qualidade JPEG (1-100)")
    parser.add_argument("--max-fps", type=float, default=15.0, help="Limite máximo de FPS de envio")
    parser.add_argument("--tts", action="store_true", help="Ativar TTS (usa make_warning do assist.py)")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Cooldown entre alertas TTS (s)")
    parser.add_argument("--use-roi", action="store_true", help="Usar distância por ROI ao invés do ponto central")
    parser.add_argument("--depth-model", default=os.getenv("DEPTH_MODEL", "MiDaS_small"), help="Modelo MiDaS (MiDaS_small, DPT_Large, DPT_Hybrid)")
    parser.add_argument("--temporal-filter", default=os.getenv("TEMPORAL_FILTER", "exponential"), help="Filtro temporal (none, mean, median, exponential)")
    return parser.parse_args()


def graceful_exit(cap: cv2.VideoCapture) -> None:
    try:
        if cap is not None:
            cap.release()
    finally:
        cv2.destroyAllWindows()

def draw_detections_local(frame: np.ndarray, detections, show_conf: bool = True, color=(0, 255, 0)) -> np.ndarray:
    frame_copy = frame.copy()
    for det in detections:
        bbox = det.get('bbox', (0, 0, 0, 0))
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except Exception:
            x1, y1, x2, y2 = 0, 0, 0, 0
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

        name_pt = det.get('name_pt') or det.get('name', 'objeto')
        if show_conf:
            try:
                label = f"{name_pt}: {float(det.get('confidence', 0.0)):.2f}"
            except Exception:
                label = str(name_pt)
        else:
            label = str(name_pt)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame_copy, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame_copy, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame_copy


async def run_client(ws_infer_url: str, ws_depth_url: str, camera_index: int, width: int, height: int, jpeg_quality: int, max_fps: float, enable_tts: bool, cooldown_s: float, use_roi: bool, depth_model: str, temporal_filter: str) -> None:
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("❌ Não foi possível abrir a câmera.")
        return

    print(f"Conectando a {ws_infer_url} e {ws_depth_url} ...")

    # Controle de taxa de envio
    min_period = 1.0 / max(max_fps, 1.0)

    loop = asyncio.get_running_loop()

    stop_flag = False

    def on_sigint(*_):
        nonlocal stop_flag
        stop_flag = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, on_sigint)
        except NotImplementedError:
            # Windows
            signal.signal(sig, lambda *_: on_sigint())

    last_warned_object_id = None
    last_warning_time = 0.0
    last_warned_score = -1e9

    try:
        async with websockets.connect(ws_infer_url, max_size=None) as ws_infer, \
                   websockets.connect(ws_depth_url, max_size=None) as ws_depth:
            print("✓ Conectado. Pressione ESC para sair.")

            last_send = 0.0
            while not stop_flag:
                loop_start = time.time()

                ret, frame = cap.read()
                if not ret or frame is None:
                    await asyncio.sleep(0.01)
                    continue

                # Limita FPS de envio
                now = time.time()
                if (now - last_send) < min_period:
                    await asyncio.sleep(0.001)
                    continue
                last_send = now

                ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                if not ok:
                    continue

                # 1) Detecção
                await ws_infer.send(buf.tobytes())
                try:
                    msg_det = await asyncio.wait_for(ws_infer.recv(), timeout=3.0)
                except asyncio.TimeoutError:
                    continue
                try:
                    det_data = json.loads(msg_det)
                except Exception:
                    continue
                detections = det_data.get("objects", [])

                # 2) Profundidade (pontos/rois baseados nas detecções)
                points = []
                rois = []
                for d in detections:
                    cx, cy = d.get("center", [0, 0])
                    x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])
                    points.append([int(cx), int(cy)])
                    rois.append([int(x1), int(y1), int(x2), int(y2)])

                cfg = {"points": points, "rois": rois, "model": depth_model, "temporal_filter": temporal_filter}
                await ws_depth.send(json.dumps(cfg))
                # aguarda ACK opcional (ok/bad_config) ou ignora
                try:
                    _ack = await asyncio.wait_for(ws_depth.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass

                await ws_depth.send(buf.tobytes())
                try:
                    msg_dep = await asyncio.wait_for(ws_depth.recv(), timeout=3.5)
                except asyncio.TimeoutError:
                    msg_dep = None

                distances_point = []
                distances_roi = []
                if msg_dep is not None:
                    try:
                        dep_data = json.loads(msg_dep)
                        distances_point = [p.get("distance", float("inf")) for p in dep_data.get("points", [])]
                        distances_roi = [r.get("distance", float("inf")) for r in dep_data.get("rois", [])]
                    except Exception:
                        pass

                # 3) Anotação + Relevância + TTS opcional
                h, w = frame.shape[:2]
                most_relevant = None
                best_score = -1e9
                processed = []

                for idx, d in enumerate(detections):
                    x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])
                    cx, cy = d.get("center", [0, 0])
                    obj_name = d.get("name_pt") or d.get("name", "objeto")
                    area = max(1, (x2 - x1) * (y2 - y1))
                    dist_p = distances_point[idx] if idx < len(distances_point) else float("inf")
                    dist_r = distances_roi[idx] if idx < len(distances_roi) else float("inf")
                    dist = float(dist_r if use_roi else dist_p)

                    score = float(calculate_score(d.get("id"), dist, area, obj_name, (cx, cy), w))
                    processed.append((idx, score, dist))
                    if score > best_score:
                        best_score = score
                        most_relevant = (idx, score, dist)

                frame_out = draw_detections_local(frame, detections, show_conf=True, color=(0, 255, 0))

                # Indicar o mais relevante e construir texto
                say_text = None
                if most_relevant is not None:
                    idx, score, dist = most_relevant
                    d = detections[idx]
                    x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])
                    cx, cy = d.get("center", [0, 0])
                    obj_name = d.get("name_pt") or d.get("name", "objeto")
                    pos = "direita" if cx > w * 0.66 else "esquerda" if cx < w * 0.33 else "frente"
                    proximity = "perto" if dist < 1.5 else "longe" if dist < 2.5 else "distante"
                    say_text = f"{obj_name} {proximity} à {pos}."

                    # destaque visual
                    cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_out, f"RELEVANTE | {obj_name} | {dist:.2f}m", (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                    # TTS com cooldown
                    if enable_tts and (time.time() - last_warning_time > cooldown_s):
                        if (last_warned_object_id != d.get("id") or score > last_warned_score * 1.2):
                            try:
                                make_warning(dist, obj_name, pos, proximity)
                                last_warned_object_id = d.get("id")
                                last_warned_score = score
                                last_warning_time = time.time()
                            except Exception as e:
                                print(f"TTS falhou: {e}")

                # HUD
                fps_text = f"Det FPS: {det_data.get('fps', 0.0):.1f} | Obj: {len(detections)}"
                if say_text:
                    cv2.putText(frame_out, say_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.putText(frame_out, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Cliente - ESC para sair", frame_out)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except Exception as e:
        print(f"❌ Erro no WebSocket: {e}")
    finally:
        graceful_exit(cap)


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_client(args.ws_infer, args.ws_depth, args.camera, args.width, args.height, args.jpeg, args.max_fps, args.tts, args.cooldown, args.use_roi, args.depth_model, args.temporal_filter))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()