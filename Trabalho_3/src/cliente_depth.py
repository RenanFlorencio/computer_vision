import os
import cv2
import json
import time
import asyncio
import argparse
import signal
import numpy as np

import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cliente WS para testar profundidade (/ws/depth) semelhante ao standalone")
    parser.add_argument("--ws-depth", default=os.getenv("WS_DEPTH", "ws://127.0.0.1:8000/ws/depth"), help="URL do WebSocket de profundidade (/ws/depth)")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera do OpenCV")
    parser.add_argument("--width", type=int, default=640, help="Largura do frame de captura")
    parser.add_argument("--height", type=int, default=480, help="Altura do frame de captura")
    parser.add_argument("--jpeg", type=int, default=80, help="Qualidade JPEG (1-100)")
    parser.add_argument("--max-fps", type=float, default=15.0, help="Limite máximo de FPS de envio")
    parser.add_argument("--show-depth", action="store_true", help="Solicitar e exibir depth map (vis) lado a lado")
    parser.add_argument("--depth-model", default=os.getenv("DEPTH_MODEL", "MiDaS_small"), help="Modelo MiDaS (MiDaS_small, DPT_Large, DPT_Hybrid)")
    parser.add_argument("--temporal-filter", default=os.getenv("TEMPORAL_FILTER", "exponential"), help="Filtro temporal (none, mean, median, exponential)")
    return parser.parse_args()


def graceful_exit(cap: cv2.VideoCapture) -> None:
    try:
        if cap is not None:
            cap.release()
    finally:
        cv2.destroyAllWindows()


def draw_hud(frame: np.ndarray, distance_m: float, api_fps: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    center = (w // 2, h // 2)

    # Mira central
    cv2.circle(out, center, 30, (0, 255, 0), 3)
    cv2.line(out, (center[0] - 40, center[1]), (center[0] + 40, center[1]), (0, 255, 0), 3)
    cv2.line(out, (center[0], center[1] - 40), (center[0], center[1] + 40), (0, 255, 0), 3)

    # Distância e FPS
    color = (0, 255, 0) if np.isfinite(distance_m) and distance_m < 10 else (0, 0, 255)
    cv2.putText(out, f"{distance_m:.2f}m", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    cv2.putText(out, f"API FPS: {api_fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return out


async def run_depth_client(ws_depth_url: str, camera_index: int, width: int, height: int, jpeg_quality: int, max_fps: float, show_depth: bool, depth_model: str, temporal_filter: str) -> None:
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("❌ Não foi possível abrir a câmera.")
        return

    print(f"Conectando a {ws_depth_url} ...")

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
            signal.signal(sig, lambda *_: on_sigint())

    try:
        async with websockets.connect(ws_depth_url, max_size=None, open_timeout=10) as ws_depth:
            print("✓ Conectado. Pressione ESC para sair.")

            last_send = 0.0
            local_fps_t0 = time.time()
            local_frames = 0
            api_fps_display = 0.0

            while not stop_flag:
                ret, frame = cap.read()
                if not ret or frame is None:
                    await asyncio.sleep(0.01)
                    continue

                # Ponto central
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2

                now = time.time()
                if (now - last_send) < min_period:
                    # apenas exibe o último HUD
                    out = draw_hud(frame, float("inf"), api_fps_display)
                    cv2.imshow("Depth Client - ESC para sair", out)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue
                last_send = now

                ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                if not ok:
                    continue

                # Configura pontos (centro)
                cfg = {"points": [[int(cx), int(cy)]], "model": depth_model, "temporal_filter": temporal_filter}
                if show_depth:
                    cfg.update({"return_vis": True, "vis_quality": 70})
                await ws_depth.send(json.dumps(cfg))
                # ACK opcional
                try:
                    _ = await asyncio.wait_for(ws_depth.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass

                # Envia frame
                await ws_depth.send(buf.tobytes())

                # Recebe distâncias
                try:
                    msg_dep = await asyncio.wait_for(ws_depth.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    msg_dep = None

                distance_m = float("inf")
                depth_vis_img = None
                if msg_dep is not None:
                    try:
                        dep = json.loads(msg_dep)
                        pts = dep.get("points", [])
                        if pts:
                            distance_m = float(pts[0].get("distance", float("inf")))
                        api_fps_display = float(dep.get("fps", 0.0))
                        if show_depth and "depth_vis_jpeg_b64" in dep:
                            import base64
                            jpg_bytes = base64.b64decode(dep["depth_vis_jpeg_b64"])  # noqa
                            depth_vis_img = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    except Exception:
                        pass

                # HUD + FPS local
                out = draw_hud(frame, distance_m, api_fps_display)
                if show_depth and depth_vis_img is not None:
                    try:
                        h1, w1 = out.shape[:2]
                        h2, w2 = depth_vis_img.shape[:2]
                        if h1 != h2:
                            scale = h1 / max(1, h2)
                            depth_vis_img = cv2.resize(depth_vis_img, (int(w2 * scale), h1))
                        combined = np.hstack([out, depth_vis_img])
                        out = combined
                    except Exception:
                        pass
                local_frames += 1
                if local_frames % 30 == 0:
                    elapsed = time.time() - local_fps_t0
                    local_fps = 30.0 / elapsed if elapsed > 0 else 0.0
                    local_fps_t0 = time.time()
                    cv2.putText(out, f"Local FPS: {local_fps:.1f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Depth Client - ESC para sair", out)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except Exception as e:
        print(f"❌ Erro no WebSocket: {e}")
    finally:
        graceful_exit(cap)


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_depth_client(args.ws_depth, args.camera, args.width, args.height, args.jpeg, args.max_fps, args.show_depth, args.depth_model, args.temporal_filter))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


