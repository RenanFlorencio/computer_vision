import os
import json
import time
import threading
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import base64

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from detector import ObjectDetector
from depth import DepthEstimator


# Configurações
PHONE_RTSP_URL = os.getenv("PHONE_RTSP_URL", "rtsp://127.0.0.1:8554/live/iphone")
DETECTION_FPS_LIMIT = float(os.getenv("DETECTION_FPS_LIMIT", "15"))  # Limite de iterações de detecção por segundo


class DetectionService:
    """
    Serviço de captura de vídeo + detecção + rastreamento
    Mantém o último resultado disponível para os clientes.
    """

    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.detector: Optional[ObjectDetector] = None
        self.capture: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self.latest_result: Dict[str, Any] = {"objects": [], "ts": 0.0, "fps": 0.0}
        self.latest_version: int = 0
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[Any] = None
        self.frame_id: int = 0

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, name="detector-loop", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        self._release_resources()

    def _release_resources(self) -> None:
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
            self.capture = None

    def _ensure_capture(self) -> bool:
        if self.capture is None:
            self.capture = cv2.VideoCapture(self.rtsp_url)
        return self.capture.isOpened()

    def _run_loop(self) -> None:
        self.detector = ObjectDetector()
        target_period = 1.0 / max(DETECTION_FPS_LIMIT, 1.0)

        while not self.stop_event.is_set():
            loop_start = time.time()

            if not self._ensure_capture():
                time.sleep(1.0)
                continue

            ret, frame = self.capture.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            with self.frame_lock:
                self.latest_frame = frame.copy()
                self.frame_id += 1

            try:
                detections = self.detector.detect_and_track(frame)
            except Exception:
                # Em caso de erro transitório de inferência, tenta próxima iteração
                detections = []

            avg_ms = self.detector.get_avg_inference_time() if self.detector else 0.0
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

            # Normaliza para JSON enxuto
            objects = []
            for d in detections:
                objects.append({
                    "id": d.get("id"),
                    "class_id": d.get("class_id"),
                    "name": d.get("name"),
                    "name_pt": d.get("name_pt"),
                    "confidence": round(float(d.get("confidence", 0.0)), 4),
                    "bbox": list(map(int, d.get("bbox", (0, 0, 0, 0)))),
                    "center": list(map(int, d.get("center", (0, 0)))),
                })

            self.latest_result = {
                "objects": objects,
                "ts": time.time(),
                "fps": round(fps, 2),
                "frame_id": self.frame_id,
            }
            self.latest_version += 1

            # Respeita limite de FPS do laço
            elapsed = time.time() - loop_start
            sleep_for = target_period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)


service = DetectionService(PHONE_RTSP_URL)


class DetectorHolder:
    """Inicializa o detector sob demanda e garante exclusão mútua na inferência."""

    def __init__(self) -> None:
        self.detector: Optional[ObjectDetector] = None
        self.lock = threading.Lock()

    def get(self) -> ObjectDetector:
        if self.detector is None:
            # Inicialização lazy para evitar custo no startup
            self.detector = ObjectDetector()
        return self.detector


detector_holder = DetectorHolder()


class DepthHolder:
    """Inicializa o DepthEstimator sob demanda e garante exclusão mútua na inferência."""

    def __init__(self) -> None:
        self.estimator: Optional[DepthEstimator] = None
        self.lock = threading.Lock()
        self.current_model: str = os.getenv("DEPTH_MODEL", "MiDaS_small")
        self.current_temporal: str = os.getenv("TEMPORAL_FILTER", "exponential")

    def _create(self, model: str, temporal: str) -> DepthEstimator:
        self.estimator = DepthEstimator(model=model, temporal_filter=temporal)
        self.current_model = model
        self.current_temporal = temporal
        return self.estimator

    def get(self, model: Optional[str] = None, temporal: Optional[str] = None) -> DepthEstimator:
        # Defaults from env if not provided
        model = model or self.current_model
        temporal = temporal or self.current_temporal
        if self.estimator is None:
            return self._create(model, temporal)
        if (self.current_model != model) or (self.current_temporal != temporal):
            return self._create(model, temporal)
        return self.estimator


depth_holder = DepthHolder()

app = FastAPI(title="Navegação Assistida - API de Detecção", version="0.1.0")


@app.on_event("startup")
def on_startup() -> None:
    # Modo nova API: não iniciar captura RTSP automática
    # service.start()
    pass


@app.on_event("shutdown")
def on_shutdown() -> None:
    service.stop()


@app.get("/health")
def health() -> JSONResponse:
    d = detector_holder.detector
    e = depth_holder.estimator
    status = {
        "status": "ok",
        "mode": "ws_infer",  # indicando que a API espera frames via WebSocket
        "rtsp_url": service.rtsp_url,
        "detector_device": getattr(d, "device", None),
        "avg_inference_ms": round(d.get_avg_inference_time(), 2) if d else None,
        "loop_fps": service.latest_result.get("fps", 0.0),
        "depth_model": depth_holder.current_model,
        "depth_filter": depth_holder.current_temporal,
        "depth_device": getattr(e, "device", None),
        "depth_avg_inference_ms": round(e.get_avg_inference_time(), 2) if e else None,
    }
    return JSONResponse(status)


@app.get("/latest")
def latest() -> JSONResponse:
    return JSONResponse(service.latest_result)


@app.websocket("/ws/objects")
async def ws_objects(ws: WebSocket) -> None:
    await ws.accept()

    last_version = -1
    try:
        while True:
            if service.latest_version != last_version:
                # Apenas envia quando houver atualização
                payload = json.dumps(service.latest_result)
                await ws.send_text(payload)
                last_version = service.latest_version
            await asyncio.sleep(0.02)  # ~50 Hz de checagem
    except WebSocketDisconnect:
        return


@app.websocket("/ws/depth")
async def ws_depth(ws: WebSocket) -> None:
    """
    Protocolo:
    - Mensagens de TEXTO (JSON) para configurar pontos/ROIs da próxima imagem:
      { "points": [[x,y], ...], "rois": [[x1,y1,x2,y2], ...] }
    - Mensagens BINÁRIAS (JPEG/PNG) com o frame. Resposta JSON por frame:
      {
        "frame_id": N,
        "ts": epoch,
        "fps": float,
        "points": [ {"x":x, "y":y, "distance":m} ],
        "rois":   [ {"bbox":[x1,y1,x2,y2], "distance":m} ],
        "depth_stats": {"min":v, "max":v}
      }
    """
    await ws.accept()

    pending_points: List[Tuple[int, int]] = []
    pending_rois: List[Tuple[int, int, int, int]] = []
    return_vis: bool = False
    vis_quality: int = 70
    selected_model: Optional[str] = None
    selected_temporal: Optional[str] = None
    frame_counter = 0

    try:
        while True:
            message = await ws.receive()
            data_bytes = message.get("bytes", None)
            text_data = message.get("text", None)

            if text_data is not None:
                try:
                    cfg = json.loads(text_data)
                    pts = cfg.get("points", []) or []
                    rois = cfg.get("rois", []) or []
                    # Opções de visualização
                    if "return_vis" in cfg:
                        return_vis = bool(cfg.get("return_vis"))
                    if isinstance(cfg.get("vis_quality"), int):
                        vis_quality = int(cfg.get("vis_quality"))
                    # Opções de modelo/filtro
                    if isinstance(cfg.get("model"), str):
                        selected_model = str(cfg.get("model"))
                    if isinstance(cfg.get("temporal_filter"), str):
                        selected_temporal = str(cfg.get("temporal_filter"))
                    pending_points = [(int(x), int(y)) for x, y in pts]
                    pending_rois = [tuple(map(int, r)) for r in rois]
                    await ws.send_text(json.dumps({
                        "ok": True,
                        "points": len(pending_points),
                        "rois": len(pending_rois),
                        "return_vis": return_vis,
                        "vis_quality": vis_quality,
                        "model": selected_model or depth_holder.current_model,
                        "temporal_filter": selected_temporal or depth_holder.current_temporal
                    }))
                except Exception:
                    await ws.send_text(json.dumps({"error": "bad_config"}))
                continue

            if data_bytes is None:
                continue

            npbuf = np.frombuffer(data_bytes, dtype=np.uint8)
            frame = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_text(json.dumps({"error": "invalid_image"}))
                continue

            frame_counter += 1

            with depth_holder.lock:
                estimator = depth_holder.get(model=selected_model, temporal=selected_temporal)
                depth_map = estimator.estimate(frame)
                avg_ms = estimator.get_avg_inference_time()

                point_results = []
                for (x, y) in pending_points:
                    dist = float(estimator.get_distance_at_point(depth_map, x, y, use_buffer=True))
                    point_results.append({"x": x, "y": y, "distance": dist})

                roi_results = []
                for (x1, y1, x2, y2) in pending_rois:
                    dist = float(estimator.get_distance_roi(depth_map, (x1, y1, x2, y2)))
                    roi_results.append({"bbox": [x1, y1, x2, y2], "distance": dist})

                dmin = float(np.min(depth_map)) if depth_map.size > 0 else 0.0
                dmax = float(np.max(depth_map)) if depth_map.size > 0 else 0.0

                depth_vis_b64: Optional[str] = None
                if return_vis:
                    try:
                        depth_vis = estimator.visualize_depth(depth_map)
                        ok, buf = cv2.imencode('.jpg', depth_vis, [int(cv2.IMWRITE_JPEG_QUALITY), int(vis_quality)])
                        if ok:
                            depth_vis_b64 = base64.b64encode(buf.tobytes()).decode('ascii')
                    except Exception:
                        depth_vis_b64 = None

            fps = 1000.0 / avg_ms if avg_ms and avg_ms > 0 else 0.0

            payload = {
                "frame_id": frame_counter,
                "ts": time.time(),
                "fps": round(fps, 2),
                "points": point_results,
                "rois": roi_results,
                "depth_stats": {"min": dmin, "max": dmax},
                "depth_model": depth_holder.current_model,
                "temporal_filter": depth_holder.current_temporal,
            }
            if return_vis and depth_vis_b64 is not None:
                payload["depth_vis_jpeg_b64"] = depth_vis_b64
            await ws.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        return


@app.websocket("/ws/infer")
async def ws_infer(ws: WebSocket) -> None:
    """
    Recebe frames binários (JPEG/PNG) e retorna JSON com objetos detectados por frame.
    - Mensagem de entrada: binário (conteúdo do arquivo .jpg/.png)
    - Mensagem de saída: JSON com {frame_id, ts, fps, objects}
    """
    await ws.accept()
    frame_counter = 0
    try:
        while True:
            message = await ws.receive()
            data_bytes = message.get("bytes", None)
            text_data = message.get("text", None)

            if text_data is not None:
                # Permite ping/configuração simples
                try:
                    msg = json.loads(text_data)
                    if msg.get("type") == "ping":
                        await ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))
                        continue
                except Exception:
                    # Ignora textos não-JSON
                    pass
                continue

            if data_bytes is None:
                # Nada para processar
                await asyncio.sleep(0.001)
                continue

            # Decodifica imagem
            npbuf = np.frombuffer(data_bytes, dtype=np.uint8)
            frame = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_text(json.dumps({"error": "invalid_image"}))
                continue

            frame_counter += 1

            # Inferência com exclusão mútua (GPU/CPU compartilhada)
            with detector_holder.lock:
                detector = detector_holder.get()
                detections = detector.detect_and_track(frame)
                avg_ms = detector.get_avg_inference_time()

            fps = 1000.0 / avg_ms if avg_ms and avg_ms > 0 else 0.0

            objects: List[Dict[str, Any]] = []
            for d in detections:
                objects.append({
                    "id": d.get("id"),
                    "class_id": d.get("class_id"),
                    "name": d.get("name"),
                    "name_pt": d.get("name_pt"),
                    "confidence": round(float(d.get("confidence", 0.0)), 4),
                    "bbox": list(map(int, d.get("bbox", (0, 0, 0, 0)))),
                    "center": list(map(int, d.get("center", (0, 0)))),
                })

            payload = {
                "frame_id": frame_counter,
                "ts": time.time(),
                "fps": round(fps, 2),
                "objects": objects,
            }
            await ws.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        return


@app.get("/frame.jpg")
def frame_jpg() -> Response:
    with service.frame_lock:
        frame = None if service.latest_frame is None else service.latest_frame.copy()
    if frame is None:
        return Response(status_code=503)
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return Response(status_code=500)
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/video.mjpeg")
def video_mjpeg() -> StreamingResponse:
    boundary = "frame"

    def gen():
        while True:
            with service.frame_lock:
                frame = None if service.latest_frame is None else service.latest_frame.copy()
            if frame is None:
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ok:
                time.sleep(0.01)
                continue
            jpg_bytes = buf.tobytes()
            yield (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpg_bytes)).encode() + b"\r\n\r\n" + jpg_bytes + b"\r\n"
            )

    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


