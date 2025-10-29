"""
detector.py - Wrapper otimizado para YOLOv8
Autor: Raphael
Projeto: Navegação Assistida por Visão Computacional
"""

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
import numpy as np
import cv2
import time


class ObjectDetector:
    """
    Detector de objetos usando YOLOv8
    Otimizado para navegação em tempo real
    """

    # Classes COCO relevantes para navegação
    NAVIGATION_CLASSES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
        9: 'traffic light',
        10: 'fire hydrant',
        11: 'stop sign',
        13: 'bench',
        14: 'bird',
        15: 'cat',
        16: 'dog',
        24: 'backpack',
        25: 'umbrella',
        26: 'handbag',
        27: 'tie',
        28: 'suitcase',
        39: 'bottle',
        41: 'cup',
        42: 'fork',
        43: 'knife',
        44: 'spoon',
        45: 'bowl',
        56: 'chair',
        57: 'couch',
        58: 'potted plant',
        59: 'bed',
        60: 'dining table',
        62: 'tv',
        63: 'laptop',
        67: 'cell phone',
    }

    # Tradução para português
    TRANSLATIONS = {
        'person': 'pessoa',
        'bicycle': 'bicicleta',
        'car': 'carro',
        'motorcycle': 'moto',
        'bus': 'ônibus',
        'truck': 'caminhão',
        'traffic light': 'semáforo',
        'fire hydrant': 'hidrante',
        'stop sign': 'placa de pare',
        'bench': 'banco',
        'bird': 'pássaro',
        'cat': 'gato',
        'dog': 'cachorro',
        'backpack': 'mochila',
        'umbrella': 'guarda-chuva',
        'handbag': 'bolsa',
        'suitcase': 'mala',
        'bottle': 'garrafa',
        'cup': 'copo',
        'chair': 'cadeira',
        'couch': 'sofá',
        'potted plant': 'planta',
        'bed': 'cama',
        'dining table': 'mesa',
        'tv': 'televisão',
        'laptop': 'laptop',
    }

    def __init__(self, model='yolov8n.pt', conf=0.25, iou=0.45, device='cuda', half=True):
        """
        Args:
            model: Modelo YOLO (n/s/m/l/x)
            conf: Confidence threshold (0-1)
            iou: IoU threshold para NMS
            device: 'cuda' ou 'cpu'
            half: Usar FP16 (mais rápido na GPU)
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.conf = conf
        self.iou = iou
        self.half = half and (self.device == 'cuda')

        print(f"🎯 Carregando {model}...")
        print(f"   Device: {self.device.upper()}")
        print(f"   Precision: {'FP16' if self.half else 'FP32'}")

        self.model = YOLO(model)
        self.model.to(self.device)

        # Warm-up com dummy image
        print(f"   Aquecendo GPU...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model(dummy, verbose=False,
                       device=self.device, half=self.half)

        print(f"✓ Detector pronto")

        # Métricas
        self.inference_times = []
        self.track_history = {}

    def detect_and_track(self, frame):
        """
        Run YOLOv8 inference + ByteTrack tracking
        Returns list of tracked objects with ID
        """
        start = time.time()

        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            persist=True,
            device=self.device,
            half=self.half,
            verbose=False
        )

        inf_time = (time.time() - start) * 1000
        self.inference_times.append(inf_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)

        tracked = []
        r = results[0]

        if not hasattr(r, "boxes") or r.boxes is None:
            return tracked

        for box in r.boxes:
            class_id = int(box.cls[0])
            if class_id not in self.NAVIGATION_CLASSES:
                continue

            track_id = int(box.id[0]) if box.id is not None else None
            if track_id is None:
                continue  # We ignore untracked objects

            bbox = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            name = self.model.names[class_id]
            name_pt = self.TRANSLATIONS.get(name, name)

            tracked.append({
                'id': track_id,
                'class_id': class_id,
                'name': name,
                'name_pt': name_pt,
                'confidence': float(box.conf[0]),
                'bbox': (x1, y1, x2, y2),
                'center': (cx, cy)
            })

        return tracked

    def draw_detections(self, frame, detections, show_conf=True, color=(0, 255, 0)):
        """
        Desenhar detecções no frame

        Args:
            frame: Frame BGR
            detections: Lista de detecções
            show_conf: Mostrar confiança
            color: Cor das boxes (BGR)

        Returns:
            frame anotado
        """
        frame_copy = frame.copy()

        for det in detections:
            # Bounding box
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

            # Label
            if show_conf:
                label = f"{det['name_pt']}: {det['confidence']:.2f}"
            else:
                label = det['name_pt']

            # Fundo do texto
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_copy, (x1, y1-th-4), (x1+tw, y1), color, -1)

            # Texto
            cv2.putText(frame_copy, label, (x1, y1-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame_copy

    def get_avg_inference_time(self):
        """Retornar tempo médio de inferência (ms)"""
        if len(self.inference_times) > 0:
            return sum(self.inference_times) / len(self.inference_times)
        return 0


# Teste standalone
if __name__ == "__main__":
    print("="*60)
    print("TESTE DO DETECTOR DE OBJETOS")
    print("="*60)

    try:
        # Inicializar detector
        detector = ObjectDetector()

        # Testar com webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Câmera não disponível, testando com imagem dummy")
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            dets = detector.detect(frame)
            print(f"✓ Detecções: {len(dets)}")
        else:
            print("\n🎥 Testando com câmera (5 segundos)...\n")
            start = time.time()

            while time.time() - start < 20.0:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detectar
                dets = detector.detect(frame)

                # Desenhar
                frame_out = detector.draw_detections(frame, dets)

                # Info
                fps = 1000 / detector.get_avg_inference_time() if detector.get_avg_inference_time() > 0 else 0
                cv2.putText(frame_out, f"FPS: {fps:.1f} | Objetos: {len(dets)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Detector Test - ESC para sair', frame_out)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()

            print("\n" + "="*60)
            print("ESTATÍSTICAS")
            print("="*60)
            print(
                f"Inferência média: {detector.get_avg_inference_time():.1f}ms")
            print(f"FPS médio: {1000/detector.get_avg_inference_time():.1f}")

        print("\n✅ Teste concluído com sucesso!")

    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
