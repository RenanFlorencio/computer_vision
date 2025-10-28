"""
depth.py - Versão com FILTROS TEMPORAIS
"""

from collections import deque
from pathlib import Path
import json
import time
import numpy as np
import cv2
import torch
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['GDK_BACKEND'] = 'x11'


class DepthEstimator:
    """Estimador com filtros de estabilização"""

    def __init__(self, model='MiDaS_small', device='cuda',
                 calibration_file=None, temporal_filter='exponential'):
        """
        Args:
            temporal_filter: 'none', 'mean', 'median', 'exponential'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = model

        print(f"🌊 Carregando {model}...")
        print(f"   Device: {self.device.upper()}")

        self.model = torch.hub.load("intel-isl/MiDaS", model, verbose=False)
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", verbose=False)
        if 'small' in model.lower():
            self.transform = midas_transforms.small_transform
        else:
            self.transform = midas_transforms.dpt_transform

        print(f"✓ Depth estimator pronto")

        self.inference_times = []

        # Calibração
        self.calibration_points = []
        self.scale_factor = 2.5
        self.use_interpolation = False

        if calibration_file is None:
            self.calibration_file = Path(__file__).parent / 'calibration.json'
        else:
            self.calibration_file = Path(calibration_file)

        self.load_calibration()

        # ============================================
        # FILTROS TEMPORAIS
        # ============================================
        self.temporal_filter_type = temporal_filter

        # Para filtros de média/mediana
        self.depth_history = deque(maxlen=8)  # Últimos 8 frames

        # Para filtro exponencial
        self.prev_depth = None
        self.alpha = 0.35  # Peso do frame atual (0.2-0.5)

        # Buffer para distâncias (estabilizar get_distance)
        self.distance_buffer = deque(maxlen=5)

        print(f"   Filtro temporal: {temporal_filter}")

    def estimate(self, frame):
        """Estimar profundidade com filtro temporal"""
        start = time.time()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_input = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            depth = self.model(img_input)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = depth.cpu().numpy()

        # APLICAR FILTRO TEMPORAL
        depth_map = self._apply_temporal_filter(depth_map)

        # Métricas
        inf_time = (time.time() - start) * 1000
        self.inference_times.append(inf_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)

        return depth_map

    def _apply_temporal_filter(self, depth_map):
        """Aplicar filtro temporal escolhido"""

        if self.temporal_filter_type == 'none':
            return depth_map

        elif self.temporal_filter_type == 'mean':
            # FILTRO DE MÉDIA
            self.depth_history.append(depth_map)
            if len(self.depth_history) >= 3:
                return np.mean(list(self.depth_history), axis=0)
            return depth_map

        elif self.temporal_filter_type == 'median':
            # FILTRO DE MEDIANA (mais robusto)
            self.depth_history.append(depth_map)
            if len(self.depth_history) >= 5:
                return np.median(np.array(list(self.depth_history)), axis=0)
            return depth_map

        elif self.temporal_filter_type == 'exponential':
            # FILTRO EXPONENCIAL (mais leve)
            if self.prev_depth is not None:
                filtered = self.alpha * depth_map + \
                    (1 - self.alpha) * self.prev_depth
                self.prev_depth = filtered.copy()
                return filtered
            else:
                self.prev_depth = depth_map.copy()
                return depth_map

        return depth_map

    def get_distance_at_point(self, depth_map, x, y, use_buffer=True):
        """
        Distância em um ponto com buffer opcional

        Args:
            use_buffer: Se True, suaviza resultado com buffer temporal
        """
        h, w = depth_map.shape
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        depth_value = depth_map[y, x]
        distance = self.depth_to_meters(depth_value)

        # BUFFER ADICIONAL para distâncias
        if use_buffer:
            self.distance_buffer.append(distance)
            if len(self.distance_buffer) >= 3:
                # Mediana das últimas N medições
                distance = np.median(list(self.distance_buffer))

        return distance

    def get_distance_roi(self, depth_map, bbox):
        """Distância de ROI (já é suavizada por natureza)"""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = depth_map.shape
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))

        roi = depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            return float('inf')

        # Mediana da ROI (já suaviza)
        depth_value = np.median(roi)
        return self.depth_to_meters(depth_value)

    def add_calibration_point(self, depth_value, real_distance_meters):
        """Adicionar ponto de calibração"""
        self.calibration_points.append(
            (float(depth_value), float(real_distance_meters)))

        print(f"\n📍 Ponto {len(self.calibration_points)}:")
        print(f"   Depth: {depth_value:.4f} → {real_distance_meters:.2f}m")

        if len(self.calibration_points) >= 2:
            self.use_interpolation = True
            print(f"   ✅ Interpolação ativada")

        scales = [d * dist for d, dist in self.calibration_points]
        self.scale_factor = sum(scales) / len(scales)
        print(f"   Scale: {self.scale_factor:.2f}")

    def depth_to_meters(self, depth_value):
        """Converter para metros"""
        if depth_value <= 0:
            return float('inf')

        if self.use_interpolation and len(self.calibration_points) >= 2:
            return self._interpolate_distance(depth_value)

        return self.scale_factor / depth_value

    def _interpolate_distance(self, depth_value):
        """Interpolação entre pontos"""
        points = sorted(self.calibration_points, key=lambda x: x[0])

        if depth_value <= points[0][0]:
            d, dist = points[0]
            return dist * (d / depth_value)

        if depth_value >= points[-1][0]:
            d, dist = points[-1]
            return dist * (d / depth_value)

        for i in range(len(points) - 1):
            d1, dist1 = points[i]
            d2, dist2 = points[i+1]

            if d1 <= depth_value <= d2:
                t = (depth_value - d1) / (d2 - d1)
                return dist1 + t * (dist2 - dist1)

        return self.scale_factor / depth_value

    def save_calibration(self):
        """Salvar calibração"""
        data = {
            'calibration_points': self.calibration_points,
            'scale_factor': float(self.scale_factor),
            'use_interpolation': self.use_interpolation,
            'model': self.model_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n💾 Calibração salva: {self.calibration_file}")
            return True
        except Exception as e:
            print(f"❌ Erro: {e}")
            return False

    def load_calibration(self):
        """Carregar calibração"""
        if not self.calibration_file.exists():
            print(f"ℹ️  Sem calibração salva")
            return False

        try:
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)

            self.calibration_points = data['calibration_points']
            self.scale_factor = data['scale_factor']
            self.use_interpolation = data['use_interpolation']

            print(f"✅ Calibração carregada")
            print(f"   Pontos: {len(self.calibration_points)}")
            print(f"   Scale: {self.scale_factor:.2f}")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar: {e}")
            return False

    def visualize_depth(self, depth_map, colormap=cv2.COLORMAP_INFERNO):
        """Visualização colorida"""
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_normalized = (depth_map - depth_min) / \
                (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_map)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, colormap)

    def get_avg_inference_time(self):
        if len(self.inference_times) > 0:
            return sum(self.inference_times) / len(self.inference_times)
        return 0


# TESTE STANDALONE - adicionar no final do depth.py
if __name__ == "__main__":
    print("="*70)
    print("DEPTH ESTIMATOR - VERSÃO COM FILTROS")
    print("="*70)

    try:
        # Inicializar com filtro exponencial
        estimator = DepthEstimator(temporal_filter='exponential')
        estimator.alpha = 0.35  # Ajustar suavização (0.2-0.6)

        # phone_url = "rtsp://100.118.7.80:8080/h264_ulaw.sdp"
        # cap = cv2.VideoCapture(phone_url)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Câmera indisponível")
            exit(1)

        # Status
        if estimator.use_interpolation:
            print("\n✅ Calibração carregada!")
        else:
            print("\n⚠️  Sem calibração - valores aproximados")

        print("\n📝 CONTROLES:")
        print("   1, 2, 3 = Calibrar (0.5m, 1.0m, 2.0m)")
        print("   S       = Salvar calibração")
        print("   R       = Reset/Recalibrar")
        print("   +/-     = Ajustar filtro (mais/menos suavização)")
        print("   ESC     = Sair")
        print("="*70)
        input("\n👉 ENTER para começar...")

        depth_vals = []
        frame_count = 0
        fps_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            center = (w//2, h//2)

            # Estimar profundidade (JÁ COM FILTRO!)
            depth_map = estimator.estimate(frame)
            depth_vis = estimator.visualize_depth(depth_map)

            # Valor raw
            raw = depth_map[center[1], center[0]]
            depth_vals.append(raw)
            if len(depth_vals) > 30:
                depth_vals.pop(0)
            avg_raw = np.mean(depth_vals)

            # Distância (COM BUFFER!)
            distance = estimator.get_distance_at_point(
                depth_map, center[0], center[1], use_buffer=True
            )

            # FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_time
                fps = 30 / elapsed if elapsed > 0 else 0
                fps_time = time.time()
            else:
                fps = 1000 / estimator.get_avg_inference_time() if estimator.get_avg_inference_time() > 0 else 0

            # ===== DESENHAR =====

            # Mira central
            cv2.circle(frame, center, 30, (0, 255, 0), 3)
            cv2.line(frame, (center[0]-40, center[1]),
                     (center[0]+40, center[1]), (0, 255, 0), 3)
            cv2.line(frame, (center[0], center[1]-40),
                     (center[0], center[1]+40), (0, 255, 0), 3)

            # Status calibração
            if estimator.use_interpolation:
                status = "CALIBRADO"
                color = (0, 255, 0)
            else:
                status = "SEM CALIBRACAO"
                color = (0, 0, 255)

            # Info principal
            cv2.putText(frame, f"{distance:.2f}m", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4)

            cv2.putText(frame, status, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Detalhes
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Filtro: {estimator.temporal_filter_type}", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, f"Alpha: {estimator.alpha:.2f}", (10, 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, f"Pontos: {len(estimator.calibration_points)}", (10, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Instruções
            cv2.putText(frame, "1/2/3=Cal | S=Save | +/-=Filtro | ESC=Sair",
                        (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Combinar
            combined = np.hstack([frame, depth_vis])
            cv2.imshow('Depth Estimator - Filtrado', combined)

            # Teclado
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == ord('1'):
                print(f"\n📍 Calibrando 0.5m (raw: {avg_raw:.4f})")
                estimator.add_calibration_point(avg_raw, 0.5)
            elif key == ord('2'):
                print(f"\n📍 Calibrando 1.0m (raw: {avg_raw:.4f})")
                estimator.add_calibration_point(avg_raw, 1.0)
            elif key == ord('3'):
                print(f"\n📍 Calibrando 2.0m (raw: {avg_raw:.4f})")
                estimator.add_calibration_point(avg_raw, 2.0)
            elif key == ord('s') or key == ord('S'):
                if estimator.save_calibration():
                    print("✅ Calibração salva!")
            elif key == ord('r') or key == ord('R'):
                print("\n🔄 Reset - Recalibrando...")
                estimator.calibration_points = []
                estimator.scale_factor = 2.5
                estimator.use_interpolation = False
            elif key == ord('+') or key == ord('='):
                estimator.alpha = min(0.9, estimator.alpha + 0.05)
                print(f"➕ Alpha: {estimator.alpha:.2f} (menos suave)")
            elif key == ord('-') or key == ord('_'):
                estimator.alpha = max(0.1, estimator.alpha - 0.05)
                print(f"➖ Alpha: {estimator.alpha:.2f} (mais suave)")

        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "="*70)
        print("SESSÃO FINALIZADA")
        print("="*70)
        print(f"Calibrado: {'Sim' if estimator.use_interpolation else 'Não'}")
        print(f"Pontos: {len(estimator.calibration_points)}")
        print(f"Alpha: {estimator.alpha:.2f}")

        if estimator.use_interpolation and len(estimator.calibration_points) > 0:
            if input("\n💾 Salvar? (s/n): ").lower() == 's':
                estimator.save_calibration()

        print("\n✅ Concluído!")

    except KeyboardInterrupt:
        print("\n⚠️  Interrompido")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
