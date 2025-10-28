"""
camera.py - Gerenciador de captura de vídeo otimizado
Autor: Raphael
Projeto: Navegação Assistida por Visão Computacional
"""

import cv2
import time
from threading import Thread, Lock
import numpy as np


class CameraManager:
    """
    Gerenciador de câmera com captura em thread separada
    para maximizar FPS e evitar travamentos
    """

    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        """
        Args:
            camera_id: ID da câmera (0 para padrão)
            width: Largura desejada
            height: Altura desejada
            fps: FPS alvo
        """
        self.camera_id = camera_id
        self.target_width = width
        self.target_height = height
        self.target_fps = fps

        self.cap = None
        self.frame = None
        self.running = False
        self.lock = Lock()

        # Métricas
        self.actual_fps = 0
        self.frame_count = 0
        self.start_time = None
        self.frame_times = []

    def start(self):
        """Iniciar captura de vídeo"""
        print(f"📷 Abrindo câmera {self.camera_id}...")

        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"❌ Não foi possível abrir câmera {self.camera_id}")

        # Configurar propriedades
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo

        # Capturar primeiro frame para validação
        ret, self.frame = self.cap.read()
        if not ret or self.frame is None:
            self.cap.release()
            raise RuntimeError("❌ Falha ao capturar primeiro frame")

        # Info
        h, w = self.frame.shape[:2]
        cam_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"✓ Câmera iniciada: {w}x{h} @ {cam_fps:.0f} FPS nominal")

        # Iniciar thread de captura
        self.running = True
        self.start_time = time.time()
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        return True

    def _capture_loop(self):
        """Loop de captura em thread separada"""
        last_time = time.time()

        while self.running:
            ret, frame = self.cap.read()

            if ret and frame is not None:
                # Atualizar frame com lock
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1

                    # Calcular FPS
                    current_time = time.time()
                    frame_time = current_time - last_time
                    last_time = current_time

                    # Manter histórico de 30 frames
                    self.frame_times.append(frame_time)
                    if len(self.frame_times) > 30:
                        self.frame_times.pop(0)

                    # FPS médio dos últimos frames
                    if len(self.frame_times) > 0:
                        avg_frame_time = sum(
                            self.frame_times) / len(self.frame_times)
                        self.actual_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            else:
                # Se falhar, pausar um pouco antes de tentar novamente
                time.sleep(0.01)

    def read(self):
        """
        Ler frame mais recente

        Returns:
            numpy.ndarray: Frame BGR ou None se não disponível
        """
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def get_fps(self):
        """Retornar FPS atual"""
        return self.actual_fps

    def get_resolution(self):
        """Retornar resolução atual"""
        with self.lock:
            if self.frame is not None:
                h, w = self.frame.shape[:2]
                return (w, h)
            return (0, 0)

    def is_opened(self):
        """Verificar se câmera está aberta"""
        return self.cap is not None and self.cap.isOpened() and self.running

    def stop(self):
        """Parar captura e liberar recursos"""
        print("📷 Desligando câmera...")

        self.running = False

        # Aguardar thread terminar (timeout de 1s)
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)

        # Liberar câmera
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        print("✓ Câmera desligada")


# Teste standalone
if __name__ == "__main__":
    print("="*60)
    print("TESTE DO GERENCIADOR DE CÂMERA")
    print("="*60)

    try:
        camera = CameraManager()
        camera.start()

        print("\nCapturando por 5 segundos...")
        start = time.time()

        while time.time() - start < 5.0:
            frame = camera.read()

            if frame is not None:
                # Mostrar preview
                fps_text = f"FPS: {camera.get_fps():.1f}"
                cv2.putText(frame, fps_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('Camera Test - ESC para sair', frame)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

        cv2.destroyAllWindows()

        # Estatísticas
        print("\n" + "="*60)
        print("ESTATÍSTICAS")
        print("="*60)
        print(f"FPS médio: {camera.get_fps():.1f}")
        print(f"Frames capturados: {camera.frame_count}")
        print(f"Resolução: {camera.get_resolution()}")

        camera.stop()
        print("\n✅ Teste concluído com sucesso!")

    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
