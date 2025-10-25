"""
test_integrated.py - Teste integrado de detecção + profundidade
Mostra objetos detectados com distâncias em tempo real
Autor: Raphael
"""
# ============================================================================
# CONFIGURAÇÃO DE AMBIENTE (WAYLAND FIX)
# ============================================================================
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['GDK_BACKEND'] = 'x11'

# ============================================================================
# ADICIONAR src/ AO PATH
# ============================================================================
import sys
# Adicionar pasta src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ============================================================================
# IMPORTS
# ============================================================================
from camera import CameraManager
from detector import ObjectDetector
from depth import DepthEstimator
import cv2
import numpy as np
import time
# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

# Zonas de distância (metros)
ZONE_CRITICAL = 0.5   # Vermelho - PERIGO!
ZONE_WARNING = 1.0    # Laranja - Atenção
ZONE_CAUTION = 2.0    # Amarelo - Cuidado
# > 2.0 = Verde - Seguro

# Cores para cada zona (BGR)
COLOR_CRITICAL = (0, 0, 255)      # Vermelho
COLOR_WARNING = (0, 165, 255)     # Laranja
COLOR_CAUTION = (0, 255, 255)     # Amarelo
COLOR_SAFE = (0, 255, 0)          # Verde

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def get_zone_color(distance):
    """Retornar cor baseada na distância"""
    if distance < ZONE_CRITICAL:
        return COLOR_CRITICAL, "PERIGO"
    elif distance < ZONE_WARNING:
        return COLOR_WARNING, "ATENCAO"
    elif distance < ZONE_CAUTION:
        return COLOR_CAUTION, "CUIDADO"
    else:
        return COLOR_SAFE, "SEGURO"

def draw_detection_with_distance(frame, detection, distance, depth_map=None):
    """
    Desenhar detecção com informação de distância
    
    Args:
        frame: Frame BGR
        detection: Dict com bbox, name_pt, confidence
        distance: Distância em metros
        depth_map: Mapa de profundidade (opcional, para colorir ROI)
    """
    x1, y1, x2, y2 = [int(v) for v in detection['bbox']]
    
    # Cor baseada na zona
    color, zone = get_zone_color(distance)
    
    # Bounding box
    thickness = 3 if distance < ZONE_WARNING else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Label com nome + distância
    label = f"{detection['name_pt']}: {distance:.2f}m"
    
    # Fundo do texto
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
    
    # Texto em branco
    cv2.putText(frame, label, (x1+2, y1-4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Zona (canto superior direito do bbox)
    zone_text = zone
    (ztw, zth), _ = cv2.getTextSize(zone_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x2-ztw-4, y1), (x2, y1+zth+4), color, -1)
    cv2.putText(frame, zone_text, (x2-ztw-2, y1+zth),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Centro do objeto
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    cv2.circle(frame, (cx, cy), 5, color, -1)
    
    return frame

def draw_statistics(frame, stats):
    """Desenhar estatísticas na tela"""
    y = 30
    spacing = 30
    
    # Fundo semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (350, 250), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Título
    cv2.putText(frame, "ESTATISTICAS", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += spacing
    
    # FPS
    cv2.putText(frame, f"Camera FPS: {stats['cam_fps']:.1f}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y += spacing - 5
    
    cv2.putText(frame, f"Detector: {stats['det_time']:.1f}ms ({stats['det_fps']:.1f} FPS)", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y += spacing - 5
    
    cv2.putText(frame, f"Depth: {stats['depth_time']:.1f}ms ({stats['depth_fps']:.1f} FPS)", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y += spacing - 5
    
    cv2.putText(frame, f"Pipeline: {stats['pipeline_fps']:.1f} FPS", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    y += spacing
    
    # Detecções
    cv2.putText(frame, f"Objetos detectados: {stats['num_detections']}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += spacing - 5
    
    # Por zona
    cv2.putText(frame, f"  PERIGO: {stats['critical']}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CRITICAL, 1)
    y += spacing - 10
    
    cv2.putText(frame, f"  ATENCAO: {stats['warning']}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WARNING, 1)
    y += spacing - 10
    
    cv2.putText(frame, f"  CUIDADO: {stats['caution']}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CAUTION, 1)
    y += spacing - 10
    
    cv2.putText(frame, f"  SEGURO: {stats['safe']}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_SAFE, 1)
    
    return frame

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTE INTEGRADO: YOLO + MiDaS")
    print("="*70)
    
    try:
        # Inicializar componentes
        print("\n📦 Inicializando componentes...")
        
        camera = CameraManager(width=640, height=480)
        detector = ObjectDetector(model='yolov8n.pt', conf=0.3)
        depth_estimator = DepthEstimator(temporal_filter='exponential')
        depth_estimator.alpha = 0.35
        
        # Verificar calibração
        if not depth_estimator.use_interpolation:
            print("\n⚠️  AVISO: MiDaS não calibrado!")
            print("   Distâncias serão aproximadas")
            print("   Execute 'python depth.py' para calibrar")
            input("\n👉 Pressione ENTER para continuar mesmo assim...")
        else:
            print(f"✅ MiDaS calibrado ({len(depth_estimator.calibration_points)} pontos)")
        
        # Iniciar câmera
        camera.start()
        
        print("\n🎥 Sistema iniciado!")
        print("="*70)
        print("CONTROLES:")
        print("   ESC     = Sair")
        print("   ESPAÇO  = Pausar/Continuar")
        print("   S       = Screenshot")
        print("="*70)
        input("\n👉 Pressione ENTER para começar...")
        
        # Variáveis
        frame_count = 0
        paused = False
        
        fps_time = time.time()
        fps_frames = 0
        pipeline_fps = 0
        
        while True:
            if not paused:
                # 1. CAPTURAR FRAME
                frame = camera.read()
                if frame is None:
                    continue
                
                h, w = frame.shape[:2]
                
                # 2. DETECTAR OBJETOS
                start_det = time.time()
                detections = detector.detect(frame, filter_classes=True)
                det_time = (time.time() - start_det) * 1000
                
                # 3. ESTIMAR PROFUNDIDADE
                start_depth = time.time()
                depth_map = depth_estimator.estimate(frame)
                depth_time = (time.time() - start_depth) * 1000
                
                # 4. CALCULAR DISTÂNCIAS DOS OBJETOS
                for det in detections:
                    distance = depth_estimator.get_distance_roi(
                        depth_map, det['bbox'], 
                    )
                    det['distance'] = distance
                
                # 5. CLASSIFICAR POR ZONA
                critical_objs = [d for d in detections if d['distance'] < ZONE_CRITICAL]
                warning_objs = [d for d in detections if ZONE_CRITICAL <= d['distance'] < ZONE_WARNING]
                caution_objs = [d for d in detections if ZONE_WARNING <= d['distance'] < ZONE_CAUTION]
                safe_objs = [d for d in detections if d['distance'] >= ZONE_CAUTION]
                
                # 6. DESENHAR NO FRAME
                frame_display = frame.copy()
                
                for det in detections:
                    draw_detection_with_distance(frame_display, det, det['distance'])
                
                # 7. ESTATÍSTICAS
                fps_frames += 1
                if time.time() - fps_time >= 1.0:
                    pipeline_fps = fps_frames
                    fps_frames = 0
                    fps_time = time.time()
                
                stats = {
                    'cam_fps': camera.get_fps(),
                    'det_time': det_time,
                    'det_fps': 1000/det_time if det_time > 0 else 0,
                    'depth_time': depth_time,
                    'depth_fps': 1000/depth_time if depth_time > 0 else 0,
                    'pipeline_fps': pipeline_fps,
                    'num_detections': len(detections),
                    'critical': len(critical_objs),
                    'warning': len(warning_objs),
                    'caution': len(caution_objs),
                    'safe': len(safe_objs),
                }
                
                draw_statistics(frame_display, stats)
                
                # 8. LEGENDAS
                cv2.putText(frame_display, "ESC=Sair | ESPACO=Pausar | S=Screenshot",
                           (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 9. MOSTRAR
                cv2.imshow('Teste Integrado: YOLO + MiDaS', frame_display)
                
                frame_count += 1
            
            else:
                # Pausado - mostrar mensagem
                frame_paused = frame_display.copy()
                cv2.putText(frame_paused, "PAUSADO", (w//2-60, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.imshow('Teste Integrado: YOLO + MiDaS', frame_paused)
            
            # TECLADO
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("\n⚠️  Saindo...")
                break
            
            elif key == 32:  # ESPAÇO
                paused = not paused
                print(f"{'⏸️  Pausado' if paused else '▶️  Continuando'}")
            
            elif key == ord('s') or key == ord('S'):
                # Screenshot
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame_display)
                print(f"📸 Screenshot salvo: {filename}")
        
        # LIMPAR
        camera.stop()
        cv2.destroyAllWindows()
        
        # RELATÓRIO FINAL
        print("\n" + "="*70)
        print("RELATÓRIO FINAL")
        print("="*70)
        print(f"Tempo total: {(time.time() - fps_time):.1f}s")
        print(f"Frames processados: {frame_count}")
        print(f"FPS médio do pipeline: {frame_count/(time.time()-fps_time):.1f}")
        print(f"\nComponentes:")
        print(f"  Câmera: {camera.get_fps():.1f} FPS")
        print(f"  Detector: {detector.get_avg_inference_time():.1f}ms")
        print(f"  Depth: {depth_estimator.get_avg_inference_time():.1f}ms")
        
        print("\n✅ Teste concluído!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrompido pelo usuário")
        camera.stop()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
