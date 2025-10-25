"""Teste simplificado de depth - clique qualquer tecla para medir centro"""

import os
import sys

# Adicionar src ao path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

# Imports
import cv2
import numpy as np
from depth import DepthEstimator
from camera import CameraManager
from detector import ObjectDetector


estimator = DepthEstimator()
cap = cv2.VideoCapture(0)

print("="*60)
print("TESTE SIMPLES: Pressione ESPAÇO para medir CENTRO")
print("ESC para sair")
print("="*60)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    center = (w//2, h//2)
    
    # Depth
    depth_map = estimator.estimate(frame)
    depth_vis = estimator.visualize_depth(depth_map)
    
    # Medir centro
    distance = estimator.get_distance_at_point(depth_map, center[0], center[1])
    
    # Desenhar
    cv2.circle(frame, center, 15, (0, 255, 0), 3)
    cv2.circle(depth_vis, center, 15, (0, 255, 0), 3)
    
    text = f"Centro: {distance:.2f}m"
    cv2.putText(frame, text, (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(depth_vis, text, (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    combined = np.hstack([frame, depth_vis])
    cv2.imshow('Depth - Medindo CENTRO', combined)
    
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == 32:  # ESPAÇO
        print(f"📍 Centro: {distance:.2f}m")

cap.release()
cv2.destroyAllWindows()
