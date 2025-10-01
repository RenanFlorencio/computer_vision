import subprocess
import os
import time

initial_time = time.time()
print("Starting segmentation process...")
subprocess.run(["python3", "segmentador_all.py"])
segmentation_finish_time = time.time()
print(f"Segmentation completed. In {segmentation_finish_time - initial_time:.2f} seconds.")
print("Starting automatic COLMAP pipeline...")
subprocess.run(["python3", "automatic_all.py"])
colmap_finish_time = time.time()
print(f"COLMAP processing completed. In {colmap_finish_time - segmentation_finish_time:.2f} seconds.")
print(f"All tasks completed. In {colmap_finish_time - initial_time:.2f} seconds. Where segmentation took {segmentation_finish_time - initial_time:.2f} seconds and COLMAP took {colmap_finish_time - segmentation_finish_time:.2f} seconds.")