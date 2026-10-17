"""
End-to-End Visual Validation for the Computer Vision Pipeline.
Integrates Calibration, Lane Management, Detection, Tracking, and Traffic Analytics.
"""

import cv2
import sys
import os
import logging
import numpy as np

# Configure logging format
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Append project root to import paths to allow relative imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from core.cv.calibration import CameraCalibration
from core.cv.lane_manager import LaneManager
from core.cv.detector import VehicleDetector
from core.cv.tracker import VehicleTracker
from core.cv.analyzer import TrafficAnalyzer

VIDEO_PATH = os.path.join(ROOT_DIR, "data", "videos", "highway_mini.mp4")
WEIGHTS_PATH = os.path.join(ROOT_DIR, "data", "weights", "best.pt")

def main() -> None:
    logging.info("--- Initializing Full CV Pipeline ---")
    
    # 1. Initialize camera calibration
    calibrator = CameraCalibration(VIDEO_PATH)
    calib_config_path = os.path.join(ROOT_DIR, "data", "configs", "calibration.json")

    calibrator.load_first_frame()

    if not calibrator.load_calibration(calib_config_path):
        logging.warning("Calibration not found. Launching interactive calibration...")
        logging.info("Please calibrate the camera for speed estimation.")
        calibrator.collect_2d_points()
        calibrator.collect_3d_points()
        calibrator.calibrate()
        calibrator.save_calibration(calib_config_path)
    else:
        logging.info("Calibration loaded successfully from JSON.")

    # 2. Initialize lane manager
    lane_manager = LaneManager()
    if not lane_manager.load_lanes():
        logging.warning("Lane config not found. Launching interactive annotation...")
        lane_manager.annotate_interactively(VIDEO_PATH)
        
    # 3. Initialize YOLO detector and SORT tracker
    if not os.path.exists(WEIGHTS_PATH):
        logging.error(f"Weights not found at {WEIGHTS_PATH}")
        return
        
    detector = VehicleDetector(model_path=WEIGHTS_PATH)
    tracker = VehicleTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # 4. Initialize traffic analyzer

    cap = cv2.VideoCapture(VIDEO_PATH)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps == 0 or np.isnan(actual_fps):
        actual_fps = 25.0
    cap.release()

    analyzer = TrafficAnalyzer(lane_manager=lane_manager, calibrator=calibrator, fps=actual_fps)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logging.error("Failed to open video stream.")
        return

    logging.info("Starting real-time processing. Press 'q' to stop.")

    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.info("Video stream ended.")
            break
            
        # Step 1: Detection (YOLO)
        detections = detector.detect(frame)
        
        # Step 2: Tracking (SORT)
        tracked_objects = tracker.update(detections)
        
        # Step 3: Analytics (speed, queue lengths, wait times)
        metrics = analyzer.process_frame(tracked_objects)

        if frame_count % 15 == 0:
            print(f"\n--- Фрейм {frame_count} | Поточні метрики ---")
            print(f"Черги: {metrics['queue_lengths']}")
            
            if metrics["speeds"]:
                speeds_str = ", ".join([f"ID:{k} - {v:.1f}km/h" for k, v in metrics["speeds"].items()])
                print(f"Швидкості: {speeds_str}")
                
            if metrics["wait_times"]:
                waits_str = ", ".join([f"ID:{k} - {v:.1f}s" for k, v in metrics["wait_times"].items()])
                print(f"Час очікування: {waits_str}")
        
        # Visualization: Draw semi-transparent lanes
        display_frame = lane_manager.draw_lanes(frame, alpha=0.3)
        
        # Visualization: Draw bounding boxes, IDs, and calculated speeds
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Retrieve metrics for the specific track ID
            speed = metrics["speeds"].get(track_id, 0.0)
            wait_time = metrics["wait_times"].get(track_id, 0.0)
            
            # Format the label text above the vehicle
            label = f"ID:{track_id} | {speed:.1f}km/h"
            if wait_time > 0:
                label += f" | Wait:{wait_time:.1f}s"
                
            cv2.putText(display_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Visualization: Information panel for queue lengths (top-left)
        y_offset = 30
        cv2.putText(display_frame, "Queue Lengths:", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for lane_id, count in metrics["queue_lengths"].items():
            y_offset += 30
            cv2.putText(display_frame, f"{lane_id}: {count} veh", (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        cv2.imshow("Adaptive Traffic Control - CV Pipeline", display_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()