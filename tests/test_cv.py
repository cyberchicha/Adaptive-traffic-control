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
        logging.info("Please calibrate the camera for spatial estimation.")
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
            
        frame_count += 1
            
        # Step 1: Detection (YOLO)
        detections = detector.detect(frame)
        
        # Step 2: Tracking (SORT)
        tracked_objects = tracker.update(detections)
        
        # Step 3: Analytics (Queue Count N, Queue Length L, Density D)
        metrics = analyzer.process_frame(tracked_objects)

        # Print supervisor-approved metrics to console
        if frame_count % 15 == 0:
            print(f"\n--- Frame {frame_count} | Current Metrics ---")
            for lane_id in metrics["N"].keys():
                n = metrics["N"][lane_id]
                l = metrics["L"][lane_id]
                d = metrics["D"][lane_id]
                
                if n > 0:
                    print(f"{lane_id}: N = {n} veh | L = {l} m | D = {d} veh/m")
        
        # Visualization: Draw semi-transparent lanes
        display_frame = lane_manager.draw_lanes(frame, alpha=0.3)
        
        # Visualization: Draw bounding boxes and IDs
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Since speed and wait times are now internal criteria, only ID is displayed
            label = f"ID:{track_id}"
            cv2.putText(display_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Visualization: Information panel for lane metrics (top-left)
        y_offset = 30
        cv2.putText(display_frame, "Lane Metrics (N, L, D):", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for lane_id in metrics["N"].keys():
            n = metrics["N"][lane_id]
            if n > 0:
                l = metrics["L"][lane_id]
                d = metrics["D"][lane_id]
                
                y_offset += 30
                text = f"{lane_id}: N={n}, L={l}m, D={d}"
                cv2.putText(display_frame, text, (20, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        cv2.imshow("Adaptive Traffic Control - CV Pipeline", display_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()