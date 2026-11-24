"""
End-to-End Visual Validation for the Computer Vision Pipeline.
Integrates IPM Calibration, Lane Management, Detection, Tracking, and Traffic Analytics.
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

from config import (
    CALIBRATION_CONFIG_PATH,
    MODEL_WEIGHTS_PATH,
    TEST_VIDEO_PATH,
)

from core.cv.calibration import CameraCalibration
from core.cv.lane_manager import LaneManager
from core.cv.detector import VehicleDetector
from core.cv.tracker import VehicleTracker
from core.cv.analyzer import TrafficAnalyzer

def main() -> None:
    logging.info("--- Initializing Full CV Pipeline ---")

    video_path = str(TEST_VIDEO_PATH)
    weights_path = str(MODEL_WEIGHTS_PATH)
    
    # 1. Initialize IPM camera calibration
    calibrator = CameraCalibration(video_path)
    if not calibrator.load_calibration(CALIBRATION_CONFIG_PATH):
        logging.warning("Calibration not found. Launching interactive calibration...")
        calibrator.load_first_frame()
        calibrator.collect_2d_points()
        calibrator.collect_3d_points()
        calibrator.calibrate()
        calibrator.save_calibration(CALIBRATION_CONFIG_PATH)
    else:
        logging.info("IPM Calibration loaded successfully from JSON.")

    # 2. Initialize lane manager
    lane_manager = LaneManager()
    if not lane_manager.load_lanes():
        logging.warning("Lane config not found. Launching interactive annotation...")
        lane_manager.annotate_interactively(video_path)
        
    # 3. Initialize YOLO detector and SORT tracker
    if not os.path.exists(weights_path):
        logging.error(f"Weights not found at {weights_path}")
        return
        
    detector = VehicleDetector(model_path=weights_path)
    tracker = VehicleTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # 4. Open video stream once and initialize traffic analyzer
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video stream: {video_path}.")
        return

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0 or np.isnan(actual_fps):
        actual_fps = 25.0
        
    analyzer = TrafficAnalyzer(lane_manager=lane_manager, calibrator=calibrator, fps=actual_fps)

    window_name = "Adaptive Traffic Control - CV Pipeline"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    logging.info(f"Video FPS: {actual_fps:.2f}. Starting real-time processing. Press 'q' to stop.")

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
                
                if n > 0 or d > 0:
                    print(f"{lane_id}: N = {n} veh | L = {l:.2f} m | D = {d:.4f} ({d * 100:.1f}%)")
        
        # Visualization: Draw semi-transparent lanes
        display_frame = lane_manager.draw_lanes(frame, alpha=0.3)
        
        # Visualization: Draw bounding boxes and IDs
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj[0:5])
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                display_frame,
                f"ID:{track_id}",
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        # Visualization: Information panel for lane metrics (top-left)
        y_offset = 30
        cv2.putText(
            display_frame,
            "Lane Metrics (N, L, D):",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        
        for lane_id in metrics["N"].keys():
            n = metrics["N"][lane_id]
            l = metrics["L"][lane_id]
            d = metrics["D"][lane_id]
            if n > 0 or d > 0:
                y_offset += 30
                text = f"{lane_id}: N={n}, L={l:.1f}m, D={d:.3f}"
                cv2.putText(
                    display_frame,
                    text,
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
            
        cv2.imshow(window_name, display_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()