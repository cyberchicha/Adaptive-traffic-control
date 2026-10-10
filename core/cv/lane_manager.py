import cv2
import numpy as np
import json
import os
from config import LANES_CONFIG_PATH

class LaneManager:
    """
    Manages the geometry of traffic lanes using polygons.
    Provides functionality to add lanes, draw them for debugging, 
    and determine which lane a tracked vehicle currently occupies.
    """
    def __init__(self, config_path=LANES_CONFIG_PATH):
        """
        Initializes the LaneManager instance.
        Creates an empty dictionary to store lane polygons.
        """
        self.lanes = {}
        self.config_path = str(config_path)
        self.color_palette = [
            (0, 255, 0),        # Green
            (255, 0, 0),        # Blue
            (0, 255, 255),      # Yellow
            (255, 0, 255),      # Magenta
            (0, 165, 255),      # Orange
            (255, 255, 0)       # Cyan
        ]

    def add_lane(self, lane_id: str, points: list):
        """
        Adds a new lane polygon to the manager.
        Converts the list of coordinate points into a NumPy array and stores it.
        """
        polygon = np.array(points, np.int32)
        # Storing the polygon array in the dictionary using the lane_id as the key
        self.lanes[lane_id] = polygon
        print(f"[INFO] Added lane '{lane_id}' with {len(points)} points.")

    def get_vehicle_lane(self, cx: int, cy: int):
        """
        Determines which lane a specific vehicle center point belongs to.
        Uses OpenCV's pointPolygonTest to check point enclosure.
        """
        for lane_id, polygon in self.lanes.items():
            # Using OpenCV's pointPolygonTest to check if the point (cx, cy) is inside the polygon
            if cv2.pointPolygonTest(polygon, (cx, cy), measureDist=False) >= 0:
                return lane_id
        return None

    def draw_lanes(self, frame, alpha=0.3):
        """
        Draws the defined lane polygons onto a given video frame.
        Applies a semi-transparent colored overlay for visualization.
        """
        overlay = frame.copy()

        for i, (lane_id, polygon) in enumerate(self.lanes.items()):
            color = self.color_palette[i % len(self.color_palette)]

            cv2.polylines(overlay, [polygon], isClosed=True, color=color, thickness=2)
            cv2.fillPoly(overlay, [polygon], color=color)
            
            x, y = polygon[0]
            cv2.putText(overlay, lane_id, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def save_lanes(self):
        """
        Saves current lane polygons to a JSON file.
        """
        # JSON serialization
        data = {lane_id: polygon.tolist() for lane_id, polygon in self.lanes.items()}
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            # Serializing the data dictionary to JSON format and write it to the file with a 4-space indent
            json.dump(data, f, indent=4)
        print(f"[SUCCESS] Saved {len(self.lanes)} lanes to {self.config_path}")

    def load_lanes(self):
        """
        Loads lane polygons from a JSON configuration file.
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                for lane_id, points in data.items():
                    self.add_lane(lane_id, points)
            print(f"[SUCCESS] Loaded {len(self.lanes)} lanes from {self.config_path}")
            return True
        else:
            print(f"[WARNING] Config file {self.config_path} not found. Needs annotation.")
            return False

    def annotate_interactively(self, video_path):
        """
        Opens a GUI window to manually click and draw lane polygons on the first video frame.
        Allows the user to define multiple lanes and save them to a JSON configuration file.
        """
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise Exception("Failed to read video")

        current_points = []
        lane_counter = len(self.lanes) + 1
        window_name = "Lane Annotation Tool"

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 768)

        # Defining an internal callback function to handle mouse events within the window
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                current_points.append([x, y])
                print(f"[INFO] Point added: ({x}, {y})")

        cv2.setMouseCallback(window_name, mouse_callback)

        print("1. Click to draw points around the border of a single lane.")
        print("2. Press 'n' (Next) to save the current lane and start the next one.")
        print("3. Press 's' (Save) to save all lanes to JSON and exit.")
        print("4. Press 'q' (Quit) to exit without saving.") 

        while True:
            temp_frame = frame.copy()
            self.draw_lanes(temp_frame, alpha=0.4)

            # Checking if the user has started drawing a new polygon
            if len(current_points) > 0:
                pts = np.array(current_points, np.int32)
                cv2.polylines(temp_frame, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
                for pt in current_points:
                    cv2.circle(temp_frame, tuple(pt), 4, (0, 0, 255), -1)

            cv2.imshow(window_name, temp_frame)
            key = cv2.waitKey(1) & 0xFF

            # Check if the 'n' key was pressed
            if key == ord('n'):
                if len(current_points) >= 3:
                    self.add_lane(f"lane_{lane_counter}", current_points)
                    current_points = []
                    lane_counter += 1
                else:
                    print("[WARNING] You need at least 3 points to form a lane polygon.")
            # Checking if the 's' key was pressed
            elif key == ord('s'):
                if len(current_points) >= 3:
                    self.add_lane(f"lane_{lane_counter}", current_points)
                self.save_lanes()
                break
            # Checking if the 'q' key was pressed
            elif key == ord('q'):
                print("[INFO] Annotation cancelled.")
                break

        cv2.destroyAllWindows()