import cv2
import numpy as np
import json
import os
from config import CALIBRATION_CONFIG_PATH

class CameraCalibration:
    """
    Maps 2D video pixels to 2D real-world road-plane coordinates (X, Y in meters)
    using Inverse Perspective Mapping (IPM) via a 3x3 homography matrix H.
    """
    def __init__(self, video_path = None):
        self.video_path = str(video_path) if video_path is not None else None
        self.image = None
        self.image_points = []                      # 2D pixel coordinates (u, v)
        self.object_points = []                     # 3D/2D road-plane coordinates in meters (X, Y, Z=0)
        self.homography = None
        self.inv_homography = None
        self.window_name = "Calibration Window"

    def init_window(self):
        """
        Initializes the OpenCV window for interactive point selection.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)

    def load_first_frame(self):
        """
        Loads the first frame from the video to be used for interactive calibration.
        """
        if not self.video_path:
            raise ValueError("Video path is not specified.")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")
        
        # Reading the first frame from the video stream
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError("Failed to read frame from video.")
        
        # Creating an independent copy of the read frame and store it in the class attribute
        self.image = frame.copy()
        print("[INFO] First frame loaded successfully!")
        return True

    def collect_2d_points(self):
        """
        Opens a GUI window to collect 2D image points via mouse clicks.
        """
        if self.image is None:
            raise RuntimeError("Image not loaded! Call load_first_frame() first.")
        
        temp_image = self.image.copy()
        self.image_points = []
        self.init_window()
        cv2.imshow(self.window_name, temp_image)

        # Defining an internal nested function to handle mouse callback events
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"[INFO] Added point: ({x}, {y})")
                self.image_points.append([x, y])
                # Drawing a filled green circle of radius 5 at the clicked location
                cv2.circle(temp_image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow(self.window_name, temp_image)

        # Registering the click_event function as the mouse callback for the active window
        cv2.setMouseCallback(self.window_name, click_event)
        print("[INSTRUCTION] Click on the image to add points. Press 'w' to finish.")

        # Starting an infinite loop to keep the GUI window open and responsive
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('w'):
                break
            cv2.imshow(self.window_name, temp_image)

        cv2.destroyAllWindows()
        # Converting the collected Python list of 2D points into a NumPy array of 32-bit floats
        self.image_points = np.array(self.image_points, dtype=np.float32)
        print(f"[INFO] Collected {len(self.image_points)} 2D points.")

    def collect_3d_points(self):
        """
        Prompts the user to input corresponding 3D world coordinates for the collected 2D points.
        """
        self.object_points = []
        print(f"[INSTRUCTION] Enter {len(self.image_points)} 3D points (x y z in meters):")
        for i in range(len(self.image_points)):
            while True:
                try:
                    # Prompting the user for input and split the resulting string by spaces
                    coords = input(f"Point {i+1}: ").split()
                    # Converting the list of string inputs into three float variables (x, y, z)
                    x, y, z = map(float, coords)
                    self.object_points.append([x, y, z])
                    break
                except ValueError:
                    print("[ERROR] Invalid format. Enter three numbers separated by spaces.")
        
        # Converting the collected Python list of 3D points into a NumPy array of 32-bit floats
        self.object_points = np.array(self.object_points, dtype=np.float32)

    def calibrate(self):
        """
        Computes the 3x3 homography matrix H mapping 2D image pixels to
        2D road-plane coordinates (X, Y) in meters (assuming planar road Z = 0).
        """
        src_pts = np.asarray(self.image_points, dtype=np.float32)
        obj_pts = np.asarray(self.object_points, dtype=np.float32)

        if len(src_pts) < 4 or len(obj_pts) < 4:
            raise ValueError("Minimum 4 points required for homography calibration.")
        if len(src_pts) != len(obj_pts):
            raise ValueError("Number of 2D image points must match the number of 3D world points.")

        # Using planar road coordinates (X, Y), ignoring Z-coordinate for homography computation
        dst_pts = obj_pts[:, :2]

        H, _ = cv2.findHomography(src_pts, dst_pts, method=0)
        if H is None:
            raise RuntimeError("Homography computation failed.")

        self.homography = H.astype(np.float64)
        self.inv_homography = np.linalg.inv(self.homography)

        print("[INFO] IPM Homography calibration successfully!")

    def pixel_to_world(self, u: float, v: float):
        """
        Converts 2D pixel coordinates (u, v) into real-world 3D coordinates (X, Y)
        on the road plane in meters using the homography matrix H.
        """
        if self.homography is None:
            raise RuntimeError("Camera is not calibrated yet.")

        pt = np.array([[[float(u), float(v)]]], dtype=np.float64)
        world_pt = cv2.perspectiveTransform(pt, self.homography)

        return float(world_pt[0][0][0]), float(world_pt[0][0][1])

    def world_to_pixel(self, x: float, y: float):
        """
        Converts real-world road-plane coordinates (X, Y) in meters back to
        2D pixel coordinates (u, v).
        """
        if self.inv_homography is None:
            raise RuntimeError("Camera is not calibrated yet.")

        pt = np.array([[[float(x), float(y)]]], dtype=np.float64)
        pixel_pt = cv2.perspectiveTransform(pt, self.inv_homography)

        return float(pixel_pt[0][0][0]), float(pixel_pt[0][0][1])

    def save_calibration(self, config_path="data/configs/calibration.json"):
        """
        Saves 2D and 3D calibration points to a JSON file.
        """
        config_path = str(config_path)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        data = {
            "points_2d": np.asarray(self.image_points).tolist(),
            "points_3d": np.asarray(self.object_points).tolist()
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"[SUCCESS] Calibration saved to {config_path}.")

    def load_calibration(self, config_path="data/configs/calibration.json"):
        """
        Loads points from JSON and computes the IPM homography matrix.
        """
        config_path = str(config_path)
        if not os.path.exists(config_path):
            return False
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.image_points = np.array(data["points_2d"], dtype=np.float32)
            self.object_points = np.array(data["points_3d"], dtype=np.float32)
            
            self.calibrate()
            print(f"[SUCCESS] Loaded calibration from {config_path}.")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load calibration: {e}.")
            return False