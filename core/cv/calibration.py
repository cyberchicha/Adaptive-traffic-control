import cv2
import numpy as np
import json
import os

class CameraCalibration:
    """
    Maps 2D video pixels to 3D real-world coordinates (in meters) using solvePnP.
    This enables the system to measure the physical length of vehicle queues.
    """
    def __init__(self, video_path):
        self.video_path = video_path
        self.image = None
        self.image_points = []                      # Stores 2D pixel coordinates clicked by the user
        self.object_points = []                     # Stores the corresponding 3D real-world coordinates
        self.intrinsic = None                       # Intrinsic camera matrix variable
        self.dist_coeffs = np.zeros((4, 1))         # Distortion coefficients as a 4x1 array of zeros
        self.rvec = None                            # Rotation vector variable
        self.tvec = None                            # Translation vector variable
        self.window_name = "Calibration Window"

    def init_window(self):
        """
        Initializes the OpenCV window for interactive point selection.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)

    def load_first_frame(self):
        """
        Loads the first frame from the video to be used for calibration.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise Exception("Failed to open video file")
        
        # Reading the first frame from the video stream
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise Exception("Failed to read frame from video")
        
        # Creating an independent copy of the read frame and store it in the class attribute
        self.image = frame.copy()
        print("[INFO] First frame loaded successfully!")
        return True

    def collect_2d_points(self):
        """
        Opens a GUI window to collect 2D image points via mouse clicks.
        """
        if self.image is None:
            raise Exception("Image not loaded!")
        
        temp_image = self.image.copy()
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
        print(f"[INFO] Collected {len(self.image_points)} 2D points")

    def collect_3d_points(self):
        """
        Prompts the user to input corresponding 3D world coordinates for the collected 2D points.
        """
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
        Calculates the camera pose (rotation and translation vectors) using solvePnP.
        """
        if len(self.image_points) < 4:
            raise Exception("Minimum 4 points required for calibration")
        
        # Extracting the height and width of the image
        h, w = self.image.shape[:2]
        
        # Constructing an approximate intrinsic camera matrix using the image dimensions
        self.intrinsic = np.array([
            [w, 0, w/2],                # First row: approximate focal length X (equal to width) and center X
            [0, h, h/2],                # Second row: approximate focal length Y (equal to height) and center Y
            [0, 0, 1]                   # Third row: standard [0, 0, 1] for homogeneous coordinates
        ], dtype=np.float32)
        
        # Calling solvePnP to estimate the rotation and translation vectors
        ret, rvec, tvec = cv2.solvePnP(
            self.object_points,
            self.image_points,
            self.intrinsic,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # Checking if the solvePnP algorithm failed to find a valid solution
        if not ret:
            raise Exception("Calibration failed")

        self.rvec = rvec
        self.tvec = tvec

        print("[INFO] Calibration successful!")
        print("Rotation Vector:\n", rvec)
        print("Translation Vector:\n", tvec)

    def pixel_to_world(self, u, v):
        """
        Converts 2D pixel coordinates (u, v) into real-world 3D coordinates (X, Y)
        on the road plane (assuming Z=0).
        
        :param u: X-coordinate of the pixel
        :param v: Y-coordinate of the pixel
        :return: Tuple (X, Y) in meters
        """
        if self.rvec is None or self.tvec is None:
            raise Exception("Camera is not calibrated yet")

        # Converting the 3x1 rotation vector into a 3x3 rotation matrix
        R, _ = cv2.Rodrigues(self.rvec)
        
        K_inv = np.linalg.inv(self.intrinsic)       # Calculating the mathematical inverse of the intrinsic camera matrix
        R_inv = np.linalg.inv(R)                    # Calculating the mathematical inverse of the 3x3 rotation matrix
        
        # Creating a 3x1 column vector representing the pixel in homogeneous coordinates (u, v, 1)
        uv_point = np.array([[u], [v], [1.0]])
        
        # Projection formula
        left_side = R_inv.dot(K_inv).dot(uv_point)
        right_side = R_inv.dot(self.tvec)
        
        # Calculating the scaling factor 's' to force the world Z-coordinate to be exactly 0 (ground level)
        s = right_side[2][0] / left_side[2][0]
        
        # Calculate the final 3D world coordinates
        world_point = s * left_side - right_side
        
        return float(world_point[0][0]), float(world_point[1][0])

    def save_calibration(self, config_path="data/configs/calibration.json"):
        """
        Saves 2D and 3D points to a JSON file.
        """
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        data = {
            "points_2d": np.array(self.image_points).tolist(),
            "points_3d": np.array(self.object_points).tolist()
        }
        
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"[SUCCESS] Calibration saved to {config_path}")

    def load_calibration(self, config_path="data/configs/calibration.json"):
        """
        Loads points from JSON and automatically computes the perspective matrix.
        """
        if not os.path.exists(config_path):
            return False
            
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                
            self.image_points = np.array(data["points_2d"], dtype=np.float32)
            self.object_points = np.array(data["points_3d"], dtype=np.float32)
            
            self.calibrate()
            print(f"[SUCCESS] Loaded calibration from {config_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load calibration: {e}")
            return False