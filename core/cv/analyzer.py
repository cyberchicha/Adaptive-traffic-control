import numpy as np
import cv2

class TrafficAnalyzer:
    """
    Aggregates data from Object Tracker and Camera Calibrator to compute 
    macroscopic traffic parameters for RL Agent state vector (St).
    Computes Queue Count (N), Queue Length (L), and Linear Density (D) per lane.
    Speed is calculated internally strictly to classify stationary/moving vehicles.
    """
    def __init__(self, lane_manager, calibrator, fps: int = 25):
        """
        Initializes the TrafficAnalyzer.
        """
        self.lane_manager = lane_manager
        self.calibrator = calibrator
        self.fps = fps
        self.frame_count = 0 
        
        # Virtual distance between the two trigger lines in meters
        self.real_distance_meters = 36.0 
        
        # PRECOMPUTE PERFECT DEPTH MAPPING
        # Projecting 100 physical distances (from 0m to 36m) into 2D Y-pixels
        distances = np.linspace(0.0, self.real_distance_meters, 100)
        y_pixels = []
        
        for d in distances:
            pt_3d = np.array([[0.0, float(d), 0.0]], dtype=np.float32)
            img_pt, _ = cv2.projectPoints(
                pt_3d, self.calibrator.rvec, self.calibrator.tvec, 
                self.calibrator.intrinsic, self.calibrator.dist_coeffs
            )
            y_pixels.append(img_pt[0][0][1])
            
        y_pixels = np.array(y_pixels)
        
        # Sorting arrays to ensure np.interp works correctly (requires increasing X values)
        sort_idx = np.argsort(y_pixels)
        self.y_arr = y_pixels[sort_idx]
        self.dist_arr = distances[sort_idx]
        
        # Setting top and bottom physical limits for the measurement zone
        self.y_start = int(min(y_pixels))
        self.y_end = int(max(y_pixels))
        
        # Format: {track_id: {'start_time': t, 'entry_dist': d, 'speed_kmh': float, 'active': bool, 'locked': bool}}
        self.vehicle_data = {}

    def _get_centroid(self, x1, y1, x2, y2):
        """
        Returns bottom-center centroid of a bounding box.
        """
        return int((x1 + x2) / 2), int(y2)

    def _is_inside_zone(self, cy):
        """
        Checks if vehicle is between the trigger lines.
        """
        return self.y_start <= cy <= self.y_end

    def process_frame(self, tracked_objects):
        self.frame_count += 1
        current_video_time = self.frame_count / self.fps 
        
        metrics = {
            "N": {lane_id: 0 for lane_id in self.lane_manager.lanes.keys()},    # Number of vehicles in queue (units)
            "L": {lane_id: 0.0 for lane_id in self.lane_manager.lanes.keys()},  # Queue Length (meters)
            "D": {lane_id: 0.0 for lane_id in self.lane_manager.lanes.keys()}   # Linear Density (vehicles/meter)
        }
        
        current_ids = set()

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            current_ids.add(track_id)
            
            cx, cy = self._get_centroid(x1, y1, x2, y2)
            lane_id = self.lane_manager.get_vehicle_lane(cx, cy)
            
            if not lane_id:
                continue
            
            vehicle = self.vehicle_data.setdefault(track_id, {
                "start_time": None, "entry_dist": None, "speed_kmh": None, "active": False
            })
            
            in_zone = self._is_inside_zone(cy)
            current_dist = float(np.interp(cy, self.y_arr, self.dist_arr))
            
            # INTERNAL SPEED CALCULATION
            if in_zone and not vehicle["active"]:
                vehicle["start_time"] = current_video_time
                vehicle["entry_dist"] = current_dist
                vehicle["active"] = True
                
            elif in_zone and vehicle["active"]:
                time_diff = current_video_time - vehicle["start_time"]
                if time_diff > 0.4:
                    dist_covered = abs(current_dist - vehicle["entry_dist"])
                    live_speed = (dist_covered / time_diff) * 3.6
                    
                    if live_speed < 250.0:
                        # EMA speed smoothing
                        vehicle["speed_kmh"] = vehicle["speed_kmh"] * 0.85 + live_speed * 0.15
            
            elif not in_zone and vehicle["active"]:
                vehicle["active"] = False

            # FORMATION OF METRICS N(t) AND L(t)
            # If the vehicle is moving slower than 5 km/h, it is considered part of the queue
            # Safeguard: if the speed is None, set it to the default of 30 km/h.
            current_speed = vehicle.get("speed_kmh")
            if current_speed is None:
                current_speed = 30.0
                vehicle["speed_kmh"] = current_speed
                
            # If a vehicle is moving slower than 5 km/h, it is considered part of the queue.
            is_queued = current_speed < 5.0
            
            if is_queued:
                metrics["N"][lane_id] += 1
                
                # Length of the queue L(t) is defined as the distance to the farthest vehicle
                if current_dist > metrics["L"][lane_id]:
                    metrics["L"][lane_id] = round(current_dist, 2)
                        
        # FORMATION OF THE D(t) METRIC
        for lane_id in metrics["N"].keys():
            n = metrics["N"][lane_id]
            l = metrics["L"][lane_id]
            
            # Calculation of linear density: D = N / L (vehicles/m)
            if l > 0:
                metrics["D"][lane_id] = round(n / l, 4)
            else:
                metrics["D"][lane_id] = 0.0
                        
        self._cleanup_lost_tracks(current_ids)
        return metrics

    def _cleanup_lost_tracks(self, current_ids):
        """
        Clears memory for tracked objects that left the camera view.
        """
        lost_ids = set(self.vehicle_data.keys()) - current_ids
        for track_id in lost_ids:
            del self.vehicle_data[track_id]