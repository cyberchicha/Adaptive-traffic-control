import numpy as np
import cv2

class TrafficAnalyzer:
    """
    Aggregates data from the Object Tracker, Lane Manager, and Camera Calibrator.
    Implements a hybrid "Live Preview + Final Lock" speed trap logic. 
    It calculates smooth, real-time dynamic speed while the vehicle is in the zone, 
    and locks a highly precise cumulative average speed upon exit.
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
        self.wait_times = {}

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
            "queue_lengths": {lane_id: 0 for lane_id in self.lane_manager.lanes.keys()},
            "speeds": {},
            "wait_times": {}
        }
        
        current_ids = set()

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            current_ids.add(track_id)
            
            cx, cy = self._get_centroid(x1, y1, x2, y2)
            lane_id = self.lane_manager.get_vehicle_lane(cx, cy)
            
            if lane_id:
                metrics["queue_lengths"][lane_id] += 1
            
            vehicle = self.vehicle_data.setdefault(track_id, {
                "start_time": None, "entry_dist": None, "speed_kmh": None, 
                "active": False, "locked": False
            })
            
            in_zone = self._is_inside_zone(cy)
            
            # STAGE 1: Vehicle enters the zone
            if in_zone and not vehicle["active"] and not vehicle["locked"]:
                vehicle["start_time"] = current_video_time
                # Get precise entry distance using interpolation
                vehicle["entry_dist"] = float(np.interp(cy, self.y_arr, self.dist_arr))
                vehicle["active"] = True
                
            # STAGE 2: Vehicle is driving inside the zone (LIVE PREVIEW)
            elif in_zone and vehicle["active"] and not vehicle["locked"]:
                time_diff = current_video_time - vehicle["start_time"]
                
                # Wait 0.4 seconds before showing live speed to prevent jitter
                if time_diff > 0.4:
                    current_dist = float(np.interp(cy, self.y_arr, self.dist_arr))
                    dist_covered = abs(current_dist - vehicle["entry_dist"])
                    
                    live_speed = (dist_covered / time_diff) * 3.6
                    
                    if live_speed < 250.0:
                        if vehicle["speed_kmh"] is None:
                            vehicle["speed_kmh"] = live_speed
                        else:
                            # Apply EMA smoothing (85% old, 15% new) for visual stability
                            vehicle["speed_kmh"] = vehicle["speed_kmh"] * 0.85 + live_speed * 0.15
                            
            # STAGE 3: Vehicle exits the zone (FINAL LOCK)
            elif not in_zone and vehicle["active"] and not vehicle["locked"]:
                vehicle["active"] = False
                time_diff = current_video_time - vehicle["start_time"]
                
                if time_diff > 0:
                    # Use the EXACT 36.0m theoretical distance for maximum precision
                    final_speed = (self.real_distance_meters / time_diff) * 3.6
                    if final_speed < 250.0:
                        vehicle["speed_kmh"] = final_speed
                        vehicle["locked"] = True
            
            # OUTPUT CALCULATION
            if vehicle["speed_kmh"] is not None:
                speed_val = round(vehicle["speed_kmh"], 2)
                metrics["speeds"][track_id] = speed_val
                
                # Jam detection penalty
                if speed_val < 5.0 and lane_id:
                    if track_id not in self.wait_times:
                        self.wait_times[track_id] = current_video_time
                    metrics["wait_times"][track_id] = current_video_time - self.wait_times[track_id]
                else:
                    if track_id in self.wait_times:
                        del self.wait_times[track_id]
                        
        self._cleanup_lost_tracks(current_ids)
        return metrics

    def _cleanup_lost_tracks(self, current_ids):
        """
        Clears memory for tracked objects that left the camera view.
        """
        lost_ids = set(self.vehicle_data.keys()) - current_ids
        for track_id in lost_ids:
            del self.vehicle_data[track_id]
            if track_id in self.wait_times:
                del self.wait_times[track_id]