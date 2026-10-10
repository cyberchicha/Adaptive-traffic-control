import numpy as np
from collections import deque
import time

class TrafficAnalyzer:
    """
    Aggregates data from the Object Tracker, Lane Manager, and Camera Calibrator.
    Calculates business metrics needed for traffic light control (SUMO/RL integration):
    queue lengths, vehicle speeds in real-world metrics, and wait times in traffic jams.
    """

    def __init__(self, lane_manager, calibrator, fps: int = 30):
        """
        :param lane_manager: Instance of LaneManager to determine vehicle positions in lanes.
        :param calibrator: Instance of CameraCalibrator to transform pixels to meters.
        :param fps: Assumed frames per second for track history buffering.
        """
        self.lane_manager = lane_manager
        self.calibrator = calibrator
        self.fps = fps
        
        # Buffer to store recent real-world positions for speed calculation.
        # Format: {track_id: deque([(world_x, world_y, timestamp), ...])}
        self.track_history = {}
        
        # Dictionary to track how long a vehicle has been stationary (Wait Time).
        # Format: {track_id: timestamp_when_stopped}
        self.wait_times = {}

    def _get_centroid(self, x1, y1, x2, y2):
        """
        Calculates the centroid of a bounding box. 
        Uses the bottom-center point as it best represents where the vehicle touches the road.
        """
        return int((x1 + x2) / 2), int(y2)

    def process_frame(self, tracked_objects):
        """
        Processes a single frame's tracked objects to extract analytical metrics.
        """
        metrics = {
            "queue_lengths": {lane_id: 0 for lane_id in self.lane_manager.lanes.keys()},
            "speeds": {},
            "wait_times": {}
        }
        
        current_time = time.time()
        current_ids = set()

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            current_ids.add(track_id)
            
            # 1. Spatial Analysis: Determining which lane the vehicle is in
            cx, cy = self._get_centroid(x1, y1, x2, y2)
            lane_id = self.lane_manager.get_vehicle_lane(cx, cy)
            
            if lane_id:
                metrics["queue_lengths"][lane_id] += 1
            
            # 2. Speed Estimation: Transform centroid to 3D world coordinates
            world_x, world_y = self.calibrator.pixel_to_world(cx, cy)
            
            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=self.fps)
            
            self.track_history[track_id].append((world_x, world_y, current_time))
            
            # Require at least 5 frames of history to calculate a stable speed
            if len(self.track_history[track_id]) > 5:
                # Retrieving the oldest and newest point in the history buffer
                x_start, y_start, t_start = self.track_history[track_id][0]
                x_end, y_end, t_end = self.track_history[track_id][-1]
                
                # Calculating Euclidean distance in real-world meters
                distance = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
                time_diff = t_end - t_start
                
                if time_diff > 0:
                    speed_ms = distance / time_diff
                    speed_kmh = speed_ms * 3.6
                    metrics["speeds"][track_id] = speed_kmh
                    
                    # 3. Wait Time Calculation: Penalize RL Agent if a car is in a lane but not moving
                    # Threshold: speeds below 5 km/h are considered "stationary in a traffic jam"
                    if speed_kmh < 5.0 and lane_id:
                        if track_id not in self.wait_times:
                            self.wait_times[track_id] = current_time
                        metrics["wait_times"][track_id] = current_time - self.wait_times[track_id]
                    else:
                        # Vehicle is moving; reset wait time penalty
                        if track_id in self.wait_times:
                            del self.wait_times[track_id]
        
        self._cleanup_lost_tracks(current_ids)
        return metrics

    def _cleanup_lost_tracks(self, current_ids):
        """
        Removes history and wait times for IDs no longer present in the frame.
        """
        lost_ids = set(self.track_history.keys()) - current_ids
        for track_id in lost_ids:
            del self.track_history[track_id]
            if track_id in self.wait_times:
                del self.wait_times[track_id]