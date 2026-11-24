import numpy as np
from collections import deque

class TrafficAnalyzer:
    """
    Aggregates data from Object Tracker, Lane Manager, and IPM Camera Calibrator
    to compute macroscopic traffic parameters per lane:
      - N: number of queued (stationary) vehicles
      - L: spatial queue length in meters (sum of vehicle lengths and valid gaps)
      - D: normalized lane occupancy / density in [0.0, 1.0]
    """
    def __init__(
            self,
            lane_manager, 
            calibrator, 
            fps: float = 25.0,
            queue_speed_threshold_kmh: float = 5.0,
            speed_window_sec: float = 0.5,
            min_speed_st: float = 0.2,
            avg_vehicle_length: float = 4.5,
            avg_vehicle_area: float = 8.5,
            max_queue_gap_meters: float = 6.0,
            real_distance_meters: float = 36.0,
            ):
        """
        Initializes the TrafficAnalyzer.
        """
        self.lane_manager = lane_manager
        self.calibrator = calibrator
        self.fps = max(1, int(fps))
        self.frame_count = 0

        self.queue_speed_threshold_kmh = queue_speed_threshold_kmh
        self.speed_window_frames = max(2, int(round(speed_window_sec * self.fps)))
        self.min_speed_st = min_speed_st
        self.avg_vehicle_length = avg_vehicle_length
        self.avg_vehicle_area = avg_vehicle_area
        self.max_queue_gap_meters = max_queue_gap_meters
        self.real_distance_meters = real_distance_meters

        # Computing reference trigger-line Y pixels via IPM for visualization compatibility
        _, py_0 = self.calibrator.world_to_pixel(0.0, 0.0)
        _, py_far = self.calibrator.world_to_pixel(0.0, self.real_distance_meters)
        self.y_start = int(min(py_0, py_far))
        self.y_end = int(max(py_0, py_far))

        # Precompute physical area (m^2) for every configured lane polygon via IPM
        self.lane_areas = self._compute_lane_areas()

        # Format: {track_id: {"history": deque[(t, wx, wy)], "speed_kmh": float | None}}
        self.vehicle_data = {}

    def _compute_lane_areas(self) -> dict:
        """
        Projects each lane polygon to real-world coordinates (meters) and computes
        its physical area using the shoelace formula.
        """
        areas = {}
        for lane_id, polygon in self.lane_manager.lanes.items():
            world_pts = np.array(
               [self.calibrator.pixel_to_world(float(x), float(y)) for x, y in polygon],
                dtype=np.float64,
            )
            if len(world_pts) < 3:
                areas[lane_id] = 0.0
                continue
            x = world_pts[:, 0]
            y = world_pts[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            areas[lane_id] = float(area)

        return areas

    @staticmethod
    def _get_centroid(x1, y1, x2, y2):
        """
        Returns bottom-center contact point of a bounding box on the road plane.
        """
        return int((x1 + x2) / 2), int(y2)

    def _compute_spatial_queue_length(self, y_positions: list) -> float:
        """
        Computes spatial queue length (meters) as the sum of physical vehicle lengths
        and inter-vehicle gaps that do not exceed max_queue_gap_meters.
        """
        n = len(y_positions)
        if n == 0:
            return 0.0

        sorted_y = sorted(y_positions)
        total_length = n * self.avg_vehicle_length

        for i in range(n - 1):
            center_dist = sorted_y[i + 1] - sorted_y[i]
            bumper_gap = max(0.0, center_dist - self.avg_vehicle_length)
            if bumper_gap <= self.max_queue_gap_meters:
                total_length += bumper_gap

        return round(total_length, 2)

    def process_frame(self, tracked_objects):
        """
        Processes tracked objects for the current frame and returns per-lane N, L, D metrics.
        """
        self.frame_count += 1
        current_time = self.frame_count / self.fps 

        lane_ids = list(self.lane_manager.lanes.keys())
        metrics = {
            "N": {lane_id: 0 for lane_id in lane_ids},    # Number of vehicles in queue (units)
            "L": {lane_id: 0.0 for lane_id in lane_ids},  # Queue Length (meters)
            "D": {lane_id: 0.0 for lane_id in lane_ids},   # Linear Density (vehicles/meter)
        }

        lane_vehicle_counts = {lane_id: 0 for lane_id in lane_ids}
        queued_y_positions = {lane_id: [] for lane_id in lane_ids}
        current_ids = set()

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj[0:5])
            current_ids.add(track_id)

            cx, cy = self._get_centroid(x1, y1, x2, y2)
            lane_id = self.lane_manager.get_vehicle_lane(cx, cy)
            if not lane_id:
                continue

            lane_vehicle_counts[lane_id] += 1

            # Projecting bottom-center contact point to road plane (meters) via IPM
            wx, wy = self.calibrator.pixel_to_world(cx, cy)

            vehicle = self.vehicle_data.setdefault(
                track_id,
                {
                    "history": deque(maxlen=self.speed_window_frames),
                    "speed_kmh": None,
                },
            )
            history = vehicle["history"]
            history.append((current_time, wx, wy))

            # Instantaneous speed over sliding window
            t0, x0, y0 = history[0]
            dt = current_time - t0
            if dt >= self.min_speed_st - 1e-6:
                dist_meters = float(np.hypot(wx - x0, wy - y0))
                live_speed = (dist_meters / dt) * 3.6               # Converting m/s to km/h

                # Reject unrealistic tracking jitter spikes
                if live_speed < 200.0:
                    if vehicle["speed_kmh"] is None:
                        vehicle["speed_kmh"] = live_speed
                    else:
                        vehicle["speed_kmh"] = vehicle["speed_kmh"] * 0.6 + live_speed * 0.4

            current_speed = vehicle["speed_kmh"]
            if current_speed is not None and current_speed <= self.queue_speed_threshold_kmh:
                metrics["N"][lane_id] += 1
                queued_y_positions[lane_id].append(wy)

            # Computing spatial queue length L (m) and normalized occupancy density D in [0, 1]
            for lane_id in lane_ids:
                metrics["L"][lane_id] = self._compute_spatial_queue_length(queued_y_positions[lane_id])

                lane_area = self.lane_areas.get(lane_id, 0.0)
                if lane_area > 0.0:
                    occupied_area = lane_vehicle_counts[lane_id] * self.avg_vehicle_area
                    metrics["D"][lane_id] = round(min(1.0, occupied_area / lane_area), 4)
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