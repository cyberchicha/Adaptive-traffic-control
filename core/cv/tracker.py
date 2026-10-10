import numpy as np
from core.cv.sort import Sort

class VehicleTracker:
    """
    Wraps the SORT (Simple Online and Realtime Tracking) algorithm.
    Assigns and maintains consistent integer IDs for vehicles across video frames.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initializes the SORT tracker with specific parameters.
        """
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    def update(self, detections: list) -> np.ndarray:
        """
        Updates the tracker with new bounding box detections from YOLO.
        """
        # If no detections exist in the current frame, SORT requires an empty array of specific shape
        if len(detections) == 0:
            return self.tracker.update(np.empty((0, 5)))

        # SORT algorithm strictly expects the input format to be: [x1, y1, x2, y2, score]
        # Slicing [:5] to drop the cls_id, keeping only coordinates and confidence
        dets_for_sort = np.array([d[:5] for d in detections])
        
        # Updating the tracker and getting the results
        tracked_objects = self.tracker.update(dets_for_sort)
        
        return tracked_objects