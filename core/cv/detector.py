from ultralytics import YOLO
from config import MODEL_WEIGHTS_PATH

class VehicleDetector:
    """
    Wraps the custom-trained YOLOv8n model (single class: 0 - 'vehicle').
    """
    def __init__(self, model_path = MODEL_WEIGHTS_PATH, conf_threshold: float = 0.25):
        """
        Initializes the YOLO model for the single 'vehicle' class.
        """
        self.model = YOLO(str(model_path))
        self.vehicle_classes = {0}
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        Runs inference on a single frame and returns detections as
        [x1, y1, x2, y2, conf, cls_id].
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    detections.append([x1, y1, x2, y2, conf, cls_id])
                    
        return detections