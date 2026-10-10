from ultralytics import YOLO
from config import MODEL_WEIGHTS_PATH

class VehicleDetector:
    """
    Wraps the custom trained YOLO object detection model.
    Its sole responsibility is to process a video frame, detect 'vehicle' objects, 
    and return raw bounding box coordinates.
    """
    def __init__(self, model_path = MODEL_WEIGHTS_PATH):
        """
        Initializes the YOLO model for the single 'vehicle' class.
        """
        self.model = YOLO(str(model_path))
        self.vehicle_classes = [0]

    def detect(self, frame):
        """
        Runs inference on a single frame and extracts vehicle detections.
        """
        results = self.model(frame, verbose=False)
        detections = []
        
        # Iterating through the results (one result object per frame)
        for r in results:
            # Extracting the bounding boxes object
            boxes = r.boxes
            
            # Iterating through each individual detected box in the frame
            for box in boxes:
                # Extracting the class ID as an integer
                cls_id = int(box.cls[0])
                
                # Checking if the detected object is target 'vehicle' class (id 0)
                if cls_id in self.vehicle_classes:
                    # Extracting the top-left (x1, y1) and bottom-right (x2, y2) coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Extracting the confidence score of the detection as a float
                    conf = float(box.conf[0])

                    # Appending the collected data as a list to detections list
                    detections.append([x1, y1, x2, y2, conf, cls_id])
                    
        return detections