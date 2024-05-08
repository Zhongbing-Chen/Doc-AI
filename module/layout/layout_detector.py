from ultralytics import YOLO


class LayoutDetector:
    def __init__(self, model_path, device):
        self.device = device
        self.model = YOLO(model_path)

    def detect(self, image, conf=0.1, iou=0.45):
        # Detect layout
        results = self.model.predict(image, conf=conf, iou=iou, device=self.device)
        return results[0]
