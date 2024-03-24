from ultralytics import YOLO


class LayoutDetector:
    def __init__(self, model_path):
        self.model = YOLO(
            "/home/zhongbing/Projects/MLE/Document-AI/yolo-document-layout-analysis/layout_analysis/8mpt/v2/best.pt")

    def detect(self, image):
        # Detect layout
        results = self.model.predict(image, conf=0.1, iou=0.45)
        return results[0]
