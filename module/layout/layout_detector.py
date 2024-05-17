from ultralytics import YOLO
from ultralytics.engine.results import Results


class LayoutDetector:
    """
    The layout detector class

    Attributes:
        model: the layout detection model
        device: the device to run the model
    """

    def __init__(self, model_path, device):
        self.device = device
        self.model = YOLO(model_path)

    def detect(self, image, conf=0.1, iou=0.45) -> Results:
        """
        Detect the layout of the image
        :param image: the image
        :param conf: the confidence threshold
        :param iou: the iou threshold
        :return: the layout result
        """
        # Detect layout
        results = self.model.predict(image, conf=conf, iou=iou, device=self.device)
        return results[0]
