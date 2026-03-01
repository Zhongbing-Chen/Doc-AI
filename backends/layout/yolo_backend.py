"""
YOLO-based layout detection backend for standalone document parser
"""
from typing import Any, Dict
from PIL import Image

from ...interfaces import LayoutDetectorBackend


class YOLOLayoutBackend(LayoutDetectorBackend):
    """
    YOLO-based layout detection backend

    Uses YOLOv8 model for detecting layout blocks in documents.
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.confidence = 0.5
        self.iou_threshold = 0.45

    def initialize(self, config: Dict) -> None:
        """
        Initialize YOLO model

        Args:
            config: Configuration dictionary with keys:
                - device: 'cpu' or 'cuda'
                - model_source: 'huggingface', 'modelscope', or local path
                - model_path: Optional custom model path
                - confidence: Confidence threshold (default: 0.5)
                - iou_threshold: IOU threshold (default: 0.45)
        """
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        device = config.get('device', 'cpu')
        model_source = config.get('model_source', 'huggingface')
        model_path = config.get('model_path')
        self.confidence = config.get('confidence', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.45)

        # Determine model path
        if model_path:
            final_model_path = model_path
        elif model_source == "huggingface":
            final_model_path = hf_hub_download(
                repo_id="zhongbing/yolov8-layout",
                filename="best_backup.pt"
            )
        elif model_source == "modelscope":
            from modelscope import snapshot_download
            final_model_path = snapshot_download('zhongbing/yolov8-layout')
        else:
            final_model_path = model_source

        self.device = device
        self.model = YOLO(final_model_path)
        print(f"YOLO layout detector initialized on {device}")

    def detect(self, image: Image.Image, conf: float = None, iou: float = None, **kwargs) -> Any:
        """
        Detect layout using YOLO

        Args:
            image: PIL Image to process
            conf: Confidence threshold (overrides initialized value)
            iou: IOU threshold (overrides initialized value)
            **kwargs: Additional YOLO parameters

        Returns:
            ultralytics Results object containing detected boxes
        """
        conf = conf if conf is not None else self.confidence
        iou = iou if iou is not None else self.iou_threshold

        results = self.model.predict(image, conf=conf, iou=iou, device=self.device)
        return results[0]

    def cleanup(self) -> None:
        """Cleanup YOLO model resources"""
        if self.model is not None:
            del self.model
            self.model = None
