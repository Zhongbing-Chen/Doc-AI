from huggingface_hub import hf_hub_download
from modelscope.hub.file_download import model_file_download
from ultralytics import YOLO
from ultralytics.engine.results import Results


class LayoutDetector:
    """
    The layout detector class

    Attributes:
        model: the layout detection model
        device: the device to run the model
    """

    type_dict = {0: 'Text', 1: 'Title', 2: 'Header', 3: 'Footer', 4: 'Figure', 5: 'Table', 6: 'Toc',
                 7: 'Figure caption', 8: 'Table caption'}

    # type_dict = {0: 'title', 1: 'plain text', 2: 'abandon', 3: 'figure', 4: 'figure_caption', 5: 'table', 6: 'table_caption', 7: 'table_footnote', 8: 'isolate_formula', 9: 'formula_caption'}

    def __init__(self, model_source, device):
        """
        Initialize the layout detector
        :param model_source: the model source, could be "huggingface" or "modelscope"
        :param device: the device to run the model
        """
        if model_source == "huggingface":
            model_path = hf_hub_download(repo_id="zhongbing/yolov8-layout", filename="best_backup.pt", )
        elif model_source == "modelscope":
            model_path = model_file_download("zhongbing/yolov8-layout-cdla-plus", file_path="best_backup.pt")
        else:
            model_path = model_source

        self.device = device
        self.model = YOLO(model_path)

    def detect(self, image, conf=0.05, iou=0.45) -> Results:
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
