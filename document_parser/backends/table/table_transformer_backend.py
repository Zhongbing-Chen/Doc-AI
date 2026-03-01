"""
Table Transformer backend for table parsing
"""
from typing import Dict, Optional, List
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from document_parser.interfaces import TableParserBackend


class TableTransformerBackend(TableParserBackend):
    """
    Table Transformer backend for table parsing

    Uses Table Transformer model for detecting table structure.

    Note: This backend requires proper model setup including WiredTableRecognition
    configuration. If WiredTableRecognition is not available, use the VLM backend
    or disable table parsing.
    """

    def __init__(self):
        self.pipe = None
        self.table_cls = None
        self.wired_engine = None
        self.lineless_engine = None
        self.device = None
        self._initialized = False
        self._init_error = None

    def initialize(self, config: Dict) -> None:
        """
        Initialize Table Transformer

        Args:
            config: Configuration dictionary with keys:
                - device: 'cpu' or 'cuda'
                - model_source: 'huggingface', 'modelscope', or 'local'
                - model_path: Optional custom model path for local
        """
        device = config.get('device', 'cpu')
        model_source = config.get('model_source', 'huggingface')
        model_path = config.get('model_path')

        self.device = device

        try:
            from module.table.table_parser import TableExtractor
            from wired_table_rec.main import WiredTableInput, WiredTableRecognition

            # Create WiredTableInput config for WiredTableRecognition
            use_cuda = device == 'cuda'
            wired_config = WiredTableInput(
                model_type="unet",
                model_path=None,
                use_cuda=use_cuda,
                device=device
            )

            # Initialize TableExtractor components manually
            from huggingface_hub import snapshot_download as hf_download
            from table_cls import TableCls
            from model.table_transformer.inference import TableExtractionPipeline

            # Download model if needed
            if model_source == "huggingface":
                model_dir = hf_download("zhongbing/table-transformer-finetuned")
            elif model_source == "modelscope":
                from modelscope import snapshot_download
                model_dir = snapshot_download("zhongbing/table-transformer-finetuned")
            else:
                model_dir = model_path or "/Users/zhongbing/Projects/MLE/Doc-AI/model/table_transformer"

            # Initialize table classifier
            self.table_cls = TableCls(model_type="yolo", model_path=None)

            # Initialize wired table recognition with proper config
            self.wired_engine = WiredTableRecognition(wired_config)

            # Initialize table extraction pipeline
            self.pipe = TableExtractionPipeline(
                str_device=device,
                det_config_path=None,
                det_model_path=None,
                str_config_path=os.path.join(model_dir, "structure_config.json"),
                str_model_path=os.path.join(model_dir, 'model_20.pth'))

            self._initialized = True
            print(f"Table Transformer initialized on {device}")

        except ImportError as e:
            self._init_error = f"Missing dependency: {e}"
            print(f"Warning: Table Transformer initialization skipped - {self._init_error}")
        except Exception as e:
            self._init_error = str(e)
            print(f"Warning: Table Transformer initialization skipped - {self._init_error}")

    def parse(self, image: Image.Image, bbox: Optional[List[float]] = None, **kwargs) -> List:
        """
        Parse table structure

        Args:
            image: PIL Image containing the table
            bbox: Optional bounding box [x1, y1, x2, y2] to crop
            **kwargs: Additional parameters

        Returns:
            List of TableStructure objects representing table cells, or empty list if not initialized
        """
        # Return empty list if not initialized
        if not self._initialized:
            return []

        from entity.block import TableStructure
        from module.table.table_parser import TableExtractor

        # Crop image if bbox provided
        if bbox:
            image = image.crop(bbox)

        # Create temporary extractor instance to use its parse method
        temp_extractor = TableExtractor.__new__(TableExtractor)
        temp_extractor.pipe = self.pipe
        temp_extractor.table_cls = self.table_cls
        temp_extractor.wired_engine = self.wired_engine
        temp_extractor.lineless_engine = self.lineless_engine

        # Parse table
        table_structure = temp_extractor.parse(image)

        return table_structure

    def cleanup(self) -> None:
        """Cleanup model resources"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if self.table_cls is not None:
            del self.table_cls
            self.table_cls = None
        if self.wired_engine is not None:
            del self.wired_engine
            self.wired_engine = None
        if self.lineless_engine is not None:
            del self.lineless_engine
            self.lineless_engine = None
        self._initialized = False