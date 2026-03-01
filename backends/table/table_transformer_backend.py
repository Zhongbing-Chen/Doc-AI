"""
Table Transformer backend for table parsing
"""
from typing import Dict, Optional, List
from PIL import Image

from ...interfaces import TableParserBackend
from ...entity.block import TableStructure


class TableTransformerBackend(TableParserBackend):
    """
    Table Transformer backend for table parsing

    Uses Table Transformer model for detecting table structure.
    Wraps the existing TableExtractor implementation.
    """

    def __init__(self):
        self.pipe = None
        self.table_cls = None
        self.wired_engine = None
        self.lineless_engine = None
        self.device = None

    def initialize(self, config: Dict) -> None:
        """
        Initialize Table Transformer

        Args:
            config: Configuration dictionary with keys:
                - device: 'cpu' or 'cuda'
                - model_source: 'huggingface', 'modelscope', or 'local'
                - model_path: Optional custom model path for local
        """
        from ...module.table.table_parser import TableExtractor

        device = config.get('device', 'cpu')
        model_source = config.get('model_source', 'huggingface')
        model_path = config.get('model_path')

        self.device = device

        # Use existing TableExtractor implementation
        if model_source == "local" and model_path:
            extractor = TableExtractor(model_source=model_path, device=device)
        else:
            extractor = TableExtractor(model_source=model_source, device=device)

        # Extract internal components
        self.pipe = extractor.pipe
        self.table_cls = extractor.table_cls
        self.wired_engine = extractor.wired_engine
        self.lineless_engine = getattr(extractor, 'lineless_engine', None)

        print(f"Table Transformer initialized on {device}")

    def parse(self, image: Image.Image, bbox: Optional[List[float]] = None, **kwargs) -> List[TableStructure]:
        """
        Parse table structure

        Args:
            image: PIL Image containing the table
            bbox: Optional bounding box [x1, y1, x2, y2] to crop
            **kwargs: Additional parameters

        Returns:
            List of TableStructure objects representing table cells
        """
        from ...module.table.table_parser import TableExtractor

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