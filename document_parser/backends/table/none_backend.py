"""
None backend for table parsing (pass-through)
"""
from typing import Dict, Optional, List
from PIL import Image

from document_parser.interfaces import TableParserBackend
from entity.block import TableStructure


class NoneTableBackend(TableParserBackend):
    """
    Pass-through backend that returns empty results

    Use this when table parsing should be skipped.
    """

    def __init__(self):
        pass

    def initialize(self, config: Dict) -> None:
        """
        Initialize none backend

        Args:
            config: Configuration dictionary (unused)
        """
        pass

    def parse(self, image: Image.Image, bbox: Optional[List[float]] = None, **kwargs) -> List[TableStructure]:
        """
        Return empty list (skip table parsing)

        Args:
            image: PIL Image (unused)
            bbox: Optional bounding box (unused)
            **kwargs: Additional parameters (unused)

        Returns:
            Empty list
        """
        return []

    def cleanup(self) -> None:
        """No cleanup needed"""
        pass