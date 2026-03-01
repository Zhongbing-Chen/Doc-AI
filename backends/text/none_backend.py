"""
None backend for text extraction (pass-through)
"""
from typing import Dict, Optional, List
from PIL import Image
import fitz

from ...interfaces import TextExtractorBackend


class NoneTextBackend(TextExtractorBackend):
    """
    Pass-through backend that returns empty text

    Use this when text extraction should be skipped.
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

    def extract(self, image: Image.Image, bbox: Optional[List[float]] = None, **kwargs) -> str:
        """
        Return empty string (skip text extraction)

        Args:
            image: PIL Image (unused)
            bbox: Optional bounding box (unused)
            **kwargs: Additional parameters (unused)

        Returns:
            Empty string
        """
        return ""

    def is_scanned(self, pdf_page: fitz.Page) -> bool:
        """
        Always return False

        Args:
            pdf_page: fitz.Page object (unused)

        Returns:
            False
        """
        return False

    def cleanup(self) -> None:
        """No cleanup needed"""
        pass