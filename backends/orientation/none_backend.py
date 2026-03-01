"""
None backend for orientation detection (pass-through)
"""
from typing import Dict, Tuple
from PIL import Image
import fitz

from ...interfaces import OrientationDetectorBackend


class NoneOrientationBackend(OrientationDetectorBackend):
    """
    Pass-through backend that skips orientation detection

    Use this when orientation detection should be skipped.
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

    def detect(self, image: Image.Image, pdf_page: fitz.Page, **kwargs) -> Tuple[Image.Image, int]:
        """
        Return original image (skip orientation detection)

        Args:
            image: PIL Image
            pdf_page: fitz.Page object (unused)
            **kwargs: Additional parameters (unused)

        Returns:
            Tuple of (original_image, 0)
        """
        return image, 0

    def cleanup(self) -> None:
        """No cleanup needed"""
        pass