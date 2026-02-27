"""
None backend for skew detection (pass-through)
"""
from typing import Dict, Tuple
from PIL import Image

from document_parser.interfaces import SkewDetectorBackend


class NoneSkewBackend(SkewDetectorBackend):
    """
    Pass-through backend that skips skew detection

    Use this when skew detection should be skipped.
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

    def detect_and_correct(self, image: Image.Image, **kwargs) -> Tuple[Image.Image, float]:
        """
        Return original image (skip skew detection)

        Args:
            image: PIL Image
            **kwargs: Additional parameters (unused)

        Returns:
            Tuple of (original_image, 0.0)
        """
        return image, 0.0

    def cleanup(self) -> None:
        """No cleanup needed"""
        pass