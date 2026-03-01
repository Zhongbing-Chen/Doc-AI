"""
None backend for layout detection (pass-through)
"""
from typing import Any, Dict
from PIL import Image

from ...interfaces import LayoutDetectorBackend


class NoneLayoutBackend(LayoutDetectorBackend):
    """
    Pass-through backend that returns empty results

    Use this when layout detection should be skipped.
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

    def detect(self, image: Image.Image, **kwargs) -> Any:
        """
        Return None (skip layout detection)

        Args:
            image: PIL Image (unused)
            **kwargs: Additional parameters (unused)

        Returns:
            None
        """
        return None

    def cleanup(self) -> None:
        """No cleanup needed"""
        pass