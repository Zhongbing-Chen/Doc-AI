"""
Tesseract-based orientation detection backend
"""
from typing import Dict, Tuple
from PIL import Image
import fitz

from document_parser.interfaces import OrientationDetectorBackend


class TesseractOrientationBackend(OrientationDetectorBackend):
    """
    Tesseract-based orientation detection backend

    Uses Tesseract's OSD (Orientation and Script Detection) to identify
    document rotation and correct it.
    """

    def __init__(self):
        self.confidence_threshold = 0.5

    def initialize(self, config: Dict) -> None:
        """
        Initialize Tesseract backend

        Args:
            config: Configuration dictionary with keys:
                - confidence_threshold: Minimum confidence for orientation detection
        """
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        print("Tesseract orientation backend initialized")

    def detect(self, image: Image.Image, pdf_page: fitz.Page, **kwargs) -> Tuple[Image.Image, int]:
        """
        Detect and correct orientation

        Args:
            image: PIL Image to process
            pdf_page: fitz.Page object (for rotation metadata)
            **kwargs: Additional parameters

        Returns:
            Tuple of (rotated_image, rotation_angle)
        """
        from module.rotation.orientation_corrector import OrientationCorrector

        # Use existing OrientationCorrector
        img_rotated, rotated_angle = OrientationCorrector.correct_orientation(image, pdf_page)

        return img_rotated, rotated_angle

    def cleanup(self) -> None:
        """No cleanup needed"""
        pass