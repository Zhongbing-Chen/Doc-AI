"""
RapidOCR-based text extraction backend
"""
from typing import Dict, Optional, List
from PIL import Image
import fitz
import numpy as np

from document_parser.interfaces import TextExtractorBackend


class RapidOCRBackend(TextExtractorBackend):
    """
    RapidOCR-based text extraction backend

    Uses RapidOCR for optical character recognition.
    Good for scanned documents or images.
    """

    def __init__(self):
        self.ocr_engine = None

    def initialize(self, config: Dict) -> None:
        """
        Initialize RapidOCR backend

        Args:
            config: Configuration dictionary with keys:
                - languages: List of language codes (unused for RapidOCR, auto-detects)
        """
        from rapidocr_onnxruntime import RapidOCR

        self.ocr_engine = RapidOCR()
        print("RapidOCR initialized")

    def extract(self, image: Image.Image, bbox: Optional[List[float]] = None, **kwargs) -> str:
        """
        Extract text using OCR

        Args:
            image: PIL Image to process
            bbox: Optional bounding box [x1, y1, x2, y2] to crop
            **kwargs: Additional parameters

        Returns:
            Extracted text string
        """
        # Crop image if bbox provided
        if bbox:
            image = image.crop(bbox)

        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Run OCR
        result, elapse = self.ocr_engine(img_array)

        if result is None or len(result) == 0:
            return ""

        # result is list of [box, text, confidence]
        # Extract just the text and join
        texts = [item[1] for item in result]
        return ' '.join(texts)

    def is_scanned(self, pdf_page: fitz.Page) -> bool:
        """
        Check if page is scanned

        Args:
            pdf_page: fitz.Page object

        Returns:
            True if scanned, False otherwise
        """
        page_text = pdf_page.get_text()
        return len(page_text.strip()) == 0

    def cleanup(self) -> None:
        """Cleanup OCR engine"""
        if self.ocr_engine is not None:
            self.ocr_engine = None