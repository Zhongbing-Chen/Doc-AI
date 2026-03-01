"""
Tesseract-based text extraction backend
"""
from typing import Dict, Optional, List
from PIL import Image
import fitz
import pytesseract

from ...interfaces import TextExtractorBackend


class TesseractTextBackend(TextExtractorBackend):
    """
    Tesseract-based text extraction backend

    Uses Tesseract OCR engine for text recognition.
    Classic OCR engine with good language support.
    """

    def __init__(self):
        self.languages = ['chi_sim', 'eng']

    def initialize(self, config: Dict) -> None:
        """
        Initialize Tesseract backend

        Args:
            config: Configuration dictionary with keys:
                - languages: List of language codes (default: ['chi_sim', 'eng'])
        """
        self.languages = config.get('languages', ['chi_sim', 'eng'])
        print(f"Tesseract OCR initialized with languages: {self.languages}")

    def extract(self, image: Image.Image, bbox: Optional[List[float]] = None, **kwargs) -> str:
        """
        Extract text using Tesseract

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

        # Run Tesseract OCR
        lang = '+'.join(self.languages)
        text = pytesseract.image_to_string(image, lang=lang)

        return text.strip()

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
        """No cleanup needed"""
        pass