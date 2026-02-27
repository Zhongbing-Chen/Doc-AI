"""
Auto text extraction backend - intelligently selects between Fitz and OCR
"""
from typing import Dict, Optional, List
from PIL import Image
import fitz

from document_parser.interfaces import TextExtractorBackend


class AutoTextBackend(TextExtractorBackend):
    """
    Automatic text extraction backend

    Intelligently selects between:
    - Fitz (for digital PDFs with text layer)
    - OCR (for scanned PDFs or images)

    This is the recommended default for text extraction.
    """

    def __init__(self):
        self.fitz_backend = None
        self.ocr_backend = None
        self.ocr_engine = "rapidocr"  # or "tesseract"

    def initialize(self, config: Dict) -> None:
        """
        Initialize auto backend

        Args:
            config: Configuration dictionary with keys:
                - ocr_engine: 'rapidocr' or 'tesseract' (default: 'rapidocr')
                - languages: List of language codes for OCR
        """
        self.ocr_engine = config.get('ocr_engine', 'rapidocr')

        # Initialize Fitz backend
        from .fitz_backend import FitzTextBackend
        self.fitz_backend = FitzTextBackend()
        self.fitz_backend.initialize(config)

        # Initialize OCR backend
        if self.ocr_engine == 'tesseract':
            from .tesseract_backend import TesseractTextBackend
            self.ocr_backend = TesseractTextBackend()
        else:
            from .rapidocr_backend import RapidOCRBackend
            self.ocr_backend = RapidOCRBackend()

        self.ocr_backend.initialize(config)

        print(f"Auto text backend initialized (Fitz + {self.ocr_engine})")

    def extract(self, image: Image.Image, pdf_page: fitz.Page, bbox: Optional[List[float]] = None, is_scanned: bool = None, **kwargs) -> str:
        """
        Extract text using appropriate method

        Args:
            image: PIL Image
            pdf_page: fitz.Page object
            bbox: Optional bounding box
            is_scanned: Optional flag indicating if page is scanned (auto-detected if None)
            **kwargs: Additional parameters

        Returns:
            Extracted text string
        """
        # Auto-detect if scanned
        if is_scanned is None:
            is_scanned = self.is_scanned(pdf_page)

        if is_scanned:
            # Use OCR for scanned pages
            return self.ocr_backend.extract(image, bbox=bbox, **kwargs)
        else:
            # Use Fitz for digital pages
            return self.fitz_backend.extract(image, pdf_page, bbox=bbox, **kwargs)

    def is_scanned(self, pdf_page: fitz.Page) -> bool:
        """
        Check if page is scanned

        Args:
            pdf_page: fitz.Page object

        Returns:
            True if scanned, False otherwise
        """
        return self.fitz_backend.is_scanned(pdf_page)

    def cleanup(self) -> None:
        """Cleanup both backends"""
        if self.fitz_backend:
            self.fitz_backend.cleanup()
        if self.ocr_backend:
            self.ocr_backend.cleanup()