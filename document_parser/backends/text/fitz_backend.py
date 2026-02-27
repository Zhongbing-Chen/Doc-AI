"""
Fitz-based text extraction backend (extracts from PDF text layer)
"""
from typing import Dict, Optional, List
from PIL import Image
import fitz

from document_parser.interfaces import TextExtractorBackend


class FitzTextBackend(TextExtractorBackend):
    """
    Fitz-based text extraction backend

    Extracts text from existing PDF text layer using PyMuPDF (fitz).
    Fast and accurate for digital PDFs with selectable text.
    """

    def __init__(self):
        pass

    def initialize(self, config: Dict) -> None:
        """
        Initialize Fitz backend

        Args:
            config: Configuration dictionary (unused for Fitz)
        """
        pass

    def extract(self, image: Image.Image, pdf_page: fitz.Page, bbox: Optional[List[float]] = None, **kwargs) -> str:
        """
        Extract text from PDF text layer

        Args:
            image: PIL Image (unused)
            pdf_page: fitz.Page object
            bbox: Optional bounding box [x1, y1, x2, y2] in page coordinates
            **kwargs: Additional parameters

        Returns:
            Extracted text string
        """
        if bbox:
            # Extract text from specific region
            # bbox is in image coordinates, need to convert to page coordinates
            rect = fitz.Rect(bbox)
            text = pdf_page.get_text("text", clip=rect)
        else:
            # Extract text from entire page
            text = pdf_page.get_text("text")

        return text.strip()

    def is_scanned(self, pdf_page: fitz.Page) -> bool:
        """
        Check if page is scanned (no text layer)

        Args:
            pdf_page: fitz.Page object

        Returns:
            True if scanned (no text), False otherwise
        """
        page_text = pdf_page.get_text()
        return len(page_text.strip()) == 0

    def cleanup(self) -> None:
        """No cleanup needed"""
        pass