"""
Text parser for standalone document parser
"""
import fitz
import numpy as np


class TextExtractor:
    """
    Text extractor for standalone document parser.
    Provides methods for extracting text from PDF pages and images.
    """

    @staticmethod
    def parse_by_fitz(pdf_page, bbox=None):
        """
        Extract text from PDF page using fitz (PyMuPDF)

        Args:
            pdf_page: fitz.Page object
            bbox: Optional bounding box [x1, y1, x2, y2] in page coordinates

        Returns:
            Extracted text string
        """
        if bbox:
            rect = fitz.Rect(bbox)
            text = pdf_page.get_text("text", clip=rect)
        else:
            text = pdf_page.get_text("text")
        return text.strip()

    @staticmethod
    def ocr_all_image_result(image):
        """
        Perform OCR on entire image using RapidOCR

        Args:
            image: PIL Image or numpy array

        Returns:
            OCR result list with [(bbox, text, confidence), ...]
        """
        try:
            from rapidocr_onnxruntime import RapidOCR
            ocr_engine = RapidOCR()
            img_array = np.array(image)
            result, _ = ocr_engine(img_array)
            return result
        except ImportError:
            # RapidOCR not available
            return []

    @staticmethod
    def detection_all_image_result(image):
        """
        Detect text regions in image (for digital PDFs with text layer)

        Args:
            image: PIL Image or numpy array

        Returns:
            Text detection result list
        """
        try:
            from rapidocr_onnxruntime import RapidOCR
            ocr_engine = RapidOCR()
            img_array = np.array(image)
            result, _ = ocr_engine(img_array)
            return result
        except ImportError:
            return []
