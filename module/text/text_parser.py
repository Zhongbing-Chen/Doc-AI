import fitz
from PIL import Image
from rapidocr_onnxruntime import RapidOCR


class TextExtractor:
    """
    The text extractor class

    Attributes:
        engine: the OCR engine

    Methods:
        parse_by_fitz: parse the text using fitz
        parse_by_ocr: parse the text using OCR
        is_scanned_pdf_page: determine if the PDF page is scanned
    """
    engine = RapidOCR()

    @classmethod
    def parse_by_fitz(cls, pdf_page, bbox) -> str:
        """
        Parse the text using fitz
        :param pdf_page: the pdf page
        :param bbox: the bounding box
        :return: the text
        """
        rect = fitz.Rect(bbox)
        return pdf_page.get_text("text", clip=rect)

    @classmethod
    def parse_by_ocr(cls, pdf_page, bbox) -> str:
        """
        Parse the text using OCR
        :param pdf_page: the pdf page
        :param bbox: the bounding box
        :return: the text
        """
        print("Extract with OCR")
        rect = fitz.Rect(bbox)

        # Extract the image from the PDF based on the bounding box
        pix = pdf_page.get_pixmap(clip=rect)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        result, elapse = cls.engine(img, use_det=True, use_cls=True, use_rec=True)
        if result is None:
            return ""

        # Extract the text from the OCR result, and join them together, item[0] is the bbox, item[1] is the text
        return " ".join(item[1] for item in result)

    @classmethod
    def is_scanned_pdf_page(cls, pdf_page) -> bool:
        """
        Determine if the PDF page is scanned
        :param pdf_page: the PDF page
        :return: True if the page is scanned, False otherwise
        """
        page_text = pdf_page.get_text()

        # Determine if the page is scanned (no text layer)
        return len(page_text.strip()) == 0
