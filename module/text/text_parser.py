import fitz
from PIL import Image
from rapidocr_onnxruntime import RapidOCR


class TextExtractor:
    engine = RapidOCR()

    @classmethod
    def parse_by_fitz(cls, pdf_page, bbox):
        rect = fitz.Rect(bbox)
        return pdf_page.get_text("text", clip=rect)

    @classmethod
    def parse_by_ocr(cls, pdf_page, bbox):
        print("Extract with OCR")
        rect = fitz.Rect(bbox)

        # Extract the image from the PDF based on the bounding box
        pix = pdf_page.get_pixmap(clip=rect)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        result, elapse = cls.engine(img, use_det=True, use_cls=True, use_rec=True)
        if result is None:
            return ""
        return " ".join(item[1] for item in result)

    @classmethod
    def is_scanned_pdf_page(cls, pdf_page):
        page_text = pdf_page.get_text()

        # Determine if the page is scanned (no text layer)
        return len(page_text.strip()) == 0
