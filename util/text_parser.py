import fitz


class TextParser:

    @classmethod
    def parse(cls, pdf_page, bbox):
        rect = fitz.Rect(bbox)
        return pdf_page.get_text("text", clip=rect)
