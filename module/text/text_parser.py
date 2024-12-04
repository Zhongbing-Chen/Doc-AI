from typing import List

import fitz
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from entity.block import OcrBlock, Box


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

        # Extract the text from the OCR results, and join them together, item[0] is the bbox, item[1] is the text
        return " ".join(item[1] for item in result)



    @classmethod
    def ocr_all_image_result(cls, image):
        """
        Parse the text using OCR
        :param image: the pdf page
        :return: the text
        """
        print("Extract with OCR")

        # Extract the image from the PDF based on the bounding box

        result, elapse = cls.engine(image, use_det=True, use_cls=True, use_rec=True)
        if result is None:
            return []

        return result

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

    @classmethod
    def match_box_to_ocr(cls, boxes: List[Box], ocr_blocks: List[OcrBlock],
                         overlap_threshold: float = 0.6):
        """Match layout blocks to OCR text blocks based on overlap ratio"""
        for box in boxes:
            matching_texts = []

            for ocr_block in ocr_blocks:
                overlap_area = cls.calculate_overlap_area(box, ocr_block)
                ocr_area = cls.calculate_box_area(ocr_block)

                # Calculate what portion of the OCR box is overlapped
                if ocr_area > 0:
                    overlap_ratio = overlap_area / ocr_area
                    if overlap_ratio >= overlap_threshold:
                        matching_texts.append(ocr_block.content)

            # Sort matching texts by vertical position for proper reading order
            matching_texts = [(text, ocr_block.y_1)
                              for text, ocr_block in zip(matching_texts, ocr_blocks)
                              ]
            matching_texts.sort(key=lambda x: x[1])

            # Join all matching texts with newlines, preserving vertical order
            box.content = '\n'.join(text for text, _ in matching_texts) if matching_texts else None

        return boxes

    @classmethod
    def calculate_overlap_area(cls, box1: Box, box2: Box):
        """Calculate the overlapping area between two bounding boxes"""
        x_left = max(box1.x_1, box2.x_1)
        y_top = max(box1.y_1, box2.y_1)
        x_right = min(box1.x_2, box2.x_2)
        y_bottom = min(box1.y_2, box2.y_2)

        if x_right > x_left and y_bottom > y_top:
            return (x_right - x_left) * (y_bottom - y_top)
        return 0

    @classmethod
    def calculate_box_area(cls, box):
        """Calculate area of a bounding box"""
        return (box.x_2 - box.x_1) * (box.y_2 - box.y_1)
