"""Box classes for document parser"""
from typing import List
import numpy as np


class Box:
    """Base box class representing a bounding box"""
    x_1: float
    y_1: float
    x_2: float
    y_2: float
    content: str

    def __init__(self, x1, y1, x2, y2, content=None):
        self.x_1 = x1
        self.y_1 = y1
        self.x_2 = x2
        self.y_2 = y2
        self.content = content

    @property
    def bbox(self) -> list:
        return [self.x_1, self.y_1, self.x_2, self.y_2]


class OcrBlock(Box):
    """
    The OcrBlock class represents a block of text recognized by OCR.

    Attributes:
        x_1 (float): The x-coordinate of the top-left corner.
        y_1 (float): The y-coordinate of the top-left corner.
        x_2 (float): The x-coordinate of the bottom-right corner.
        y_2 (float): The y-coordinate of the bottom-right corner.
        content (str): The recognized text.
        confidence (float): The confidence score.
    """

    def __init__(self, x1, y1, x2, y2, text, confidence, raw_ocr_result=None):
        super().__init__(x1, y1, x2, y2, text)
        self.confidence = confidence
        self.raw_ocr_result = raw_ocr_result
        self.relationships = {}

    def __repr__(self):
        return f"OcrBlock(x1={self.x_1}, y1={self.y_1}, x2={self.x_2}, y2={self.y_2}, text='{self.content}', confidence={self.confidence})"

    @classmethod
    def from_rapid_ocr(cls, ocr_result):
        """Extract OCR text blocks with their coordinates and content"""
        ocr_blocks = []
        for block in ocr_result:
            if len(block) == 3:
                coords = block[0]
                text = block[1]
                confidence = block[2]
            else:
                coords = block
                text = ''
                confidence = 0.0

            x1 = min(coord[0] for coord in coords)
            y1 = min(coord[1] for coord in coords)
            x2 = max(coord[0] for coord in coords)
            y2 = max(coord[1] for coord in coords)

            ocr_blocks.append(
                OcrBlock(x1=x1, x2=x2, y1=y1, y2=y2, text=text, confidence=confidence, raw_ocr_result=block)
            )

        return ocr_blocks


class BoxUtil:
    """Utility class for box operations"""

    @classmethod
    def match_box_to_ocr(cls, boxes: List[Box], ocr_blocks: List[OcrBlock],
                         overlap_threshold: float = 0.6):
        """Match layout blocks to OCR text blocks based on overlap ratio"""
        for box in boxes:
            matching_texts = []

            for ocr_block in ocr_blocks:
                overlap_area = cls.calculate_overlap_area(box, ocr_block)
                ocr_area = cls.calculate_box_area(ocr_block)

                if ocr_area > 0:
                    overlap_ratio = overlap_area / ocr_area
                    if overlap_ratio >= overlap_threshold:
                        matching_texts.append(ocr_block)

            matching_texts = [(ocr_block.content, ocr_block.y_1)
                              for ocr_block in matching_texts
                              ]
            matching_texts.sort(key=lambda x: x[1])

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

    @classmethod
    def calculate_min_distance(cls, box1: Box, box2: Box) -> float:
        """Calculate the minimum physical distance between two boxes"""
        if box1.x_2 < box2.x_1:
            dx = box2.x_1 - box1.x_2
        elif box2.x_2 < box1.x_1:
            dx = box1.x_1 - box2.x_2
        else:
            dx = 0

        if box1.y_2 < box2.y_1:
            dy = box2.y_1 - box1.y_2
        elif box2.y_2 < box1.y_1:
            dy = box1.y_1 - box2.y_2
        else:
            dy = 0

        return np.sqrt(dx ** 2 + dy ** 2)

    @classmethod
    def extend_layout_box(cls, layout_box: Box, ocr_box: Box) -> Box:
        """Extend layout box to include OCR box"""
        layout_box.x_1 = min(layout_box.x_1, ocr_box.x_1)
        layout_box.y_1 = min(layout_box.y_1, ocr_box.y_1)
        layout_box.x_2 = max(layout_box.x_2, ocr_box.x_2)
        layout_box.y_2 = max(layout_box.y_2, ocr_box.y_2)
        return layout_box
