from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import fitz
import numpy as np
import torch
from ultralytics.engine.results import Results

from entity.item import Item
from util.gap_tree import GapTree
from util.text_parser import TextParser


@dataclass
class Page:
    page_num: Union[None, int]
    items: Union[None, list]
    rotated_angle: Union[None, float]
    pdf_page: fitz.Page
    image: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor]
    zoom_factor: float
    is_scanned: bool

    def __init__(self, pdf_page, page_num, image, zoom_factor, items=None, rotated_angle=None):
        self.pdf_page = pdf_page
        self.page_num = page_num
        self.items = items
        self.rotated_angle = rotated_angle
        self.image = image
        self.zoom_factor = zoom_factor
        self.is_scanned = TextParser.is_scanned_pdf_page(pdf_page)

    def build_items(self, results: Results):

        self.items = [Item.from_bbox(i, self.page_num, box) for i, box in
                      enumerate(results.boxes.data)]

    # extract text from the items using pdf api fitz
    def extract_text(self):
        for item in self.items:
            if not self.is_scanned:
                item.content = TextParser.parse_by_fitz(self.pdf_page, item.adjusted_bbox(self.zoom_factor))
            else:
                item.content = TextParser.parse_by_ocr(self.pdf_page, item.adjusted_bbox(self.zoom_factor))
            # recognize the table content based on the table structure
            item.recognize_table_content(self.pdf_page, self.zoom_factor)

    def recognize_table(self):
        for item in self.items:
            item.recognize_table_structure(self.image)

    @property
    def texts(self):
        return [item.markdown_content for item in self.items]

    def draw_bbox(self):
        image = np.array(self.image)

        for item in self.items:
            # Get the bounding box coordinates and convert them to integers
            x1, y1, x2, y2 = map(int, item.bbox)

            # Draw the bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the item id on the image
            cv2.putText(image, str(item.id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            image = item.depict_table(image)

        # Return the image with bounding boxes
        return image

    def sort(self):

        gtree = GapTree(lambda item: item.bbox)

        sorted_text_blocks = gtree.sort(self.items)

        # reassign the id of the items with the sorted order
        for i, item in enumerate(sorted_text_blocks):
            item.id = i

        # assign the sorted items to the page
        self.items = sorted_text_blocks

    def filter_items_by_label(self, filters=None):
        if filters is None:
            return

        items = []
        for item in self.items:
            if item.label not in filters:
                items.append(item)
        self.items = items
