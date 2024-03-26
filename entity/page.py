from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import fitz
import numpy as np
import torch
from ultralytics.engine.results import Results

from entity.item import Item
from util.text_parser import TextParser


@dataclass
class Page:
    page_num: Union[None, int]
    items: Union[None, list]
    rotated_angle: Union[None, float]
    pdf_page: fitz.Page
    image: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor]
    zoom_factor: float

    def __init__(self, pdf_page, page_num, image, zoom_factor, items=None, rotated_angle=None):
        self.pdf_page = pdf_page
        self.page_num = page_num
        self.items = items
        self.rotated_angle = rotated_angle
        self.image = image
        self.zoom_factor = zoom_factor

    def build_items(self, results: Results):

        self.items = [Item.from_bbox(i, self.page_num, box) for i, box in
                      enumerate(results.boxes.data)]

    # extract text from the items using pdf api fitz
    def extract_text(self):
        for item in self.items:
            item.content = TextParser.parse(self.pdf_page, [coord / self.zoom_factor for coord in item.bbox])

            # recognize the table content based on the table structure
            item.recognize_table_content(self.pdf_page, self.zoom_factor)

    def recognize_table(self):
        for item in self.items:
            item.recognize_table_structure(self.image)

    def texts(self):
        return [item.text for item in self.items]

    def draw_bbox(self):
        image = np.array(self.image)

        for item in self.items:
            # Get the bounding box coordinates and convert them to integers
            x1, y1, x2, y2 = map(int, item.bbox)

            # Draw the bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            image = item.depict_table(image)

        # Return the image with bounding boxes
        return image
