from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
from fitz import fitz
from torch import Tensor
from ultralytics.engine.results import Results

from util.table_parser import TableParser

type_dict = {0: 'Header', 1: 'Text', 2: 'Reference', 3: 'Figure caption', 4: 'Figure', 5: 'Table caption', 6: 'Table',
             7: 'Title', 8: 'Footer', 9: 'Equation'}


@dataclass
class Page:
    page_num: Union[None, int]
    items: Union[None, list]
    rotated_angle: Union[None, float]
    pdf_page: Union[None, fitz.Page]
    image: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor]

    def __init__(self, pdf_page, page_num, image, items=None, rotated_angle=None):
        self.pdf_page = pdf_page
        self.page_num = page_num
        self.items = items
        self.rotated_angle = rotated_angle
        self.image = image

    @classmethod
    def from_results(cls, page_num, results: Results, zoom_factor: float,
                     image: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor]):
        return cls(
            page_num=page_num,
            items=[Item.from_bbox(f"{page_num}_{i}", page_num, zoom_factor, box) for i, box in
                   enumerate(results.boxes.data)],
            image=image,
            rotated_angle=0,
            pdf_page=None
        )

    def build_items(self, results: Results, zoom_factor: float, ):

        self.items = [Item.from_bbox(i, self.page_num, zoom_factor, box) for i, box in
                      enumerate(results.boxes.data)]

    # extract text from the items using pdf api fitz
    def extract_text(self, pdf_page):
        for item in self.items:
            rect = fitz.Rect(item.x_1, item.y_1, item.x_2, item.y_2)
            text = pdf_page.get_text("text", clip=rect)
            item.content = text

    def recognize_table(self):
        for item in self.items:
            item.recognize_table_structure(self.image)

    def texts(self):
        return [item.text for item in self.items]


@dataclass
class TableStructure:
    pass


@dataclass
class Item:
    id: str
    x_1: float
    y_1: float
    x_2: float
    y_2: float
    label: str
    label_id: int
    page_num: int
    layout_score: float
    content: str = Union[str, None]
    table_structure = Union[None, dict]

    @classmethod
    def from_bbox(cls, id, page, zoom_factor, bbox: Tensor):
        x_1, y_1, x_2, y_2, layout_score, label_id = bbox
        return cls(
            id=id,
            x_1=x_1.item() / zoom_factor,
            y_1=y_1.item() / zoom_factor,
            x_2=x_2.item() / zoom_factor,
            y_2=y_2.item() / zoom_factor,
            label=type_dict[label_id.item()],
            label_id=label_id.item(),
            page_num=page,
            layout_score=layout_score.item(),
        )

    def recognize_table_structure(self, img):
        if self.label == "Table":
            # recognize table
            print("Recognize Table")
            self.table_structure = TableParser.parse(img, self.bbox)

    @property
    def bbox(self):
        return [self.x_1, self.y_1, self.x_2, self.y_2]
