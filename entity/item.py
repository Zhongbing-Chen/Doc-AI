from dataclasses import dataclass
from typing import Union

from torch import Tensor

from util.table_parser import TableParser

type_dict = {0: 'Header', 1: 'Text', 2: 'Reference', 3: 'Figure caption', 4: 'Figure', 5: 'Table caption', 6: 'Table',
             7: 'Title', 8: 'Footer', 9: 'Equation'}


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
    table_structure = Union[None, list]

    @classmethod
    def from_bbox(cls, id, page, bbox: Tensor):
        x_1, y_1, x_2, y_2, layout_score, label_id = bbox
        return cls(
            id=id,
            x_1=x_1.item(),
            y_1=y_1.item(),
            x_2=x_2.item(),
            y_2=y_2.item(),
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

    def recognize_table_content(self, pdf_page, zoom_factor):
        if self.label == "Table":
            # recognize table content
            print("Recognize Table Content")
            for i in self.table_structure:
                i.recognize_content(pdf_page, zoom_factor)

    @property
    def bbox(self):
        return [self.x_1, self.y_1, self.x_2, self.y_2]

    def adjusted_bbox(self, zoom_factor):
        return [coord / zoom_factor for coord in self.bbox]

    def depict_table(self):
        pass
