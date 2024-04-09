from dataclasses import dataclass
from typing import Optional

from util.text_parser import TextParser


@dataclass
class TableStructure:
    bbox: list
    cell_text: str
    column_header: bool
    column_nums: list
    projected_row_header: bool
    row_nums: list
    spans: list
    subcell: Optional[bool] = False

    def recognize_content(self, pdf, zoom_factor):
        # recognize the content of the table using fitz api
        # print(TextParser.parse(pdf, [coord / zoom_factor for coord in self.bbox]))
        self.cell_text = TextParser.parse_by_fitz(pdf, [coord / zoom_factor for coord in self.bbox])
