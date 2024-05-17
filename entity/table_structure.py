from dataclasses import dataclass
from typing import Optional

from module.text.text_parser import TextExtractor


@dataclass
class TableStructure:
    """
    The table structure class

    Attributes:
        bbox: the bounding box of the table
        cell_text: the text of the cell
        column_header: the flag to indicate whether the cell is a column header
        column_nums: the column numbers of the cell
        projected_row_header: the flag to indicate whether the cell is a projected row header
        row_nums: the row numbers of the cell
        spans: the spans of the cell
        subcell: the flag to indicate whether the cell is a subcell

    Methods:
        recognize_content: recognize the content of the table
    """
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
        self.cell_text = TextExtractor.parse_by_fitz(pdf, [coord / zoom_factor for coord in self.bbox])
