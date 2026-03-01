"""Block classes for document parser"""
from dataclasses import dataclass
from typing import List, Optional
import cv2
import numpy as np
import pandas as pd

from .box import Box


# Layout type mapping (YOLO model labels)
TYPE_DICT = {
    0: 'Text',
    1: 'Title',
    2: 'Header',
    3: 'Footer',
    4: 'Figure',
    5: 'Table',
    6: 'Toc',
    7: 'Figure caption',
    8: 'Table caption'
}


@dataclass
class TableStructure:
    """
    Table structure class representing a cell in a table.

    Attributes:
        bbox: Bounding box [x1, y1, x2, y2]
        cell_text: Text content of the cell
        column_header: Whether it's a column header
        column_nums: Column numbers spanned
        projected_row_header: Whether it's a projected row header
        row_nums: Row numbers spanned
        spans: Cell spans
        subcell: Whether it's a subcell
        table_type: 'wired' or 'lineless'
    """
    bbox: list
    cell_text: str
    column_header: bool
    column_nums: list
    projected_row_header: bool
    row_nums: list
    spans: list
    subcell: Optional[bool] = False
    table_type: str = "wired"

    def __post_init__(self):
        # Set box coordinates from bbox
        self.x_1 = self.bbox[0]
        self.y_1 = self.bbox[1]
        self.x_2 = self.bbox[2]
        self.y_2 = self.bbox[3]
        self.content = self.cell_text


@dataclass
class Block(Box):
    """
    Block class representing a layout block in a document.

    Attributes:
        block_id: Unique identifier
        x_1, y_1, x_2, y_2: Bounding box coordinates
        label: Block label (Text, Title, Table, etc.)
        label_id: Label ID
        page_num: Page number
        layout_score: Detection confidence
        content: Extracted text content
        table_structure: Table structure if this is a table
        related_blocks: Related block IDs
    """
    block_id: str
    x_1: float
    y_1: float
    x_2: float
    y_2: float
    label: str
    label_id: int
    page_num: int
    layout_score: float
    content: str = None
    table_structure: Optional[List[TableStructure]] = None
    related_blocks: list = None

    def __post_init__(self):
        if self.related_blocks is None:
            self.related_blocks = []
        if self.table_structure is None:
            self.table_structure = []

    @classmethod
    def from_bbox(cls, id, page, bbox) -> 'Block':
        """Create a Block from YOLO bbox tensor"""
        import torch
        x_1, y_1, x_2, y_2, layout_score, label_id = bbox
        return Block(
            block_id=id,
            x_1=float(x_1),
            y_1=float(y_1),
            x_2=float(x_2),
            y_2=float(y_2),
            label=TYPE_DICT.get(int(label_id), 'Text'),
            label_id=int(label_id),
            page_num=page,
            layout_score=float(layout_score),
        )

    def recognize_table_structure(self, img, table_parser, dir_path) -> None:
        """Recognize table structure if this block is a table"""
        if self.label == "Table":
            file_path = f"{dir_path}/table_{self.page_num}_{self.block_id}.png"
            img.crop(self.bbox).save(file_path)
            self.table_structure = table_parser.parse(img, self.bbox)

    @property
    def adjusted_bbox(self):
        """Get adjusted bounding box (alias for compatibility)"""
        return self.bbox

    def adjusted_bbox_with_zoom(self, zoom_factor) -> list:
        """Adjust bounding box with zoom factor"""
        return [coord / zoom_factor for coord in self.bbox]

    @property
    def markdown_content(self) -> str:
        """Get markdown content"""
        if self.label == "Table":
            return self.markdown_table_content
        return self.content if self.content else ""

    @property
    def markdown_table_content(self) -> str:
        """Get markdown content for table"""
        if not self.table_structure:
            return ""

        max_row_num = max(max(cell.row_nums) for cell in self.table_structure) + 1
        max_column_num = max(max(cell.column_nums) for cell in self.table_structure) + 1

        df = pd.DataFrame("", index=range(max_row_num), columns=range(max_column_num))

        for cell in self.table_structure:
            for row in cell.row_nums:
                for column in cell.column_nums:
                    df.iloc[row, column] = cell.content.strip().replace("\n", " ") if cell.content else ""

        if not df.empty:
            headers = df.iloc[0].values
            df.columns = headers
            df.drop(index=0, axis=0, inplace=True)

        return df.to_markdown(index=False)

    def depict_table(self, img) -> np.ndarray:
        """Draw table structure on image"""
        if self.label != "Table":
            return img

        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cells = self.table_structure

        for cell in cells:
            bbox = cell.bbox

            if cell.column_header:
                color = (255, 0, 114)
                alpha = 0.3
            elif cell.projected_row_header:
                color = (242, 153, 25)
                alpha = 0.3
            else:
                color = (76, 189, 204)
                alpha = 0.3

            color = color[::-1]  # BGR

            roi = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            overlay = np.full(roi.shape, color, dtype=np.uint8)
            cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

        return img

    def union(self, next_block):
        """Union this block with another block"""
        self.x_1 = min(self.x_1, next_block.x_1)
        self.y_1 = min(self.y_1, next_block.y_1)
        self.x_2 = max(self.x_2, next_block.x_2)
        self.y_2 = max(self.y_2, next_block.y_2)
        if self.layout_score < next_block.layout_score:
            self.layout_score = next_block.layout_score
            self.label = next_block.label

    def to_json(self):
        """Convert to JSON format"""
        return {
            "id": self.block_id,
            "x_1": self.x_1,
            "y_1": self.y_1,
            "x_2": self.x_2,
            "y_2": self.y_2,
            "label": self.label,
            "label_id": self.label_id,
            "page_num": self.page_num,
            "layout_score": self.layout_score,
            "content": self.content,
            "related_blocks": self.related_blocks
        }
