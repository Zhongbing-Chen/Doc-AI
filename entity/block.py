from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch import Tensor

from module.table.table_parser import TableExtractor

type_dict = {0: 'Header', 1: 'Text', 2: 'Reference', 3: 'Figure caption', 4: 'Figure', 5: 'Table caption', 6: 'Table',
             7: 'Title', 8: 'Footer', 9: 'Equation'}


@dataclass
class Block:
    """
    The block class

    Attributes:
        x_1: the x coordinate of the top left corner
        y_1: the y coordinate of the top left corner
        x_2: the x coordinate of the bottom right corner
        y_2: the y coordinate of the bottom right corner
        label: the label of the block
        label_id: the label id of the block
        page_num: the page number
        layout_score: the layout score of the block
        content: the content of the block
        table_structure: the table structure of the block
        related_blocks: the related blocks of the block

    Methods:
        from_bbox: create a block from the bounding box
        recognize_table_structure: recognize the table structure if this block is a table
        recognize_table_content: recognize the table content
        adjusted_bbox: adjust the bounding box with the zoom factor
        markdown_content: get the markdown content
        markdown_table_content: get the markdown content for the table
        depict_table: depict the table structure on the image
        union: union the block with the next block
        to_json: convert the block to json format
    """
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
    related_blocks = []

    @classmethod
    def from_bbox(cls, id, page, bbox: Tensor) -> 'Block':
        """
        Create a block from the bounding box
        :param id: the id of the block
        :param page: the page number
        :param bbox: the bounding box
        :return: the block object
        """
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

    def recognize_table_structure(self, img, table_parser) -> None:
        """
        Recognize the table structure if this block is a table
        :param img: the image
        :param table_parser: the table parser
        :return: None
        """
        if self.label == "Table":
            # recognize table
            print("Recognize Table")
            self.table_structure = table_parser.parse(img, self.bbox)

    def recognize_table_content(self, pdf_page, zoom_factor) -> None:
        """
        Recognize the table content
        :param pdf_page: the pdf page
        :param zoom_factor: the zoom factor
        :return: None
        """
        if self.label == "Table":
            # recognize table content
            print("Recognize Table Content")
            for i in self.table_structure:
                i.recognize_content(pdf_page, zoom_factor)

    @property
    def bbox(self) -> list:
        return [self.x_1, self.y_1, self.x_2, self.y_2]

    def adjusted_bbox(self, zoom_factor) -> list:
        """
        Adjust the bounding box with the zoom factor
        :param zoom_factor: the zoom factor
        :return: the adjusted bounding box
        """
        return [coord / zoom_factor for coord in self.bbox]

    @property
    def markdown_content(self) -> str:
        """
        Get the markdown content
        :return: the markdown content for this block, if it is a table, return the markdowned table content
        """
        if self.label == "Table":
            return self.markdown_table_content
        return self.content

    @property
    def markdown_table_content(self) -> str:
        """
        Get the markdown content for the table
        :return: the markdown content for the table
        """

        # if the table structure is empty, return an empty string
        if not self.table_structure:
            return ""

        # get the maximum row number and column number
        max_row_num = max(max(cell.row_nums) for cell in self.table_structure) + 1
        max_column_num = max(max(cell.column_nums) for cell in self.table_structure) + 1

        # create a dataframe to store the table content
        df = pd.DataFrame("", index=range(max_row_num), columns=range(max_column_num))

        # fill the dataframe with the table content
        for cell in self.table_structure:
            for row in cell.row_nums:
                for column in cell.column_nums:
                    df.iloc[row, column] = cell.cell_text.strip().replace("\n", " ")

        # convert the dataframe to markdown format
        if not df.empty:
            headers = df.iloc[0].values
            df.columns = headers
            df.drop(index=0, axis=0, inplace=True)
        return df.to_markdown(index=False)

    def depict_table(self, img) -> np.ndarray:
        """
        Depict the table structure on the image
        :param img: the image
        :return: the image with the table structure depicted
        """
        if self.label != "Table":
            return img

        # Convert the image to RGB if it is in BGR format (as OpenCV uses BGR by default)
        if img.shape[2] == 3:  # If the image has 3 channels
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create a figure and axis for drawing
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        cells = self.table_structure

        # Assume img is already in the correct color format (RGB or BGR as per your working environment)

        for cell in cells:
            bbox = cell.bbox

            if cell.column_header:
                color = (255, 0, 114)  # Example color
                alpha = 0.3  # Transparency factor
            elif cell.projected_row_header:
                color = (242, 153, 25)  # Example color
                alpha = 0.3  # Transparency factor
            else:
                color = (76, 189, 204)  # Example color
                alpha = 0.3  # Transparency factor

            # Convert color to BGR if using OpenCV for image processing
            color = color[::-1]  # Flip to BGR if needed

            # Extract the region of interest (ROI) where the rectangle will be drawn
            roi = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

            # Create a rectangle to overlay
            overlay = np.full(roi.shape, color, dtype=np.uint8)

            # Blend the overlay with the ROI
            cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

            # Draw the edge of the rectangle
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

        return img

    def union(self, next_block):
        """
        Union the block with the next block
        :param next_block: the next block
        :return: None
        """
        self.x_1 = min(self.x_1, next_block.x_1)
        self.y_1 = min(self.y_1, next_block.y_1)
        self.x_2 = max(self.x_2, next_block.x_2)
        self.y_2 = max(self.y_2, next_block.y_2)
        if self.layout_score < next_block.layout_score:
            self.layout_score = next_block.layout_score
            self.label = next_block.label

    def to_json(self):
        """
        Convert the block to json format
        :return: the json format of the block
        """
        return {
            "id": self.id,
            "x_1": self.x_1,
            "y_1": self.y_1,
            "x_2": self.x_2,
            "y_2": self.y_2,
            "label": self.label,
            "label_id": self.label_id,
            "page_num": self.page_num,
            "layout_score": self.layout_score,
            "content": self.content,
            "table_structure": self.table_structure,
            "related_blocks": self.related_blocks
        }
