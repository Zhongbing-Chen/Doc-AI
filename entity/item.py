from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np
from matplotlib import pyplot as plt, patches
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

    def depict_table(self, img):
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
