"""Visualizer for document parser - depicts bounding boxes"""
import io

import cv2
import fitz
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from ..entity.page import Page


class Visualizer:
    """Visualizer class for drawing bounding boxes on pages"""

    @classmethod
    def depict_bbox(cls, pages: list, dir_path: str):
        """Draw bounding boxes on all pages and save images"""
        for page in pages:
            img, original_image = cls.draw_bbox(page)
            file_path = f"{dir_path}/{page.page_num}_original.png"
            original_image.save(file_path)
            plt.imsave(f"{dir_path}/{page.page_num}.png", img)

    @classmethod
    def draw_bbox(cls, page: Page):
        """Draw bounding boxes on page image"""
        pix = page.pdf_page.get_pixmap(matrix=fitz.Matrix(page.zoom_factor, page.zoom_factor))
        img_bytes = pix.tobytes("ppm")
        img_stream = io.BytesIO(img_bytes)
        image_pil = Image.open(img_stream)
        image = np.array(image_pil)

        for item in page.blocks:
            x1, y1, x2, y2 = map(int, item.bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, str(item.block_id) + " " + item.label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            image = item.depict_table(image)

        return image, image_pil
