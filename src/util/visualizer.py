# depict bbox using page's structure and visualize the table with cells
# depict bbox using page's structure
import io

import cv2
import fitz
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from entity.page import Page


class Visualizer:
    @classmethod
    def depict_bbox(cls, pages: list[Page], dir_path: str):
        # Get the current date and time as a string

        for page in pages:
            img, original_image = cls.draw_bbox(page)

            file_path = f"{dir_path}/{page.page_num}_original.png"
            # save the cropped image of the table, the file name is the page number + the block id
            # todo remove the saving of the image

            original_image.save(file_path)
            # visualize the image

            # save the image
            plt.imsave(f"{dir_path}/{page.page_num}.png", img)

    @classmethod
    def draw_bbox(cls, page: Page):
        """
        Draw the bounding box on the image
        :return: the image with bounding boxes
        """

        # Get the image from the pdf page
        pix = page.pdf_page.get_pixmap(matrix=fitz.Matrix(page.zoom_factor, page.zoom_factor))
        # convert the image to numpy array
        # Convert the PyMuPDF pixmap into a bytes object
        img_bytes = pix.tobytes("ppm")
        # Use io.BytesIO to convert bytes into a file-like object for PIL
        img_stream = io.BytesIO(img_bytes)

        # Use PIL to open the image
        image_pil = Image.open(img_stream)
        image = np.array(image_pil)
        for item in page.blocks:
            # Get the bounding box coordinates and convert them to integers
            x1, y1, x2, y2 = map(int, item.bbox)

            # Draw the bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the item id on the image
            cv2.putText(image, str(item.block_id) + " " + item.label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0),
                        2)

            image = item.depict_table(image)

        # Return the image with bounding boxes
        return image, image_pil
