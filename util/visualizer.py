# depict bbox using page's structure and visualize the table with cells
import os
# depict bbox using page's structure
from datetime import datetime
from matplotlib import pyplot as plt, patches

from entity.page import Page


class Visualizer:
    @staticmethod
    def depict_bbox(pages: list[Page], dir_path: str):
        # Get the current date and time as a string

        for page in pages:
            img = page.draw_bbox()

            file_path = f"{dir_path}/{page.page_num}_original.png"
            # save the cropped image of the table, the file name is the page number + the block id
            # todo remove the saving of the image

            page.image.save(file_path)
            # visualize the image

            # save the image
            plt.imsave(f"{dir_path}/{page.page_num}.png", img)
