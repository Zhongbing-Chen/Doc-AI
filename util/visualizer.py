# depict bbox using page's structure and visualize the table with cells
from matplotlib import pyplot as plt, patches

from entity.page import Page
from matplotlib.patches import Patch


class Visualizer:
    @staticmethod
    def depict_bbox(pages: list[Page]):
        # depict bbox using page's structure
        for page in pages:
            img = page.draw_bbox()
            # visualize the image

            # save the image
            plt.imsave(f"{page.page_num}.png", img)
