# depict bbox using page's structure and visualize the table with cells
from matplotlib import pyplot as plt, patches

from entity.page import Page


class Visualizer:
    @staticmethod
    def depict_bbox(pages: list[Page]):
        # depict bbox using page's structure
        for page in pages:
            img = page.draw_bbox()
            page.image.save(f"./results/detail/{page.page_num}_original.png")
            # visualize the image

            # save the image
            plt.imsave(f"./results/detail/{page.page_num}.png", img)
