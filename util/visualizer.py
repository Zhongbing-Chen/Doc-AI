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

    @staticmethod
    def visualize_cells(page, item):
        img = page.image
        cells = item.table_structure
        plt.imshow(img, interpolation="lanczos")
        plt.gcf().set_size_inches(20, 20)
        ax = plt.gca()

        for cell in cells:
            bbox = cell.bbox

            if cell['column header']:
                facecolor = (1, 0, 0.45)
                edgecolor = (1, 0, 0.45)
                alpha = 0.3
                linewidth = 2
                hatch = '//////'
            elif cell['projected row header']:
                facecolor = (0.95, 0.6, 0.1)
                edgecolor = (0.95, 0.6, 0.1)
                alpha = 0.3
                linewidth = 2
                hatch = '//////'
            else:
                facecolor = (0.3, 0.74, 0.8)
                edgecolor = (0.3, 0.7, 0.6)
                alpha = 0.3
                linewidth = 2
                hatch = '\\\\\\\\\\\\'

            rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                     edgecolor='none', facecolor=facecolor, alpha=0.1)
            ax.add_patch(rect)
            rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                     edgecolor=edgecolor, facecolor='none', linestyle='-', alpha=alpha)
            ax.add_patch(rect)
            rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=0,
                                     edgecolor=edgecolor, facecolor='none', linestyle='-', hatch=hatch, alpha=0.2)
            ax.add_patch(rect)

        plt.xticks([], [])
        plt.yticks([], [])

        # legend_elements = [Patch(facecolor=(0.3, 0.74, 0.8), edgecolor=(0.3, 0.7, 0.6),
        #                          label='Data cell', hatch='\\\\\\\\\\\\', alpha=0.3),
        #                    Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45),
        #                          label='Column header cell', hatch='//////', alpha=0.3),
        #                    Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
        #                          label='Projected row header cell', hatch='//////', alpha=0.3)]
        # plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
        #            fontsize=10, ncol=3)
        plt.gcf().set_size_inches(10, 10)
        plt.axis('off')
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
    # @staticmethod
    # def visualize_table(page, item):
    #     # visualize the table with cells
    #     pass
