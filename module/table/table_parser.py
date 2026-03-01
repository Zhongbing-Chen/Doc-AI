"""
Table extractor for standalone document parser

This is a simplified version that works without external module dependencies.
"""
import os.path
from io import BytesIO

import PIL.Image
import numpy as np
from PIL import Image

from ...entity.block import TableStructure


class TableExtractor:
    """
    The table extractor class for standalone document parser.

    This class is designed to be initialized with model components
    passed directly rather than loading them internally.
    """

    def __init__(self, pipe=None, table_cls=None, wired_engine=None, lineless_engine=None):
        """
        Initialize table extractor with provided components.

        Args:
            pipe: Table extraction pipeline (can be None if not using table transformer)
            table_cls: Table classifier (can be None if not using wired table detection)
            wired_engine: Wired table recognition engine (can be None)
            lineless_engine: Lineless table recognition engine (can be None)
        """
        self.pipe = pipe
        self.table_cls = table_cls
        self.wired_engine = wired_engine
        self.lineless_engine = lineless_engine

    @classmethod
    def adjust_bbox_positions(cls, sub_bboxes, original_bbox):
        """
        Adjusts the positions of sub-bboxes based on the original bbox's position.

        :param sub_bboxes: List of dictionaries containing sub-bboxes information.
        :param original_bbox: The original bbox coordinates as a list [x_min, y_min, x_max, y_max].
        :return: List of adjusted sub-bboxes.
        """
        adjusted_bboxes = []
        for sub_bbox_dict in sub_bboxes:
            sub_bbox = sub_bbox_dict['bbox'] if 'bbox' in sub_bbox_dict else sub_bbox_dict
            adjusted_bbox = [
                sub_bbox[0] + original_bbox[0],
                sub_bbox[1] + original_bbox[1],
                sub_bbox[2] + original_bbox[0],
                sub_bbox[3] + original_bbox[1]
            ]
            if 'bbox' in sub_bbox_dict:
                sub_bbox_dict['bbox'] = adjusted_bbox
            else:
                sub_bbox_dict = adjusted_bbox
            adjusted_bboxes.append(sub_bbox_dict)

        return adjusted_bboxes

    def parse(self, img: PIL.Image.Image, bbox=None):
        """
        Parse the table from the image

        :param img: the image
        :param bbox: the bounding box
        :return: the table structure list
        """
        if bbox:
            img = img.crop(bbox)
            img_array = np.array(img)

            # If table classifier is available, use it
            if self.table_cls is not None:
                cls, elasp = self.table_cls(img_array)
                if cls == 'wired':
                    print("Recognize Wired Table")
                    if self.wired_engine is not None:
                        html, elasp, polygons, logic_points, ocr_res = self.wired_engine(img_array, need_ocr=False)
                        table_structures = []
                        if polygons is None:
                            return table_structures
                        polygons = self.adjust_bbox_positions(polygons, bbox)
                        for bbox_item, logic_point in zip(polygons, logic_points):
                            row_nums = sorted(list(set([int(x) for x in range(logic_point[0], logic_point[1] + 1)])))
                            column_nums = sorted(list(set([int(x) for x in range(logic_point[2], logic_point[3] + 1)])))
                            if not row_nums or not column_nums:
                                continue
                            column_header = 0 in row_nums
                            projected_row_header = 0 in column_nums
                            spans = [(min(row_nums), max(row_nums) + 1, min(column_nums), max(column_nums) + 1)]
                            subcell = len(row_nums) > 1 or len(column_nums) > 1
                            table_structure = TableStructure(bbox_item, "", column_header, column_nums, projected_row_header,
                                                             row_nums, spans, subcell, table_type=cls)
                            table_structures.append(table_structure)
                        return table_structures
                else:
                    print("Recognize Lineless Table")
                    if self.pipe is not None:
                        from model.table_transformer.inference import infer_by_image
                        data = infer_by_image(img, self.pipe)
                        data = self.adjust_bbox_positions(data, bbox)
                        table_structure = [TableStructure(**table, table_type=cls) for table in data]
                        return table_structure

        # Return empty list if no components available or parsing failed
        return []
