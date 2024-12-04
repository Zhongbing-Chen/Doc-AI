import os.path
from io import BytesIO

import PIL.Image
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download as hf_download
from modelscope import snapshot_download
from table_cls import TableCls
from wired_table_rec import WiredTableRecognition

from entity.block import TableStructure
from model.table_transformer.inference import TableExtractionPipeline, infer_by_image


class TableExtractor:
    """
    The table extractor class

    Attributes:
        pipe: the table extraction pipeline

    Methods:
        adjust_bbox_positions: adjust the bbox positions
        parse: parse the table from the image
    """

    def __init__(self, model_source="huggingface", device="cpu"):

        if model_source == "huggingface":
            model_dir = hf_download("zhongbing/table-transformer-finetuned")
        elif model_source == "modelscope":
            model_dir = snapshot_download("zhongbing/table-transformer-finetuned")
        else:
            model_dir = "/Users/zhongbing/Projects/MLE/Doc-AI/model/table_transformer"

        self.table_cls = TableCls(model_type="x")

        self.wired_engine = WiredTableRecognition()

        self.pipe = TableExtractionPipeline(
            str_device=device,
            det_config_path=None,
            det_model_path=None,
            str_config_path=os.path.join(model_dir, "structure_config.json"),
            str_model_path=os.path.join(model_dir, 'model_20.pth'))

    @classmethod
    def adjust_bbox_positions(cls, sub_bboxes, original_bbox):
        """
        Adjusts the positions of sub-bboxes based on the original bbox's position.

        :param sub_bboxes: List of dictionaries containing sub-bboxes information.
        :param original_bbox: The original bbox coordinates as a list [x_min, y_min, x_max, y_max].
        :return: List of adjusted sub-bboxes.
        """
        adjusted_bboxes = []
        for sub_bbox_dict in sub_bboxes:  # Iterate over each sub-bbox
            sub_bbox = sub_bbox_dict['bbox'] if 'bbox' in sub_bbox_dict else sub_bbox_dict
            # Adjust the sub-bbox positions based on the original bbox
            adjusted_bbox = [
                sub_bbox[0] + original_bbox[0],
                sub_bbox[1] + original_bbox[1],
                sub_bbox[2] + original_bbox[0],
                sub_bbox[3] + original_bbox[1]
            ]
            # Update the dictionary with adjusted bbox
            # temp solution for the two different formats of sub-bboxes
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
        :return: the table structure
        """
        # crop the table from the image according to the bbox
        if bbox:
            img = img.crop(bbox)
            img_array = np.array(img)

            cls, elasp = self.table_cls(img_array)
            if cls == 'wired':
                print("Recognize Wired Table")
                html, elasp, polygons, logic_points, ocr_res = self.wired_engine(img_array, need_ocr=False)
                table_structures = []
                if polygons is None:
                    return table_structures
                polygons = self.adjust_bbox_positions(polygons, bbox)
                for bbox, logic_point in zip(polygons, logic_points):
                    row_nums = sorted(list(set([int(x) for x in range(logic_point[0], logic_point[1] + 1)])))
                    column_nums = sorted(list(set([int(x) for x in range(logic_point[2], logic_point[3] + 1)])))
                    if not row_nums or not column_nums:
                        continue
                    column_header = 0 in row_nums
                    projected_row_header = 0 in column_nums
                    spans = [(min(row_nums), max(row_nums) + 1, min(column_nums), max(column_nums) + 1)]
                    subcell = len(row_nums) > 1 or len(column_nums) > 1
                    table_structure = TableStructure(bbox, "", column_header, column_nums, projected_row_header,
                                                     row_nums, spans, subcell, table_type=cls)
                    table_structures.append(table_structure)
                return table_structures
            else:
                print("Recognize Lineless Table")
                data = infer_by_image(img, self.pipe)
                data = self.adjust_bbox_positions(data, bbox)
                table_structure = [TableStructure(**table, table_type=cls) for table in data]
                return table_structure


if __name__ == '__main__':
    table_parser = TableExtractor()
    img = Image.open('/home/zhongbing/Projects/MLE/table-transformer/detr/img/complex.jpg').convert("RGB")
    result = table_parser.parse(img)
    print(result)
