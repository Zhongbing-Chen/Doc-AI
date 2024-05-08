from PIL import Image

from model.table_transformer.inference import TableExtractionPipeline, infer_by_image
from entity.table_structure import TableStructure


class TableExtractor:

    def __init__(self, device="cpu"):
        self.pipe = TableExtractionPipeline(
            str_device=device,
            det_config_path=None,
            det_model_path=None,
            str_config_path='/home/zhongbing/Projects/MLE/table-transformer/detr/config/structure_config.json',
            str_model_path='/home/zhongbing/Projects/MLE/table-transformer/detr/models/model_20.pth')

    @classmethod
    def adjust_bbox_positions(cls, sub_bboxes, original_bbox):
        """
        Adjusts the positions of sub-bboxes based on the original bbox's position.

        :param sub_bboxes: List of dictionaries containing sub-bboxes information.
        :param original_bbox: The original bbox coordinates as a list [x_min, y_min, x_max, y_max].
        :return: List of adjusted sub-bboxes.
        """
        adjusted_bboxes = []
        for sub_bbox_dict in sub_bboxes:  # Assuming sub_bboxes is a nested list containing one list of dicts
            sub_bbox = sub_bbox_dict['bbox']
            adjusted_bbox = [
                sub_bbox[0] + original_bbox[0],
                sub_bbox[1] + original_bbox[1],
                sub_bbox[2] + original_bbox[0],
                sub_bbox[3] + original_bbox[1]
            ]
            # Update the dictionary with adjusted bbox
            sub_bbox_dict['bbox'] = adjusted_bbox
            adjusted_bboxes.append(sub_bbox_dict)

        return adjusted_bboxes

    def parse(self, img, bbox=None):

        # crop the table from the image according to the bbox
        if bbox:
            img = img.crop(bbox)

            data = infer_by_image(img, self.pipe)
            data = self.adjust_bbox_positions(data, bbox)
            table_structure = [TableStructure(**table) for table in data]
            return table_structure


if __name__ == '__main__':
    table_parser = TableExtractor()
    img = Image.open('/home/zhongbing/Projects/MLE/table-transformer/detr/img/complex.jpg').convert("RGB")
    result = table_parser.parse(img)
    print(result)
