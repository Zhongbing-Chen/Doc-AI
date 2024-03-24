from PIL import Image

from model.table_transformer.inference import TableExtractionPipeline, infer_by_image


class TableParser:
    pipe = TableExtractionPipeline(
        str_device="cpu",
        det_config_path=None,
        det_model_path=None,
        str_config_path='/home/zhongbing/Projects/MLE/table-transformer/detr/config/structure_config.json',
        str_model_path='/home/zhongbing/Projects/MLE/table-transformer/detr/models/model_20.pth')

    @classmethod
    def parse(cls, img, bbox=None):
        return infer_by_image(img, cls.pipe)


if __name__ == '__main__':
    table_parser = TableParser()
    img = Image.open('/home/zhongbing/Projects/MLE/table-transformer/detr/img/complex.jpg').convert("RGB")
    result = TableParser.parse(img)
    print(result)
