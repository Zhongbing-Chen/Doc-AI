import gc
import io
import os
import sys
import time
from datetime import datetime

import fitz
import ocrmypdf
import psutil
from PIL import Image

from entity.page import Page
from module.layout.layout_detector import LayoutDetector
from module.rotation.orientation_corrector import OrientationCorrector
from module.table.table_parser import TableExtractor
from module.text.text_parser import TextExtractor
from util.visualizer import Visualizer


def get_variable_memory(variable):
    """
    获取Python变量的内存使用量(单位为bytes)。

    参数:
    variable -- 需要检查内存使用量的变量

    返回:
    变量的内存使用量(单位为bytes)
    """
    # 获取变量的引用计数
    ref_count = sys.getrefcount(variable)

    # 遍历所有Python对象,找到与变量匹配的对象
    total_size = 0
    for obj in gc.get_objects():
        if id(obj) == id(variable):
            total_size = sys.getsizeof(obj)
            break

    # 根据引用计数调整内存使用量
    return total_size * (ref_count - 1)


class PDFProcessor:
    """
    The pdf processor class
    zoom_factor: the zoom factor of the pdf
    device: the device to run the model
    layout_detector: the layout detector
    table_parser: the table parser
    """

    def __init__(self, zoom_factor=3, device="cpu", model_source="huggingface"):
        #
        self.zoom_factor = zoom_factor
        self.device = device

        self.layout_detector = LayoutDetector(model_source, device=device)

        self.table_parser = TableExtractor(model_source=model_source, device=device)

    def set_layout_detector(self, model):
        self.layout_detector = model

    @staticmethod
    def pre_ocr(file_path, output_file=None, language="eng", rotate_pages_threshold=3.0):
        # recognize the text in the pdf
        """
        using ocrmypdf to recognize the text in the pdf, and output the pdf file with the text layer
        :param file_path: the path of the pdf file
        :param output_file: the path of the output pdf file
        :param language: the language of the text in the pdf, default is "eng", could be "chi_sim+eng"
        :param rotate_pages_threshold: the threshold of the rotation, higher threshold means more strict standard to rotate the pages
        :return: the path of the output pdf file
        """
        if output_file is None:
            output_file = file_path.replace(".pdf", "_ocr.pdf")

        ocrmypdf.ocr(file_path, output_file, rotate_pages=True,
                     rotate_pages_threshold=rotate_pages_threshold, language=language,
                     deskew=True,
                     skip_text=True,
                     clean=True,
                     invalidate_digital_signatures=True,
                     oversample=200
                     # force_ocr=True
                     )
        return output_file

    def convert_page_to_image(self, page):
        """
        convert the page to image
        :param page: the page object
        :return: the image object and the bytes of the image
        """
        # convert the page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(self.zoom_factor, self.zoom_factor))
        # convert the image to numpy array
        # Convert the PyMuPDF pixmap into a bytes object
        img_bytes = pix.tobytes("ppm")
        # Use io.BytesIO to convert bytes into a file-like object for PIL
        img_stream = io.BytesIO(img_bytes)

        # Use PIL to open the image
        img = Image.open(img_stream)
        return img, img_bytes

    def process(self, file_path, language="chi_sim+eng", use_ocr=False):
        """
        process the pdf file
        :param file_path: the path of the original pdf file
        :param language: the language of the text in the pdf, default is "chi_sim+eng"
        """
        # recognize the text in the pdf, and output the pdf file with the text layer
        if use_ocr:
            file_path = self.pre_ocr(file_path, language=language)
        pages = []

        current_time_string = datetime.now().strftime("%Y%m%d%H%M%S")
        dir_path_detail = f"./results/{current_time_string}/detail"
        dir_path_table = f"./results/{current_time_string}/table"
        if not os.path.exists(dir_path_detail) or not os.path.exists(dir_path_table):
            # 如果文件夹不存在，则创建
            os.makedirs(dir_path_detail, exist_ok=True)
            os.makedirs(dir_path_table, exist_ok=True)

        # open the updated pdf file
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            # convert the page to image

            print(f"The memory taked by ocr engine:{get_variable_memory(TextExtractor.engine)}")
            page = doc.load_page(page_num)

            img, img_bytes = self.convert_page_to_image(page)
            img_rotated = img
            deskew_angle = 0
            rotated_angle = 0

            # deprecated the pre_process function
            # deskew_angle, img_rotated, rotated_angle = self.pre_process(img_rotated, page)
            # print(angle)
            # print(time.time() - start)
            page = Page(page_num=page_num, image=img_rotated, pdf_page=page, zoom_factor=self.zoom_factor,
                        rotated_angle=rotated_angle, skewed_angle=deskew_angle)
            layout = self.layout_detector.detect(page.image, conf=0.5, iou=0.45)
            # layouts.append(layout)
            page.build_blocks(layout)

            # filter item by label
            page.filter_items_by_label(filters=["Header", "Footer", "Figure"])

            # merge the overlap block
            page.merge_overlap_block(threshold=0.7)

            # sort the items
            page.sort()

            # table part
            page.recognize_table(self.table_parser, dir_path_table)

            # text part
            page.extract_text()

            # append the image to the list
            pages.append(page)

        Visualizer.depict_bbox(pages, dir_path_detail)
        return pages

    @staticmethod
    def pre_process(img, page):
        """
        pre-process the image
        :param img: the image object
        :param page: the page object
        :return: the deskew angle, the rotated image, the rotated angle
        """
        img_rotated, rotated_angle = OrientationCorrector.rotate_through_tesseract(img, page)
        print(rotated_angle)
        img_rotated, deskew_angle = OrientationCorrector.deskew_image(img_rotated)
        return deskew_angle, img_rotated, rotated_angle

    @staticmethod
    def convert_to_markdown(pages: list[Page]):
        """
        convert the pages to markdown format
        :param pages: the list of pages
        :return: the markdown content
        """
        markdown_content = []
        for page in pages:
            # convert the pages to markdown, extract the content from items in page

            markdown_content.extend(page.texts)
        # convert it to markdown format

        # filter the empty content in the list
        markdown_content = list(filter(None, markdown_content))
        # 将列表转换为Markdown格式的无序列表
        markdown_list = "\n".join(markdown_content)

        # 指定Markdown文件路径
        markdown_file_path = "list_output.md"

        # 将Markdown列表写入文件
        with open(markdown_file_path, "w", encoding="utf-8") as file:
            file.write(markdown_list)

        print(f"Markdown列表已保存到文件：{markdown_file_path}")

        return markdown_content

    @staticmethod
    def merge(pages: list[Page]):
        """
        merge the pages into one json file with the attributes of the page and blocks
        :param pages: the list of pages
        :return: the dict of the pages
        """
        content = []
        for i in pages:
            content.extend(i.to_map())
        print(content)


if __name__ == '__main__':
    start = time.time()
    pdf_processor = PDFProcessor(device="cpu", zoom_factor=3,
                                 model_source="/home/zhongbing/Projects/MLE/Document-AI/yolo-document-layout-analysis/runs/detect/train63/weights/best.pt")
    output_pages = pdf_processor.process("./pdf/test4.pdf", use_ocr=False)
    print("Time taken: ", time.time() - start)
    blocks = []
    for page in output_pages:
        blocks.extend(page.blocks)

    empty_block = []
    for block in blocks:
        if block.content is None:
            empty_block.append(block)

    markdown_content = pdf_processor.convert_to_markdown(output_pages)

    pdf_processor.merge(output_pages)

    # print(layouts)
    print("hell")
