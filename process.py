import io
import json
import multiprocessing
import os
import time
from datetime import datetime
from functools import partial

import fitz
import img2pdf
from PIL import Image

from entity.page import Page
from module.layout.layout_detector import LayoutDetector
from module.rotation.orientation_corrector import OrientationCorrector
from module.table.table_parser import TableExtractor
from module.text.text_parser import TextExtractor
from util.visualizer import Visualizer


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

    def process(self, file_path=None, document=None):
        """
        process the pdf file
        :param file_path: the path of the original pdf file
        :param language: the language of the text in the pdf, default is "chi_sim+eng"
        """
        # recognize the text in the pdf, and output the pdf file with the text layer
        pages = []

        current_time_string = datetime.now().strftime("%Y%m%d%H%M%S")
        dir_path_detail = f"./results/{current_time_string}/detail"
        dir_path_table = f"./results/{current_time_string}/table"
        if not os.path.exists(dir_path_detail) or not os.path.exists(dir_path_table):
            # 如果文件夹不存在，则创建
            os.makedirs(dir_path_detail, exist_ok=True)
            os.makedirs(dir_path_table, exist_ok=True)

        # open the updated pdf file
        if file_path is None:
            doc = document
        else:
            doc = fitz.open(file_path)

        for page_num in range(len(doc)):
            # convert the page to image
            processed_page: Page = self.process_one_page(dir_path_table, doc, page_num)

            # create PageResult using the page object
            compressed_page = processed_page.compress_copy()
            pages.append(compressed_page)
        Visualizer.depict_bbox(pages, dir_path_detail)
        doc.close()
        return pages

    def process_one_page(self, dir_path_table, doc, page_num):
        print("Processing page: ", page_num)
        start = time.time()
        page = doc[page_num]
        img, img_bytes = self.convert_page_to_image(page)
        img_rotated = img
        is_scanned = TextExtractor.is_scanned_pdf_page(page)
        # deprecated the pre_process function
        deskew_angle, img_rotated, rotated_angle = self.pre_process(img_rotated, page, is_scanned)
        # print(angle)
        # print(time.time() - start)
        page = Page(page_num=page_num, image=img_rotated, pdf_page=page, zoom_factor=self.zoom_factor,
                    rotated_angle=rotated_angle, skewed_angle=deskew_angle, is_scanned=is_scanned)
        layout = self.layout_detector.detect(page.image, conf=0.5, iou=0.45)
        # layouts.append(layout)
        page.build_blocks(layout)
        page.fix_block_using_ocr()
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
        page.add_text_layer()
        # append the image to the list

        # print the time usage
        print("Time taken for this page: ", time.time() - start)
        return page

    def process_img(self, image_path):
        # 转换图片为PDF bytes
        pdf_bytes = img2pdf.convert(image_path)

        # 直接加载到fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        return self.process(document=doc)

    def process_page(doc, self, dir_path_table, page_num):
        start = time.time()
        page = doc[page_num]
        img, img_bytes = self.convert_page_to_image(page)
        img_rotated = img
        is_scanned = TextExtractor.is_scanned_pdf_page(page)

        deskew_angle, img_rotated, rotated_angle = self.pre_process(img_rotated, page, is_scanned)

        page = Page(
            page_num=page_num,
            image=img_rotated,
            pdf_page=page,
            zoom_factor=self.zoom_factor,
            rotated_angle=rotated_angle,
            skewed_angle=deskew_angle,
            is_scanned=False
        )

        layout = self.layout_detector.detect(page.image, conf=0.5, iou=0.45)
        page.build_blocks(layout)
        page.filter_items_by_label(filters=["Header", "Footer", "Figure"])
        page.merge_overlap_block(threshold=0.7)
        page.sort()
        page.recognize_table(self.table_parser, dir_path_table)
        page.extract_text()
        page.add_text_layer()

        print(f"Time taken for page {page_num}: {time.time() - start}")
        return page

    def process_pdf_multiprocess(self, doc, dir_path_table, max_workers=None):
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()

        # Use partial to create a function with some arguments pre-filled
        process_page_partial = partial(
            self.process_page,
            doc,
            self,
            dir_path_table
        )

        # Use multiprocessing Pool to process pages in parallel
        with multiprocessing.Pool(processes=max_workers) as pool:
            pages = pool.map(process_page_partial, range(len(doc)))

        return pages

    @staticmethod
    def pre_process(img, page, is_scanned: bool = False):
        """
        pre-process the image
        :param img: the image object
        :param page: the page object
        :return: the deskew angle, the rotated image, the rotated angle
        """
        try:
            img_rotated, rotated_angle = OrientationCorrector.rotate_through_tesseract(img, page)
        except Exception as e:
            print(f"Error in rotating the image: {e}")
            img_rotated = img
            rotated_angle = 0

        if is_scanned:
            img_rotated, deskew_angle = OrientationCorrector.deskew_image(img_rotated)
        else:
            deskew_angle = 0

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

        # 将列表转换为JSON格式并保存
        json_content = json.dumps(content, ensure_ascii=False)

        # 指定JSON文件路径
        json_file_path = "json_output.json"

        # 将JSON内容写入文件
        with open(json_file_path, "w", encoding="utf-8") as file:
            file.write(json_content)
        print(content)


if __name__ == '__main__':
    start = time.time()
    pdf_processor = PDFProcessor(device="cpu", zoom_factor=3,
                                 model_source="/Users/zhongbing/Projects/MLE/Doc-AI/model/yolo/best.pt")
    # output_pages = pdf_processor.process("/Users/zhongbing/Projects/MLE/Doc-AI/pdf/ann/复宏汉霖important.PDF")
    output_pages = pdf_processor.process(
        "/Users/zhongbing/Projects/MLE/Doc-AI/pdf/test31.pdf")
    print("Time taken for this doc: ", time.time() - start)
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
