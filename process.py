import io
import os
import time


import fitz
import ocrmypdf
from PIL import Image

from entity.page import Page
from module.layout.layout_detector import LayoutDetector
from module.rotation.orientation_corrector import OrientationCorrector
from module.table.table_parser import TableExtractor
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

    def process(self, file_path, language="chi_sim+eng"):
        """
        process the pdf file
        :param file_path: the path of the original pdf file
        :param language: the language of the text in the pdf, default is "chi_sim+eng"
        """
        # recognize the text in the pdf, and output the pdf file with the text layer
        file_path = self.pre_ocr(file_path, language=language)
        pages = []

        # open the updated pdf file
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            # convert the page to image
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
            page.filter_items_by_label(filters=["Header", "Footer"])

            # merge the overlap block
            page.merge_overlap_block(threshold=0.8)

            # sort the items
            page.sort()

            # table part
            page.recognize_table(self.table_parser)

            # text part
            page.extract_text()

            # append the image to the list
            pages.append(page)
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
    pdf_processor = PDFProcessor(device="cpu", zoom_factor=1, model_source="huggingface")
    output_pages = pdf_processor.process("./pdf/test3.pdf")
    markdown_content = pdf_processor.convert_to_markdown(output_pages)
    Visualizer.depict_bbox(output_pages)
    pdf_processor.merge(output_pages)
    print("Time taken: ", time.time() - start)
    # print(layouts)
    print("hell")
