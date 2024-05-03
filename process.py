import io
import time

import fitz
import numpy as np
from PIL import Image
from jdeskew.estimator import get_angle
from jdeskew.utility import rotate
from rapid_orientation import RapidOrientation

from entity.page import Page
from module.layout.layout_detector import LayoutDetector
from module.rotation.orientation_corrector import OrientationCorrector
from util.visualizer import Visualizer


class PDFProcessor:
    layout_detector = LayoutDetector("/Users/zhongbing/Projects/MLE/Doc-AI/model/yolo/best.pt")

    def __init__(self, pdf_path, zoom_factor=3):
        self.pages: list[Page] = []
        self.pdf_path = pdf_path
        self.zoom_factor = zoom_factor

    # process the pdf into images
    @classmethod
    def convert_to_images(cls, doc, zoom_factor):
        # import the necessary libraries

        # open the pdf file
        # create a list to store the images
        images = []
        # loop through the pages
        for page_num in range(len(doc)):
            # get the page
            page = doc.load_page(page_num)
            # convert the page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor))
            # convert the image to numpy array
            # Convert the PyMuPDF pixmap into a bytes object
            img_bytes = pix.tobytes("ppm")  # Save as PPM format

            # Use io.BytesIO to convert bytes into a file-like object for PIL
            img_stream = io.BytesIO(img_bytes)

            # Use PIL to open the image
            img = Image.open(img_stream)

            img_rotated, rotated_angle = OrientationCorrector.adjust_rotation_angle(img, img_bytes, page)
            print(rotated_angle)

            img_rotated, deskew_angle = OrientationCorrector.deskew_image(img_rotated)
            # print(angle)
            # print(time.time() - start)
            page = Page(page_num=page_num, image=img_rotated, pdf_page=page, zoom_factor=zoom_factor,
                        rotated_angle=rotated_angle, skewed_angle=deskew_angle)
            # append the image to the list
            images.append(page)
        # close the pdf file
        # return the images
        return images

    def convert_page_to_image(self, page):
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

    def process_updated(self):
        doc = fitz.open(self.pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            img, img_bytes = self.convert_page_to_image(page)
            img_rotated, rotated_angle = OrientationCorrector.adjust_rotation_angle(img, img_bytes, page)
            print(rotated_angle)

            img_rotated, deskew_angle = OrientationCorrector.deskew_image(img_rotated)
            # print(angle)
            # print(time.time() - start)
            page = Page(page_num=page_num, image=img_rotated, pdf_page=page, zoom_factor=self.zoom_factor,
                        rotated_angle=rotated_angle, skewed_angle=deskew_angle)
            layout = self.layout_detector.detect(page.image, conf=0.5, iou=0.45)
            # layouts.append(layout)
            page.build_items(layout)

            # filter item by label
            page.filter_items_by_label(filters=["Header", "Footer"])

            # merge the overlap block
            page.merge_overlap_block(threshold=0.8)

            # sort the items
            page.sort()

            # table part
            page.recognize_table()

            # text part
            page.extract_text()

            # append the image to the list
            self.pages.append(page)

    def process(self):
        doc = fitz.open(self.pdf_path)
        pages = self.convert_to_images(doc, self.zoom_factor)

        for i in range(len(pages)):
            page = pages[i]
            # layout part
            layout = self.layout_detector.detect(page.image, conf=0.5, iou=0.45)
            # layouts.append(layout)
            page.build_items(layout)

            # filter item by label
            page.filter_items_by_label(filters=["Header", "Footer"])

            # merge the overlap block
            page.merge_overlap_block(threshold=0.8)

            # sort the items
            page.sort()

            # table part
            page.recognize_table()

            # text part
            page.extract_text()
            pages.append(page)

        doc.close()
        self.pages = pages
        return pages

    def convert_to_markdown(self):
        markdown_content = []
        for page in self.pages:
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

    def filter_items(self, filters=None):
        for page in self.pages:
            page.filter_items_by_label()

    def merge(self):
        content = []
        for i in self.pages:
            content.extend(i.to_map())
        print(content)


if __name__ == '__main__':
    pdf_processor = PDFProcessor("./pdf/test1.pdf")
    pdf_processor.process_updated()
    # markdown_content = pdf_processor.convert_to_markdown()
    Visualizer.depict_bbox(pdf_processor.pages)
    pdf_processor.merge()
    # print(layouts)
