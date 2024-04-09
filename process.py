import io

import fitz
from PIL import Image
from ultralytics import YOLO

from entity.page import Page
from util.visualizer import Visualizer
from rapid_orientation import RapidOrientation


class PDFProcessor:
    model = YOLO(
        "/Users/zhongbing/Projects/MLE/Doc-AI/model/yolo/best.pt")
    orientation_engine = RapidOrientation()

    def __init__(self, pdf_path, zoom_factor=3):
        self.pages = None
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
            orientation_res, elapse = cls.orientation_engine(img_bytes)
            rotated_angle = int(orientation_res)
            img_rotated = img
            # rotate
            if rotated_angle in [90, 270]:
                img_rotated = img.rotate(rotated_angle, expand=True)
                new_rotation = (360 - (page.rotation - rotated_angle)) % 360
                page.set_rotation(new_rotation)
            page = Page(page_num=page_num, image=img_rotated, pdf_page=page, zoom_factor=zoom_factor,
                        rotated_angle=rotated_angle)
            # append the image to the list
            images.append(page)
        # close the pdf file
        # return the images
        return images

    def detect_layout(self, image):
        # import the necessary libraries
        # load the model
        results = self.model.predict(image, conf=0.5, iou=0.45)
        # return the model
        return results[0]

    def process(self):
        doc = fitz.open(self.pdf_path)
        pages = self.convert_to_images(doc, self.zoom_factor)

        for i in range(len(pages)):
            page = pages[i]
            # layout part
            layout = self.detect_layout(page.image)
            # layouts.append(layout)
            page.build_items(layout)

            # filter item by label
            page.filter_items_by_label(filters=["Header", "Footer"])

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


if __name__ == '__main__':
    pdf_processor = PDFProcessor("./pdf/test2.pdf")
    layouts = pdf_processor.process()
    markdown_content = pdf_processor.convert_to_markdown()
    Visualizer.depict_bbox(layouts)
    # print(layouts)
