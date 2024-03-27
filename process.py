import io

import fitz
from PIL import Image
from ultralytics import YOLO

from entity.page import Page
from util.visualizer import Visualizer
from rapid_orientation import RapidOrientation


class PDFProcessor:
    model = YOLO(
        "/home/zhongbing/Projects/MLE/Document-AI/yolo-document-layout-analysis/layout_analysis/8mpt/v2/best.pt")
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
            markdown_content.append(page.texts)
        # convert it to markdown format
        return markdown_content




if __name__ == '__main__':
    pdf_processor = PDFProcessor("./pdf/test.pdf")
    layouts = pdf_processor.process()
    markdown_content = pdf_processor.convert_to_markdown()
    Visualizer.depict_bbox(layouts)
    print(layouts)
