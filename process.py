import io
import time

import fitz
import ocrmypdf
from PIL import Image

from entity.page import Page
from module.layout.layout_detector import LayoutDetector
from module.rotation.orientation_corrector import OrientationCorrector
from util.visualizer import Visualizer


class PDFProcessor:
    layout_detector = LayoutDetector(
        "/home/zhongbing/Projects/MLE/Document-AI/yolo-document-layout-analysis/layout_analysis/8mpt/v2/best.pt")

    def __init__(self, pdf_path, zoom_factor=3):
        self.pages: list[Page] = []
        self.pdf_path = pdf_path
        self.zoom_factor = zoom_factor

    def pre_ocr(self, output_file=None, language="eng", rotate_pages_threshold=3.0):
        if output_file is None:
            output_file = self.pdf_path.replace(".pdf", "_ocr.pdf")
        ocrmypdf.ocr(self.pdf_path, output_file, rotate_pages=True,
                     rotate_pages_threshold=rotate_pages_threshold, language=language,
                     deskew=True,
                     skip_text=True,
                     clean=True,
                     oversample=200
                     # force_ocr=True
                     )
        self.pdf_path = output_file

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

    def process(self):
        self.pre_ocr(language="chi_sim+eng")
        doc = fitz.open(self.pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            img, img_bytes = self.convert_page_to_image(page)
            img_rotated = img
            deskew_angle = 0
            rotated_angle = 0
            # deskew_angle, img_rotated, rotated_angle = self.pre_process(img_rotated, page)
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

    def pre_process(self, img, page):
        img_rotated, rotated_angle = OrientationCorrector.rotate_through_tesseract(img, page)
        print(rotated_angle)
        img_rotated, deskew_angle = OrientationCorrector.deskew_image(img_rotated)
        return deskew_angle, img_rotated, rotated_angle

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
    start = time.time()
    pdf_processor = PDFProcessor("./pdf/test2.pdf")
    pdf_processor.process()
    markdown_content = pdf_processor.convert_to_markdown()
    Visualizer.depict_bbox(pdf_processor.pages)
    pdf_processor.merge()
    print("Time taken: ", time.time() - start)
    # print(layouts)
