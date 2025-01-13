import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List

import cv2
import fitz
import numpy as np
import torch
from ultralytics.engine.results import Results

from entity.block import Block
from entity.box import BoxUtil, OcrBlock
from module.layout.layout_constant import type_dict
from util.coordinate_util import CoordinateUtil
from util.gap_tree import GapTree
from module.text.text_parser import TextExtractor


@dataclass
class Page:
    """
    The page class

    Attributes:
        page_num: the page number
        blocks: the blocks in the page
        rotated_angle: the rotated angle of the page
        skewed_angle: the skewed angle of the page
        pdf_page: the pdf page object
        image: the image of the page
        zoom_factor: the zoom factor of the page
        is_scanned: the flag to indicate whether the page is scanned

    Methods:
        build_blocks: build the blocks from the results
        extract_text: extract text from the items
        recognize_table: recognize the table structure
        draw_bbox: draw the bounding box on the image
        sort: sort the items based on the bounding box
        filter_items_by_label: filter the items by the label
        merge_overlap_block: merge the overlap block based on the threshold and the IOB algorithm
        to_map: convert the page to a map
    """
    page_num: Union[None, int]
    blocks: Union[None, list]
    rotated_angle: Union[None, float]
    skewed_angle: Union[None, float]
    pdf_page: fitz.Page
    image: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor]
    zoom_factor: float
    is_scanned: bool
    ocr_blocks: List[OcrBlock]

    def __init__(self, pdf_page, page_num, image, zoom_factor, items=None, rotated_angle=None, skewed_angle=None,
                 is_scanned=False):
        self.pdf_page = pdf_page
        self.page_num = page_num
        self.blocks = items
        self.rotated_angle = rotated_angle
        self.image = image
        self.zoom_factor = zoom_factor
        self.is_scanned = is_scanned

        if self.is_scanned:
            self.raw_ocr_result = TextExtractor.ocr_all_image_result(self.image)
            self.ocr_blocks = OcrBlock.from_rapid_ocr(self.raw_ocr_result)
        else:
            self.raw_ocr_result = TextExtractor.detection_all_image_result(self.image)
            self.ocr_blocks = OcrBlock.from_rapid_ocr(self.raw_ocr_result)

        self.skewed_angle = skewed_angle

    def build_blocks(self, results: Results):
        """
        build the blocks from the results
        :param results: the results object
        :return: None
        """
        self.blocks = [Block.from_bbox(i, self.page_num, box) for i, box in
                       enumerate(results.boxes.data)]

    @classmethod
    def build_blocks_with_results(cls, results: Results):
        """
        build the blocks from the results
        :param results: the results object
        :return: None
        """
        return [Block.from_bbox(i, "", box) for i, box in
                enumerate(results.boxes.data)]

    # extract text from the items using pdf api fitz
    def extract_text(self):
        """
        Extract text from the items
        :return: None
        """

        if not self.is_scanned:
            for block in self.blocks:
                # extract the text from the block based on the type of the page, scanned or not

                # extract the text using the fitz api
                block.content = TextExtractor.parse_by_fitz(self.pdf_page, block.adjusted_bbox(self.zoom_factor))

                # recognize the table content based on the table structure
                self.recognize_table_content(block, self.pdf_page, self.zoom_factor, self.is_scanned)
        else:
            self.ocr_blocks = OcrBlock.from_rapid_ocr(self.raw_ocr_result)
            BoxUtil.match_box_to_ocr(self.blocks, self.ocr_blocks)

            for block in self.blocks:
                # recognize the table content based on the table structure
                self.recognize_table_content(block, self.pdf_page, self.zoom_factor, self.is_scanned, self.ocr_blocks)

    @classmethod
    def recognize_table_content(cls, block: Block, pdf_page, zoom_factor, is_scanned: bool,
                                ocr_blocks: List[OcrBlock] = None) -> None:
        """
        Recognize the table content
        :param pdf_page: the pdf page
        :param zoom_factor: the zoom factor
        :return: None
        """
        if block.label == "Table":
            # recognize table content
            print("Recognize Table Content")

            if is_scanned:
                BoxUtil.match_box_to_ocr(block.table_structure, ocr_blocks)

            else:
                for cell in block.table_structure:
                    cell.content = TextExtractor.parse_by_fitz(pdf_page, [coord / zoom_factor for coord in cell.bbox])

    def recognize_table(self, table_parser, save_path: str):
        """
        Recognize the table structure
        :param table_parser: the table parser
        :return: None
        """
        for block in self.blocks:
            block.recognize_table_structure(self.image, table_parser, save_path)

    @property
    def texts(self):
        return [item.markdown_content for item in self.blocks]

    def draw_bbox(self):
        """
        Draw the bounding box on the image
        :return: the image with bounding boxes
        """
        image = np.array(self.image)

        for item in self.blocks:
            # Get the bounding box coordinates and convert them to integers
            x1, y1, x2, y2 = map(int, item.bbox)

            # Draw the bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the item id on the image
            cv2.putText(image, str(item.block_id) + " " + item.label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0),
                        2)

            image = item.depict_table(image)

        # Return the image with bounding boxes
        return image

    def sort(self):
        """
        Sort the items based on the bounding box
        :return: None
        """
        # avoid blank page
        if len(self.blocks) == 0:
            return

        # sort the items based on the bounding box
        gtree = GapTree(lambda item: item.bbox)

        sorted_text_blocks = gtree.sort(self.blocks)

        # reassign the id of the items with the sorted order
        for i, item in enumerate(sorted_text_blocks):
            item.block_id = i

        # assign the sorted items to the page
        self.blocks = sorted_text_blocks

    def filter_items_by_label(self, filters=None) -> None:
        """
        Filter the items by the label
        :param filters: the filters
        :return: None
        """
        if filters is None:
            return
        # filter the items by the label
        items = []
        for item in self.blocks:
            if item.label not in filters:
                items.append(item)
        self.blocks = items

    def merge_overlap_block(self, threshold=0.8) -> None:
        """
        Merge the overlap block based on the threshold and the IOB algorithm instead of the NMS algorithm
        :param threshold: the threshold to determine the overlap
        :return: None
        """
        i = 0  # start index
        while i < len(self.blocks) - 1:
            cur_block: Block = self.blocks[i]  # 当前块 current block
            # inner loop to check all adjacent blocks, including those already merged
            j = i + 1
            while j < len(self.blocks):
                next_block: Block = self.blocks[j]  # 下一个块 next block
                # calculate the overlap of the two blocks and the ratio of the overlap area
                ratio_a, ratio_b = CoordinateUtil.iob(cur_block.bbox, next_block.bbox)
                # if there is an overlap and the ratio of the overlap area is over 90%
                if ratio_a >= threshold or ratio_b >= threshold:
                    # determine the type of the merged block based on the predefined priority
                    cur_block.union(next_block)  # 合并当前块和下一个块 merge the current block and the next block
                    del self.blocks[j]  # 删除下一个块，因为它已经被合并 delete the next block as it has been merged
                else:
                    # if there is no overlap or the overlap is not big enough, move to the next adjacent block
                    j += 1
            # if the current block is merged with any adjacent block, update the index of the outer loop
            i += 1

    def add_text_layer(self) -> None:

        """
        Write OCR results to PDF with appropriate font sizes.
        ocr_data: list of [bbox, text, confidence] items
        """
        # Open the PDF
        if not self.is_scanned:
            return
        for item in self.raw_ocr_result:
            bbox, text, confidence = item
            # Calculate font size
            font_size = self.calculate_font_size(bbox, text)

            # Convert bbox to fitz rectangle format (x0, y0, x2, y2)
            rect = fitz.Rect(bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1])

            # Create text insertion object
            text_kwargs = {
                "fontname": "helvetica",  # Use a standard font
                "fontsize": font_size
            }

            # Insert text
            self.pdf_page.insert_text(rect.tl, text, **text_kwargs)

    @staticmethod
    def calculate_font_size(bbox, text):
        """
        Calculate appropriate font size based on bounding box and text length.
        bbox: list of coordinates [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        """
        # Calculate box width and height
        width = math.sqrt((bbox[1][0] - bbox[0][0]) ** 2 + (bbox[1][1] - bbox[0][1]) ** 2)
        height = math.sqrt((bbox[3][0] - bbox[0][0]) ** 2 + (bbox[3][1] - bbox[0][1]) ** 2)

        # Convert pixel to point (72 points = 1 inch, assume 96 DPI)
        width_pt = width * 72 / 96
        height_pt = height * 72 / 96

        # Estimate font size based on height, with some padding
        font_size_by_height = height_pt * 0.7

        # Estimate font size based on width and text length
        # Assume average character width is 0.6 times the font size
        char_count = len(text)
        font_size_by_width = width_pt / (char_count * 0.6)

        # Take the smaller of the two sizes to ensure text fits in both dimensions
        return min(font_size_by_height, font_size_by_width)

    def to_map(self):
        """
        Convert the page to a map
        :return: the map of the page
        """

        return [item.to_json() for item in self.blocks]

    def fix_block_using_ocr(self):
        for ocr_block in self.ocr_blocks:

            for block in self.blocks:
                overlap_area = BoxUtil.calculate_overlap_area(block, ocr_block)
                ocr_area = BoxUtil.calculate_box_area(ocr_block)

                if ocr_area > 0 and overlap_area > 0:
                    ocr_block.relationships[block.block_id] = overlap_area / ocr_area

        for ocr_block in self.ocr_blocks:
            if ocr_block.relationships:
                max_block_id = max(ocr_block.relationships, key=ocr_block.relationships.get)
                for block in self.blocks:
                    if block.block_id == max_block_id:
                        if ocr_block.relationships[max_block_id] < 0.8:
                            BoxUtil.extend_layout_box(block, ocr_block)

            else:
                # find key through value {0: 'Text', 1: 'Title', 2: 'Header', 3: 'Footer', 4: 'Figure', 5: 'Table', 6: 'Toc',
                #                  7: 'Figure caption', 8: 'Table caption'} get the Header's label
                label_id = self.get_key_from_value(type_dict, "Text")
                layout = Block(
                    block_id="",
                    x_1=ocr_block.x_1,
                    y_1=ocr_block.y_1,
                    x_2=ocr_block.x_2,
                    y_2=ocr_block.y_2,
                    label="Text",
                    label_id=label_id,
                    page_num=self.page_num,
                    layout_score=1.0,
                )

                self.blocks.append(layout)

    @staticmethod
    def get_key_from_value(dictionary, value):
        for key, val in dictionary.items():
            if val == value:
                return key
        return None
