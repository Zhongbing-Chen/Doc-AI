from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import fitz
import numpy as np
import torch
from ultralytics.engine.results import Results

from entity.block import Block
from util.coordinate_util import CoordinateUtil
from util.gap_tree import GapTree
from module.text.text_parser import TextExtractor


@dataclass
class Page:
    page_num: Union[None, int]
    items: Union[None, list]
    rotated_angle: Union[None, float]
    skewed_angle: Union[None, float]
    pdf_page: fitz.Page
    image: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor]
    zoom_factor: float
    is_scanned: bool

    def __init__(self, pdf_page, page_num, image, zoom_factor, items=None, rotated_angle=None, skewed_angle=None):
        self.pdf_page = pdf_page
        self.page_num = page_num
        self.items = items
        self.rotated_angle = rotated_angle
        self.image = image
        self.zoom_factor = zoom_factor
        self.is_scanned = TextExtractor.is_scanned_pdf_page(pdf_page)
        self.skewed_angle = skewed_angle

    def build_items(self, results: Results):

        self.items = [Block.from_bbox(i, self.page_num, box) for i, box in
                      enumerate(results.boxes.data)]

    # extract text from the items using pdf api fitz
    def extract_text(self):
        for item in self.items:
            if not self.is_scanned:
                item.content = TextExtractor.parse_by_fitz(self.pdf_page, item.adjusted_bbox(self.zoom_factor))
            else:
                item.content = TextExtractor.parse_by_ocr(self.pdf_page, item.adjusted_bbox(self.zoom_factor))
            # recognize the table content based on the table structure
            item.recognize_table_content(self.pdf_page, self.zoom_factor)

    def recognize_table(self):
        for item in self.items:
            item.recognize_table_structure(self.image)

    @property
    def texts(self):
        return [item.markdown_content for item in self.items]

    def draw_bbox(self):
        image = np.array(self.image)

        for item in self.items:
            # Get the bounding box coordinates and convert them to integers
            x1, y1, x2, y2 = map(int, item.bbox)

            # Draw the bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the item id on the image
            cv2.putText(image, str(item.id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            image = item.depict_table(image)

        # Return the image with bounding boxes
        return image

    def sort(self):

        gtree = GapTree(lambda item: item.bbox)

        sorted_text_blocks = gtree.sort(self.items)

        # reassign the id of the items with the sorted order
        for i, item in enumerate(sorted_text_blocks):
            item.id = i

        # assign the sorted items to the page
        self.items = sorted_text_blocks

    def filter_items_by_label(self, filters=None):
        if filters is None:
            return

        items = []
        for item in self.items:
            if item.label not in filters:
                items.append(item)
        self.items = items

    def merge_overlap_block(self, threshold=0.8):

        i = 0  # 开始索引
        while i < len(self.items) - 1:
            cur_block: Block = self.items[i]  # 当前块
            # 内层循环用于检查所有相邻块，包括已合并的块
            j = i + 1
            while j < len(self.items):
                next_block: Block = self.items[j]  # 下一个块
                # 计算两个块的重叠情况和重叠区域的比例
                ratio_a, ratio_b = CoordinateUtil.iob(cur_block.bbox, next_block.bbox)
                # 如果存在重叠，并且重叠区域的比例超过90%
                if ratio_a >= threshold or ratio_b >= threshold:
                    # 确定合并后的块的类型，基于预定义的优先级
                    cur_block.union(next_block)  # 合并当前块和下一个块
                    del self.items[j]  # 删除下一个块，因为它已经被合并
                else:
                    # 如果没有重叠或重叠不足够大，移动到下一个相邻块
                    j += 1
            # 如果当前块与任何相邻块合并，更新外层循环的索引
            i += 1

    def to_map(self):
        return [item.to_json() for item in self.items]
