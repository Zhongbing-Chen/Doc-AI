"""Layout detection helpers - COPY from MinerUClientHelper (NO CHANGES)"""
import asyncio
import re
from concurrent.futures import Executor
from PIL import Image

from ..structs import ContentBlock, BLOCK_TYPES
from ..vlm_client.utils import get_png_bytes, get_rgb_image

_layout_re = r"^<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(.*)$"
ANGLE_MAPPING = {
    "<|rotate_up|>": 0,
    "<|rotate_right|>": 90,
    "<|rotate_down|>": 180,
    "<|rotate_left|>": 270,
}


def _convert_bbox(bbox):
    """Original _convert_bbox (KEEP AS IS)"""
    bbox = tuple(map(int, bbox))
    if any(coord < 0 or coord > 1000 for coord in bbox):
        return None
    x1, y1, x2, y2 = bbox
    x1, x2 = (x2, x1) if x2 < x1 else (x1, x2)
    y1, y2 = (y2, y1) if y2 < y1 else (y1, y2)
    if x1 == x2 or y1 == y2:
        return None
    return list(map(lambda num: num / 1000.0, (x1, y1, x2, y2)))


def _parse_angle(tail: str):
    """Original _parse_angle (KEEP AS IS)"""
    for token, angle in ANGLE_MAPPING.items():
        if token in tail:
            return angle
    return None


class LayoutHelper:
    """COPY all methods from MinerUClientHelper (NO LOGIC CHANGES)"""

    def __init__(self, backend: str, layout_image_size: tuple[int, int]):
        self.backend = backend
        self.layout_image_size = layout_image_size

    def prepare_for_layout(self, image: Image.Image) -> Image.Image | bytes:
        """Original method (COPY AS IS)"""
        image = get_rgb_image(image)
        image = image.resize(self.layout_image_size, Image.Resampling.BICUBIC)
        if self.backend == "http-client":
            return get_png_bytes(image)
        return image

    def parse_layout_output(self, output: str) -> list[ContentBlock]:
        """Original method (COPY AS IS)"""
        blocks: list[ContentBlock] = []
        for line in output.split("\n"):
            match = re.match(_layout_re, line)
            if not match:
                print(f"Warning: line does not match layout format: {line}")
                continue
            x1, y1, x2, y2, ref_type, tail = match.groups()
            bbox = _convert_bbox((x1, y1, x2, y2))
            if bbox is None:
                print(f"Warning: invalid bbox in line: {line}")
                continue
            ref_type = ref_type.lower()
            if ref_type not in BLOCK_TYPES:
                print(f"Warning: unknown block type in line: {line}")
                continue
            angle = _parse_angle(tail)
            if angle is None:
                print(f"Warning: no angle found in line: {line}")
            blocks.append(ContentBlock(ref_type, bbox, angle=angle))
        return blocks

    # Batch methods (COPY AS IS)
    def batch_prepare_for_layout(
        self,
        executor: Executor | None,
        images: list[Image.Image],
    ) -> list[Image.Image | bytes]:
        """Original method (COPY AS IS)"""
        if executor is None:
            return [self.prepare_for_layout(im) for im in images]
        return list(executor.map(self.prepare_for_layout, images))

    def batch_parse_layout_output(
        self,
        executor: Executor | None,
        outputs: list[str],
    ) -> list[list[ContentBlock]]:
        """Original method (COPY AS IS)"""
        if executor is None:
            return [self.parse_layout_output(output) for output in outputs]
        return list(executor.map(self.parse_layout_output, outputs))

    # Async methods (COPY AS IS)
    async def aio_prepare_for_layout(
        self,
        executor: Executor | None,
        image: Image.Image,
    ) -> Image.Image | bytes:
        """Original method (COPY AS IS)"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.prepare_for_layout, image)

    async def aio_parse_layout_output(
        self,
        executor: Executor | None,
        output: str,
    ) -> list[ContentBlock]:
        """Original method (COPY AS IS)"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.parse_layout_output, output)
