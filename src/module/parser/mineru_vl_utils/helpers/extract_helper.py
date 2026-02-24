"""Content extraction helpers - COPY from MinerUClientHelper (NO CHANGES)"""
import asyncio
import math
from concurrent.futures import Executor
from PIL import Image

from ..structs import ContentBlock, BLOCK_TYPES
from ..vlm_client.base_client import SamplingParams
from ..vlm_client.utils import get_png_bytes, get_rgb_image


class ExtractHelper:
    """COPY all methods from MinerUClientHelper (NO LOGIC CHANGES)"""

    def __init__(
        self,
        backend: str,
        prompts: dict[str, str],
        sampling_params: dict[str, SamplingParams],
        min_image_edge: int = 28,
        max_image_edge_ratio: float = 50,
    ):
        self.backend = backend
        self.prompts = prompts
        self.sampling_params = sampling_params
        self.min_image_edge = min_image_edge
        self.max_image_edge_ratio = max_image_edge_ratio

    def resize_by_need(self, image: Image.Image) -> Image.Image:
        """Original method (COPY AS IS)"""
        edge_ratio = max(image.size) / min(image.size)
        if edge_ratio > self.max_image_edge_ratio:
            width, height = image.size
            if width > height:
                new_w, new_h = width, math.ceil(width / self.max_image_edge_ratio)
            else:  # width < height
                new_w, new_h = math.ceil(height / self.max_image_edge_ratio), height
            new_image = Image.new(image.mode, (new_w, new_h), (255, 255, 255))
            new_image.paste(image, (int((new_w - width) / 2), int((new_h - height) / 2)))
            image = new_image
        if min(image.size) < self.min_image_edge:
            scale = self.min_image_edge / min(image.size)
            new_w, new_h = round(image.width * scale), round(image.height * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        return image

    def prepare_for_extract(
        self,
        image: Image.Image,
        blocks: list[ContentBlock],
        not_extract_list: list[str] | None = None,
    ) -> tuple[list[Image.Image | bytes], list[str], list[SamplingParams | None], list[int]]:
        """Original method (COPY AS IS) - Only change: remove equation_block from skip_list"""
        image = get_rgb_image(image)
        width, height = image.size
        block_images: list[Image.Image | bytes] = []
        prompts: list[str] = []
        sampling_params: list[SamplingParams | None] = []
        indices: list[int] = []
        skip_list = {"image", "list"}  # Keep equations as plain text, don't skip
        if not_extract_list:
            for not_extract_type in not_extract_list:
                if not_extract_type in BLOCK_TYPES:
                    skip_list.add(not_extract_type)
        for idx, block in enumerate(blocks):
            if block.type in skip_list:
                continue  # Skip blocks that should not be extracted.
            x1, y1, x2, y2 = block.bbox
            scaled_bbox = (x1 * width, y1 * height, x2 * width, y2 * height)
            block_image = image.crop(scaled_bbox)
            if block.angle in [90, 180, 270]:
                block_image = block_image.rotate(block.angle, expand=True)
            block_image = self.resize_by_need(block_image)
            if self.backend == "http-client":
                block_image = get_png_bytes(block_image)
            block_images.append(block_image)
            prompt = self.prompts.get(block.type) or self.prompts["[default]"]
            prompts.append(prompt)
            params = self.sampling_params.get(block.type) or self.sampling_params.get("[default]")
            sampling_params.append(params)
            indices.append(idx)
        return block_images, prompts, sampling_params, indices

    # Batch methods (COPY AS IS)
    def batch_prepare_for_extract(
        self,
        executor: Executor | None,
        images: list[Image.Image],
        blocks_list: list[list[ContentBlock]],
        not_extract_list: list[str] | None = None,
    ) -> list[tuple[list[Image.Image | bytes], list[str], list[SamplingParams | None], list[int]]]:
        """Original method (COPY AS IS)"""
        if executor is None:
            return [
                self.prepare_for_extract(im, bls, not_extract_list)
                for im, bls in zip(images, blocks_list)
            ]
        return list(executor.map(
            self.prepare_for_extract,
            images,
            blocks_list,
            [not_extract_list] * len(images)
        ))

    # Async methods (COPY AS IS)
    async def aio_prepare_for_extract(
        self,
        executor: Executor | None,
        image: Image.Image,
        blocks: list[ContentBlock],
        not_extract_list: list[str] | None = None,
    ) -> tuple[list[Image.Image | bytes], list[str], list[SamplingParams | None], list[int]]:
        """Original method (COPY AS IS)"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            self.prepare_for_extract,
            image,
            blocks,
            not_extract_list,
        )
