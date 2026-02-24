"""
Simplified MinerU client - uses helper modules

This replaces the original mineru_client.py with a cleaner structure
that uses the extracted helper modules.
"""
import asyncio
from concurrent.futures import Executor
from typing import Literal, Sequence

from PIL import Image

from .helpers import LayoutHelper, ExtractHelper, PostProcessHelper
from .structs import BLOCK_TYPES, ContentBlock
from .vlm_client import DEFAULT_SYSTEM_PROMPT, SamplingParams, new_vlm_client


# Keep original defaults
DEFAULT_PROMPTS: dict[str, str] = {
    "table": "\nTable Recognition:",
    "[default]": "\nText Recognition:",
    "[layout]": "\nLayout Detection:",
}

DEFAULT_SAMPLING_PARAMS: dict[str, SamplingParams] = {
    "table": SamplingParams(presence_penalty=1.0, frequency_penalty=0.005),
    "[default]": SamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[layout]": SamplingParams(),
}


class MinerUClient:
    """
    Simplified VLM client for document processing

    Uses helper modules instead of nested helper class.
    Keep all core methods from original MinerUClient.
    """

    def __init__(
        self,
        backend: Literal["http-client"] = "http-client",
        model_name: str | None = None,
        server_url: str | None = None,
        server_headers: dict[str, str] | None = None,
        prompts: dict[str, str] = DEFAULT_PROMPTS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: dict[str, SamplingParams] = DEFAULT_SAMPLING_PARAMS,
        layout_image_size: tuple[int, int] = (1036, 1036),
        min_image_edge: int = 28,
        max_image_edge_ratio: float = 50,
        abandon_list: bool = False,
        abandon_paratext: bool = False,
        incremental_priority: bool = False,
        max_concurrency: int = 100,
        executor: Executor | None = None,
        http_timeout: int = 600,
        use_tqdm: bool = True,
        debug: bool = False,
        max_retries: int = 3,
        retry_backoff_factor: float = 0.5,
    ):
        # Create VLM client
        self.client = new_vlm_client(
            backend=backend,
            model_name=model_name,
            server_url=server_url,
            server_headers=server_headers,
            system_prompt=system_prompt,
            max_concurrency=max_concurrency,
            http_timeout=http_timeout,
            use_tqdm=use_tqdm,
            debug=debug,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
        )

        # Create helpers (was MinerUClientHelper)
        self.layout_helper = LayoutHelper(
            backend=backend,
            layout_image_size=layout_image_size,
        )

        self.extract_helper = ExtractHelper(
            backend=backend,
            prompts=prompts,
            sampling_params=sampling_params,
            min_image_edge=min_image_edge,
            max_image_edge_ratio=max_image_edge_ratio,
        )

        self.post_helper = PostProcessHelper(
            abandon_list=abandon_list,
            abandon_paratext=abandon_paratext,
            debug=debug,
        )

        # Store config
        self.prompts = prompts
        self.sampling_params = sampling_params
        self.incremental_priority = incremental_priority
        self.max_concurrency = max_concurrency
        self.executor = executor
        self.use_tqdm = use_tqdm
        self.debug = debug
        self.abandon_list = abandon_list
        self.abandon_paratext = abandon_paratext

    # ============== LAYOUT DETECTION ==============

    def layout_detect(self, image: Image.Image, priority: int | None = None) -> list[ContentBlock]:
        """Detect layout blocks (KEEP ORIGINAL)"""
        layout_image = self.layout_helper.prepare_for_layout(image)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        output = self.client.predict(layout_image, prompt, params, priority)
        return self.layout_helper.parse_layout_output(output)

    def batch_layout_detect(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
    ) -> list[list[ContentBlock]]:
        """Batch layout detection (KEEP ORIGINAL)"""
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))

        layout_images = self.layout_helper.batch_prepare_for_layout(
            self.executor, images
        )

        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")

        outputs = self.client.batch_predict(layout_images, prompt, params, priority)

        return self.layout_helper.batch_parse_layout_output(self.executor, outputs)

    # ============== CONTENT EXTRACTION ==============

    def content_extract(
        self,
        image: Image.Image,
        type: str = "text",
        priority: int | None = None,
    ) -> str | None:
        """Extract content from single image/block (KEEP ORIGINAL)"""
        blocks = [ContentBlock(type, [0.0, 0.0, 1.0, 1.0])]

        block_images, prompts, params, _ = self.extract_helper.prepare_for_extract(
            image, blocks, None
        )

        if not (block_images and prompts and params):
            return None

        output = self.client.predict(block_images[0], prompts[0], params[0], priority)
        blocks[0].content = output

        blocks = self.post_helper.post_process(blocks)

        return blocks[0].content if blocks else None

    def batch_content_extract(
        self,
        images: list[Image.Image],
        types: Sequence[str] | str = "text",
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str | None]:
        """Batch content extraction (KEEP ORIGINAL)"""
        if isinstance(types, str):
            types = [types] * len(images)
        if len(types) != len(images):
            raise ValueError("Length of types must match length of images")

        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))

        blocks_list = [[ContentBlock(type, [0.0, 0.0, 1.0, 1.0])] for type in types]

        prepared = self.extract_helper.batch_prepare_for_extract(
            self.executor, images, blocks_list, None
        )

        all_images, all_prompts, all_params, all_indices = [], [], [], []

        for img_idx, (block_images, prompts, params, indices) in enumerate(prepared):
            all_images.extend(block_images)
            all_prompts.extend(prompts)
            all_params.extend(params)
            all_indices.extend([(img_idx, idx) for idx in indices])

        outputs = self.client.batch_predict(all_images, all_prompts, all_params, priority)

        for (img_idx, idx), output in zip(all_indices, outputs):
            blocks_list[img_idx][idx].content = output

        blocks_list = self.post_helper.batch_post_process(self.executor, blocks_list)

        return [blocks[0].content if blocks else None for blocks in blocks_list]

    # ============== BATCH LAYOUT EXTRACTION (NEW - replaces two_step_extract) ==============

    def batch_layout_extract(
        self,
        image: Image.Image,
        blocks: list[ContentBlock],
        priority: int | None = None,
        not_extract_list: list[str] | None = None,
    ) -> list[ContentBlock]:
        """
        Extract content from all blocks in one image

        This is what two_step_extract() was doing internally.
        Users call layout_detect() then this method.
        """
        block_images, prompts, params, indices = self.extract_helper.prepare_for_extract(
            image, blocks, not_extract_list
        )

        outputs = self.client.batch_predict(block_images, prompts, params, priority)

        for idx, output in zip(indices, outputs):
            blocks[idx].content = output

        return self.post_helper.post_process(blocks)

    # ============== BATCH PROCESSING FOR MULTIPLE PAGES ==============

    def batch_layout_extract_for_pages(
        self,
        images: list[Image.Image],
        not_extract_list: list[str] | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[list[ContentBlock]]:
        """
        Process multiple pages: detect layout + extract content

        This replaces the old concurrent_two_step_extract and stepping_two_step_extract
        """
        # Stage 1: Detect layout for all pages
        all_blocks = self.batch_layout_detect(images, priority)

        # Stage 2: Extract content from each page
        results = []
        for image, blocks in zip(images, all_blocks):
            blocks = self.batch_layout_extract(image, blocks, priority, not_extract_list)
            results.append(blocks)

        return results

    async def aio_batch_layout_extract_for_pages(
        self,
        images: list[Image.Image],
        not_extract_list: list[str] | None = None,
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        use_tqdm: bool = False,
        tqdm_desc: str | None = None,
    ) -> list[list[ContentBlock]]:
        """Async version for processing multiple pages"""
        from .vlm_client.utils import gather_tasks

        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)

        if priority is None:
            if self.incremental_priority:
                priority = list(range(len(images)))
            else:
                priority = [None] * len(images)

        # Stage 1: Detect layout for all pages
        tasks = [
            self._aio_layout_detect(img, prio, semaphore)
            for img, prio in zip(images, priority)
        ]

        all_blocks = await gather_tasks(
            tasks=tasks,
            use_tqdm=use_tqdm,
            tqdm_desc=tqdm_desc or "Layout Detection",
        )

        # Stage 2: Extract content from each page
        results = []
        for idx, (image, blocks) in enumerate(zip(images, all_blocks)):
            # Use the page-specific priority (or None)
            page_priority = priority[idx] if priority else None
            blocks = await self._aio_batch_layout_extract(image, blocks, page_priority, not_extract_list, semaphore)
            results.append(blocks)

        return results

    async def _aio_layout_detect(
        self,
        image: Image.Image,
        priority: int | None,
        semaphore: asyncio.Semaphore,
    ) -> list[ContentBlock]:
        """Async layout detect for single page"""
        async with semaphore:
            layout_image = await self.layout_helper.aio_prepare_for_layout(
                self.executor, image
            )

            prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
            params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")

            output = await self.client.aio_predict(layout_image, prompt, params, priority)

            return await self.layout_helper.aio_parse_layout_output(self.executor, output)

    async def _aio_batch_layout_extract(
        self,
        image: Image.Image,
        blocks: list[ContentBlock],
        priority: int | None,
        not_extract_list: list[str] | None,
        semaphore: asyncio.Semaphore,
    ) -> list[ContentBlock]:
        """Async batch layout extract for single page"""
        prepared = await self.extract_helper.aio_prepare_for_extract(
            self.executor, image, blocks, not_extract_list
        )

        block_images, prompts, params, indices = prepared

        outputs = await self.client.aio_batch_predict(
            block_images, prompts, params, priority, semaphore
        )

        for idx, output in zip(indices, outputs):
            blocks[idx].content = output

        return await self.post_helper.aio_post_process(self.executor, blocks)
