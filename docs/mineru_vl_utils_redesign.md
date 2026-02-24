# MinerU VL Utils - Minimal Restructure Design

## Overview

**This is a NEW VLM-based document processor** that works alongside your existing processors (RapidOCR, TableTransformer).

The `mineru_vl_utils` module provides vision-language model based document analysis as an alternative to your existing OCR and model-based approaches.

**Your Current Stack:**
- `module/text/text_parser.py` - RapidOCR-based text extraction
- `module/table/table_parser.py` - TableTransformer-based table extraction
- `module/layout/layout_detector.py` - Your existing layout detection

**New VLM-Based Option:**
- `mineru_vl_utils/` - Remote VLM server API based processing

**Key Changes:**
- ✅ Keep original helper functions (no logic changes)
- ✅ Keep original `layout_detect` implementation
- ✅ Keep `content_extract` for single image/block extraction
- ❌ Remove `two_step_extract` wrapper (users can compose `layout_detect` + `batch_layout_extract`)
- ✅ Keep equations as plain text (remove equation processing logic, keep as-is)
- ❌ Remove GPU backends (keep only http-client)
- ✅ Reorganize into logical helper modules

---

## Current Structure

```
mineru_vl_utils/
├── mineru_client.py                 # Main client (788 lines) - NEEDS SPLITTING
│   ├── MinerUClientHelper class     # Helper functions
│   ├── MinerUClient class
│   └── All methods...
│
├── post_process/                    # Post-processing folder
│   ├── otsl2html.py                 # Table HTML (KEEP)
│   └── equation_*.py                # 7 equation files (DELETE)
│
└── vlm_client/
    ├── http_client.py               # (KEEP)
    ├── transformers_client.py       # (DELETE)
    ├── mlx_client.py                # (DELETE)
    ├── vllm_engine_client.py        # (DELETE)
    ├── vllm_async_engine_client.py  # (DELETE)
    └── lmdeploy_engine_client.py    # (DELETE)
```

---

## Proposed Structure

```
mineru_vl_utils/
├── __init__.py                      # Same exports
├── version.py                       # Same
├── structs.py                       # Same (ContentBlock, BlockType)
├── client.py                        # Simplified main client
│
├── helpers/                         # Extracted utilities (NO LOGIC CHANGES)
│   ├── __init__.py
│   ├── layout_helper.py             # Layout detection helpers
│   ├── extract_helper.py            # Content extraction helpers
│   └── post_process.py              # Post-processing (table only)
│
├── vlm_client/
│   ├── __init__.py
│   ├── base_client.py               # Same
│   ├── http_client.py               # Same
│   └── utils.py                     # Same
│
└── table_parser/
    └── otsl2html.py                 # Moved from post_process/
```

---

## Changes Summary

### 1. Extract Helper Functions (No Logic Changes)

**From `mineru_client.py` → `helpers/layout_helper.py`:**
- `prepare_for_layout()` - Keep original implementation
- `parse_layout_output()` - Keep original implementation
- `_convert_bbox()` - Keep original implementation
- `_parse_angle()` - Keep original implementation

**From `mineru_client.py` → `helpers/extract_helper.py`:**
- `resize_by_need()` - Keep original implementation
- `prepare_for_extract()` - Keep original implementation
- All batch preparation methods - Keep original implementation

**From `post_process/__init__.py` → `helpers/post_process.py`:**
- `post_process()` - Remove equation logic, keep table logic

### 2. Simplify Main Client

**Keep these core methods:**
- `layout_detect()` - Detect layout blocks (KEEP AS IS)
- `batch_layout_detect()` - Batch layout detection (KEEP AS IS)
- `content_extract()` - Extract from single image/block (KEEP AS IS)
- `batch_content_extract()` - Batch content extraction (KEEP AS IS)

**Remove wrapper method:**
- ❌ `two_step_extract()` - Users compose `layout_detect()` + `batch_layout_extract()`

**Add new method:**
- ✅ `batch_layout_extract(image, blocks)` - Extract content from all blocks in one image

### 3. Remove Equation Processing

Delete these files:
- `post_process/equation_block.py`
- `post_process/equation_big.py`
- `post_process/equation_double_subscript.py`
- `post_process/equation_fix_eqqcolon.py`
- `post_process/equation_leq.py`
- `post_process/equation_left_right.py`
- `post_process/equation_unbalanced_braces.py`

### 4. Remove GPU Backends

Delete these files:
- `vlm_client/transformers_client.py`
- `vlm_client/mlx_client.py`
- `vlm_client/vllm_engine_client.py`
- `vlm_client/vllm_async_engine_client.py`
- `logits_processor/` (entire folder)

---

## Implementation Details

### helpers/layout_helper.py

```python
"""Layout detection helpers - COPY from MinerUClientHelper (NO CHANGES)"""
import re
import asyncio
from concurrent.futures import Executor
from PIL import Image
from ..structs import ContentBlock, BLOCK_TYPES

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
        from ..vlm_client.utils import get_rgb_image, get_png_bytes
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
```

### helpers/extract_helper.py

```python
"""Content extraction helpers - COPY from MinerUClientHelper (NO CHANGES)"""
import math
import asyncio
from concurrent.futures import Executor
from typing import Sequence
from PIL import Image
from ..structs import ContentBlock, BLOCK_TYPES
from ..vlm_client.base_client import SamplingParams

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
        from ..vlm_client.utils import get_rgb_image, get_png_bytes

        image = get_rgb_image(image)
        width, height = image.size
        block_images: list[Image.Image | bytes] = []
        prompts: list[str] = []
        sampling_params: list[SamplingParams | None] = []
        indices: list[int] = []
        skip_list = {"image", "list"}  # Keep "equation" as plain text, don't skip
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
```

### helpers/post_process.py

```python
"""Post-processing - COPY from MinerUClientHelper, keep equations as plain text"""
import asyncio
from concurrent.futures import Executor
from ..structs import ContentBlock
from ..table_parser import otsl2html

PARATEXT_TYPES = {"header", "footer", "page_number", "aside_text", "page_footnote", "unknown"}

def post_process(
    blocks: list[ContentBlock],
    handle_equation_block: bool = False,  # Keep param but don't process
    abandon_list: bool = False,
    abandon_paratext: bool = False,
    debug: bool = False,
) -> list[ContentBlock]:
    """
    COPY from original post_process, keep equations as plain text

    Changes from original:
    - Keep equation blocks as plain text (no special processing)
    - Remove equation processing code (7 different equation fixers)
    - Keep table processing (COPY AS IS)
    - Keep filtering logic (COPY AS IS)
    """
    for block in blocks:
        # Table processing (COPY AS IS)
        if block.type == "table" and block.content:
            try:
                block.content = otsl2html.convert_otsl_to_html(block.content)
            except Exception as e:
                print("Warning: Failed to convert OTSL to HTML: ", e)
                print("Content: ", block.content)

        # Equation: Keep as plain text, no processing
        # if block.type == "equation" or block.type == "equation_block":
        #     Keep block.content as-is (plain text from VLM)
        #     No LaTeX bracket wrapping
        #     No equation fixing
        #     Just pass through

    # Removed: all equation processing (7 different equation fixers)

    # Removed: equation_block merging logic

    # Removed: add equation brackets (\\[ ... \\])

    # Filter unwanted blocks (COPY AS IS, but keep equations)
    out_blocks: list[ContentBlock] = []
    for block in blocks:
        # Keep equation blocks (don't drop them)
        if block.type == "equation_block":
            # Convert equation_block to equation for consistency
            block["type"] = "equation"
        if abandon_list and block.type == "list":
            continue
        if abandon_paratext and block.type in PARATEXT_TYPES:
            continue
        out_blocks.append(block)

    return out_blocks

class PostProcessHelper:
    """Helper class for batch/async post-processing (COPY from MinerUClientHelper)"""

    def __init__(
        self,
        abandon_list: bool = False,
        abandon_paratext: bool = False,
        debug: bool = False,
    ):
        self.abandon_list = abandon_list
        self.abandon_paratext = abandon_paratext
        self.debug = debug

    def post_process(self, blocks: list[ContentBlock]) -> list[ContentBlock]:
        """Original method (COPY AS IS, equations as plain text)"""
        try:
            return post_process(
                blocks,
                handle_equation_block=False,  # Don't process equations, keep as-is
                abandon_list=self.abandon_list,
                abandon_paratext=self.abandon_paratext,
                debug=self.debug,
            )
        except Exception as e:
            print(f"Warning: post-processing failed with error: {e}")
            return blocks

    # Batch method (COPY AS IS)
    def batch_post_process(
        self,
        executor: Executor | None,
        blocks_list: list[list[ContentBlock]],
    ) -> list[list[ContentBlock]]:
        """Original method (COPY AS IS)"""
        if executor is None:
            return [self.post_process(blocks) for blocks in blocks_list]
        return list(executor.map(self.post_process, blocks_list))

    # Async method (COPY AS IS)
    async def aio_post_process(
        self,
        executor: Executor | None,
        blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        """Original method (COPY AS IS)"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.post_process, blocks)
```

### client.py (Simplified)

```python
"""Main VLM client - simplified, uses helpers"""
from concurrent.futures import Executor
from typing import Literal, Sequence
from PIL import Image

from .structs import ContentBlock
from .vlm_client import new_vlm_client, SamplingParams
from .helpers.layout_helper import LayoutHelper
from .helpers.extract_helper import ExtractHelper
from .helpers.post_process import post_process

# Keep original defaults
DEFAULT_PROMPTS = {
    "table": "\nTable Recognition:",
    "[default]": "\nText Recognition:",
    "[layout]": "\nLayout Detection:",
}

DEFAULT_SAMPLING_PARAMS = {
    "table": SamplingParams(presence_penalty=1.0, frequency_penalty=0.005),
    "[default]": SamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[layout]": SamplingParams(),
}

class MinerUClient:
    """
    Simplified VLM client for document processing

    Key changes from original:
    - Uses helper modules instead of nested helper class
    - Removed two_step_extract wrapper
    - Removed equation-related parameters
    - Removed GPU backend support
    - Keep all method signatures and logic
    """

    def __init__(
        self,
        backend: Literal["http-client"] = "http-client",
        model_name: str | None = None,
        server_url: str | None = None,
        server_headers: dict[str, str] | None = None,
        prompts: dict[str, str] = DEFAULT_PROMPTS,
        system_prompt: str = "You are a helpful assistant.",
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
        self.layout_helper = LayoutHelper(backend, layout_image_size)
        self.extract_helper = ExtractHelper(backend, min_image_edge, max_image_edge_ratio)

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

    def batch_layout_detect(self, images: list[Image.Image], priority: Sequence[int | None] | int | None = None) -> list[list[ContentBlock]]:
        """Batch layout detection (KEEP ORIGINAL)"""
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        layout_images = [self.layout_helper.prepare_for_layout(img) for img in images]
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        outputs = self.client.batch_predict(layout_images, prompt, params, priority)
        return [self.layout_helper.parse_layout_output(output) for output in outputs]

    # ============== CONTENT EXTRACTION ==============

    def content_extract(self, image: Image.Image, type: str = "text", priority: int | None = None) -> str | None:
        """Extract content from single image/block (KEEP ORIGINAL)"""
        blocks = [ContentBlock(type, [0.0, 0.0, 1.0, 1.0])]
        block_images, prompts, params, _ = self.extract_helper.prepare_for_extract(
            image, blocks, self.prompts, self.sampling_params
        )
        if not (block_images and prompts and params):
            return None
        output = self.client.predict(block_images[0], prompts[0], params[0], priority)
        blocks[0].content = output
        blocks = post_process(blocks, self.abandon_list, self.abandon_paratext, self.debug)
        return blocks[0].content if blocks else None

    def batch_content_extract(self, images: list[Image.Image], types: Sequence[str] | str = "text", priority: Sequence[int | None] | int | None = None) -> list[str | None]:
        """Batch content extraction (KEEP ORIGINAL)"""
        if isinstance(types, str):
            types = [types] * len(images)
        if len(types) != len(images):
            raise ValueError("Length of types must match length of images")
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        blocks_list = [[ContentBlock(type, [0.0, 0.0, 1.0, 1.0])] for type in types]
        all_images, all_prompts, all_params, all_indices = [], [], [], []
        for img_idx, (img, blocks) in enumerate(zip(images, blocks_list)):
            block_images, prompts, params, indices = self.extract_helper.prepare_for_extract(
                img, blocks, self.prompts, self.sampling_params
            )
            all_images.extend(block_images)
            all_prompts.extend(prompts)
            all_params.extend(params)
            all_indices.extend([(img_idx, idx) for idx in indices])
        outputs = self.client.batch_predict(all_images, all_prompts, all_params, priority)
        for (img_idx, idx), output in zip(all_indices, outputs):
            blocks_list[img_idx][idx].content = output
        blocks_list = [post_process(b, self.abandon_list, self.abandon_paratext, self.debug) for b in blocks_list]
        return [blocks[0].content if blocks else None for blocks in blocks_list]

    # ============== BATCH LAYOUT EXTRACTION (NEW) ==============

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
            image, blocks, self.prompts, self.sampling_params, not_extract_list
        )
        outputs = self.client.batch_predict(block_images, prompts, params, priority)
        for idx, output in zip(indices, outputs):
            blocks[idx].content = output
        return post_process(blocks, self.abandon_list, self.abandon_paratext, self.debug)
```

---

## Usage Examples

### Example 1: Full Pipeline (Replaces two_step_extract)

```python
from mineru_vl_utils import MinerUClient

client = MinerUClient(server_url="http://localhost:8000")

# Old way:
# blocks = client.two_step_extract(image)

# New way (same result, more explicit):
blocks = client.layout_detect(image)
blocks = client.batch_layout_extract(image, blocks)
```

### Example 2: Use with Your Existing Layout Detection

```python
# Your existing processors
from module.text.text_parser import TextExtractor
from module.table.table_parser import TableExtractor
from module.layout.layout_detector import LayoutDetector

# New VLM-based option
from mineru_vl_utils import MinerUClient

# Option A: Use your existing stack
your_layout_detector = LayoutDetector()
your_text_extractor = TextExtractor()  # RapidOCR
your_table_extractor = TableExtractor()  # TableTransformer

# Option B: Use VLM-based processor (NEW!)
vlm_client = MinerUClient(server_url="http://localhost:8000")

# Both work with the same document!
pdf_page = pdf_document[0]

# Your existing pipeline
layout = your_layout_detector.detect(pdf_page)
for block in layout.blocks:
    if block.type == "text":
        text = your_text_extractor.parse_by_ocr(block.image)
    elif block.type == "table":
        table = your_table_extractor.parse(block.image)

# VLM-based pipeline (alternative!)
layout = vlm_client.layout_detect(pdf_page.get_pixmap().to_pil())
layout = vlm_client.batch_layout_extract(pdf_page.get_pixmap().to_pil(), layout)
```

### Example 3: Extract from Single Block

```python
# Crop a block
block_image = pdf_page.crop(bbox)

# Extract content
text = client.content_extract(block_image, type="text")
table_html = client.content_extract(block_image, type="table")
```

---

## When to Use VLM vs Your Existing Processors

### Use VLM Processor (mineru_vl_utils) when:
- ✅ You have remote VLM server available
- ✅ You want better accuracy than OCR
- ✅ You don't want to manage local models
- ✅ You want simple HTML table output
- ✅ Processing documents with complex layouts

### Use Your Existing Processors when:
- ✅ You need fast local processing
- ✅ You have limited network connectivity
- ✅ You need fine-grained control over output
- ✅ You're processing sensitive documents (can't send to server)

### Hybrid Approach (Best of Both):

```python
class HybridDocumentProcessor:
    """Choose the best processor for each task"""

    def __init__(self):
        # Your local processors (fast, offline)
        self.text_ocr = TextExtractor()  # RapidOCR
        self.table_model = TableExtractor()  # TableTransformer

        # VLM processor (accurate, remote)
        self.vlm_client = MinerUClient(server_url="http://localhost:8000")

    def process_page(self, pdf_page, use_vlm=True):
        layout = your_layout_detector.detect(pdf_page)

        for block in layout.blocks:
            block_image = pdf_page.crop(block.bbox)

            if use_vlm:
                # Use VLM for better accuracy
                if block.type == "text":
                    block.content = self.vlm_client.content_extract(
                        block_image, type="text"
                    )
                elif block.type == "table":
                    block.content = self.vlm_client.content_extract(
                        block_image, type="table"
                    )
            else:
                # Use your local processors
                if block.type == "text":
                    result = self.text_ocr.ocr_all_image_result(block_image)
                    block.content = "\n".join([r[0] for r in result])
                elif block.type == "table":
                    table_data = self.table_model.parse(block_image)
                    block.content = table_to_html(table_data)

        return layout
```

---

## Batch-First Pipeline Design

### Processing Flow

```
PDF Document (100 pages)
    ↓
Convert to Images
    ↓
┌─────────────────────────────────────────┐
│ Stage 1: Layout Detection (Batch)       │
│ - Process pages in batches of 20        │
│ - Detect all blocks with bbox + type    │
│ - 5 batches for 100 pages               │
│ - Result: Layouts for all pages         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Stage 2: Content Extraction (Batch)     │
│ - For each page, extract all blocks     │
│ - Text blocks → batch text extraction   │
│ - Table blocks → batch table extraction │
│ - Configurable batch size for blocks    │
│ - Result: All blocks with content       │
└─────────────────────────────────────────┘
    ↓
Final Output: Complete structured document
```

### Implementation

```python
class BatchDocumentProcessor:
    """
    Batch-first document processor with configurable batch sizes

    Two-stage pipeline:
    1. Detect layout for all pages (configurable pages per batch)
    2. Extract content from all blocks (configurable blocks per batch)
    """

    def __init__(
        self,
        vlm_client: MinerUClient,
        layout_batch_size: int = 20,  # Pages per batch for layout detection
        extraction_batch_size: int = 50,  # Blocks per batch for content extraction
    ):
        self.vlm_client = vlm_client
        self.layout_batch_size = layout_batch_size
        self.extraction_batch_size = extraction_batch_size

    def process_document(
        self,
        images: list[Image.Image],
        not_extract_types: list[str] | None = None,
    ) -> list[list[ContentBlock]]:
        """
        Process entire document in two stages

        Args:
            images: List of page images
            not_extract_types: Block types to skip (e.g., ["header", "footer"])

        Returns:
            List of blocks for each page, with content extracted
        """
        print(f"Processing {len(images)} pages...")
        print(f"Layout batch size: {self.layout_batch_size}")
        print(f"Extraction batch size: {self.extraction_batch_size}")

        # Stage 1: Detect layout for all pages (in batches)
        print("\n=== Stage 1: Layout Detection ===")
        all_layouts = self._batch_detect_layout(images)

        # Stage 2: Extract content from all blocks (in batches)
        print("\n=== Stage 2: Content Extraction ===")
        all_results = self._batch_extract_content(images, all_layouts, not_extract_types)

        return all_results

    def _batch_detect_layout(
        self,
        images: list[Image.Image],
    ) -> list[list[ContentBlock]]:
        """
        Stage 1: Detect layout in batches

        Processes N pages at a time (configurable via layout_batch_size)
        """
        all_layouts = []

        # Process in batches
        num_batches = (len(images) + self.layout_batch_size - 1) // self.layout_batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.layout_batch_size
            end_idx = min(start_idx + self.layout_batch_size, len(images))
            batch_images = images[start_idx:end_idx]

            print(f"  Batch {batch_idx + 1}/{num_batches}: pages {start_idx + 1}-{end_idx}")

            # Batch layout detection
            batch_layouts = self.vlm_client.batch_layout_detect(batch_images)
            all_layouts.extend(batch_layouts)

        return all_layouts

    def _batch_extract_content(
        self,
        images: list[Image.Image],
        layouts: list[list[ContentBlock]],
        not_extract_types: list[str] | None = None,
    ) -> list[list[ContentBlock]]:
        """
        Stage 2: Extract content from all blocks (in batches)

        Groups blocks by type and extracts in batches for optimal throughput
        """
        from ..helpers.extract_helper import ExtractHelper
        from ..helpers.post_process import PostProcessHelper

        extract_helper = ExtractHelper(
            backend=self.vlm_client.backend,
            prompts=self.vlm_client.prompts,
            sampling_params=self.vlm_client.sampling_params,
            min_image_edge=28,
            max_image_edge_ratio=50,
        )

        post_helper = PostProcessHelper(
            abandon_list=self.vlm_client.abandon_list,
            abandon_paratext=self.vlm_client.abandon_paratext,
            debug=self.vlm_client.debug,
        )

        results = []

        for img_idx, (image, blocks) in enumerate(zip(images, layouts)):
            print(f"  Page {img_idx + 1}/{len(images)}: {len(blocks)} blocks", end="")

            # Filter unwanted block types
            if not_extract_types:
                blocks = [b for b in blocks if b.type not in not_extract_types]
                print(f" → {len(blocks)} blocks (filtered)")
            else:
                print()

            if not blocks:
                results.append(blocks)
                continue

            # Prepare blocks for extraction
            block_images, prompts, params, indices = extract_helper.prepare_for_extract(
                image, blocks, not_extract_types
            )

            if not block_images:
                results.append(blocks)
                continue

            # Extract content in batches
            all_outputs = []
            num_extraction_batches = (len(block_images) + self.extraction_batch_size - 1) // self.extraction_batch_size

            for batch_idx in range(num_extraction_batches):
                start_idx = batch_idx * self.extraction_batch_size
                end_idx = min(start_idx + self.extraction_batch_size, len(block_images))

                batch_images = block_images[start_idx:end_idx]
                batch_prompts = prompts[start_idx:end_idx]
                batch_params = params[start_idx:end_idx]

                # Batch predict
                outputs = self.vlm_client.batch_predict(
                    batch_images, batch_prompts, batch_params
                )
                all_outputs.extend(outputs)

            # Assign content to blocks
            for idx, output in zip(indices, all_outputs):
                blocks[idx].content = output

            # Post-process
            blocks = post_helper.post_process(blocks)
            results.append(blocks)

        return results
```

### Usage Example

```python
from mineru_vl_utils import MinerUClient, BatchDocumentProcessor
from PIL import Image

# Create VLM client
vlm_client = MinerUClient(
    backend="http-client",
    server_url="http://localhost:8000",
)

# Create batch processor with CONFIGURABLE batch sizes
processor = BatchDocumentProcessor(
    vlm_client=vlm_client,
    layout_batch_size=20,      # 20 pages at a time for layout
    extraction_batch_size=50,  # 50 blocks at a time for extraction
)

# Load document (100 pages)
images = [Image.open(f"page{i}.png") for i in range(1, 101)]

# Process entire document
print("Processing 100-page document...")
all_blocks = processor.process_document(
    images,
    not_extract_types=["header", "footer", "page_number"],
)

# Access results
for page_idx, blocks in enumerate(all_blocks):
    print(f"\nPage {page_idx + 1}:")
    for block in blocks:
        if block.content:
            print(f"  [{block.type}] {len(block.content)} chars")
```

### Output Example

```
Processing 100 pages...
Layout batch size: 20
Extraction batch size: 50

=== Stage 1: Layout Detection ===
  Batch 1/5: pages 1-20
  Batch 2/5: pages 21-40
  Batch 3/5: pages 41-60
  Batch 4/5: pages 61-80
  Batch 5/5: pages 81-100

=== Stage 2: Content Extraction ===
  Page 1/100: 15 blocks
  Page 2/100: 18 blocks
  ...
  Page 100/100: 12 blocks
```

### Configurable Batch Sizes

```python
# For high-latency / low-memory servers (smaller batches)
processor = BatchDocumentProcessor(
    vlm_client=vlm_client,
    layout_batch_size=10,   # Fewer pages per request
    extraction_batch_size=20,  # Fewer blocks per request
)

# For low-latency / high-memory servers (larger batches)
processor = BatchDocumentProcessor(
    vlm_client=vlm_client,
    layout_batch_size=50,   # More pages per request
    extraction_batch_size=100,  # More blocks per request
)

# Auto-tune based on document size
processor = BatchDocumentProcessor(
    vlm_client=vlm_client,
    layout_batch_size=min(50, len(images)),  # Up to 50 pages
    extraction_batch_size=min(100, estimated_total_blocks),  # Up to 100 blocks
)
```

---

## Refactoring Steps

1. Create `helpers/` directory
2. Extract `LayoutHelper` class (NO logic changes)
3. Extract `ExtractHelper` class (NO logic changes)
4. Move `post_process()` to `helpers/post_process.py` (remove equations)
5. Move `otsl2html.py` to `table_parser/`
6. Update `client.py` to use helpers
7. Remove equation files (7 files)
8. Remove GPU backend files (4 files)
9. Remove `logits_processor/` directory
10. Test that output is consistent

---

## Benefits

- ✅ **Better organization** - Code split into logical modules
- ✅ **Keep what works** - All original helper logic preserved
- ✅ **Remove complexity** - Delete equations and GPU backends
- ✅ **More flexible** - Users can compose methods themselves
- ✅ **Less code** - Remove wrapper methods
- ✅ **Same API** - User code doesn't need to change much
