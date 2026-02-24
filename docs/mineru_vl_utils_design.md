# MinerU VL Utils Module Design Document

## Overview

The `mineru_vl_utils` module is a Vision-Language Model (VLM) client library designed for document analysis tasks. It provides a unified interface for processing document images through layout detection and content extraction using various VLM backends.

**Version**: 0.1.18
**Location**: `/module/parser/mineru_vl_utils/`

---

## Architecture

### High-Level Structure

```
mineru_vl_utils/
├── __init__.py                 # Lazy loading public API
├── version.py                  # Version information
├── structs.py                  # Core data structures
├── mineru_client.py            # Main MinerU client
├── vlm_client/                 # VLM backend implementations
│   ├── __init__.py
│   ├── base_client.py          # Abstract base client
│   ├── http_client.py          # HTTP API client
│   ├── transformers_client.py  # HuggingFace transformers
│   ├── mlx_client.py           # Apple Silicon MLX engine
│   ├── vllm_engine_client.py   # vLLM synchronous engine
│   ├── vllm_async_engine_client.py  # vLLM async engine
│   ├── lmdeploy_engine_client.py    # LMDeploy engine
│   └── utils.py                # Utility functions
├── logits_processor/           # Custom logits processors
│   ├── vllm_v0_no_repeat_ngram.py
│   └── vllm_v1_no_repeat_ngram.py
└── post_process/               # Output post-processing
    ├── __init__.py
    ├── otsl2html.py            # OTSL to HTML conversion
    ├── equation_block.py       # Equation block handling
    ├── equation_big.py         # Big equation fixes
    ├── equation_double_subscript.py
    ├── equation_fix_eqqcolon.py
    ├── equation_leq.py
    ├── equation_left_right.py
    └── equation_unbalanced_braces.py
```

---

## Core Components

### 1. Data Structures (`structs.py`)

#### BlockType Class
Defines all supported document block types as constants:

| Type | Description |
|------|-------------|
| `TEXT` | Regular text content |
| `TITLE` | Section/paragraph headings |
| `TABLE` | Tabular data |
| `IMAGE` | Images/figures |
| `CODE` | Code blocks |
| `ALGORITHM` | Pseudocode/algorithm blocks |
| `EQUATION` | Mathematical formulas |
| `EQUATION_BLOCK` | Multi-line equations |
| `REF_TEXT` | Bibliographic references |
| `LIST` | Ordered/unordered lists |
| `HEADER` / `FOOTER` | Page headers/footers |
| `PAGE_NUMBER` | Page numbering |
| `PAGE_FOOTNOTE` | Footnotes |
| `ASIDE_TEXT` | Marginal text (binding area, etc.) |
| `TABLE_CAPTION` / `IMAGE_CAPTION` / `CODE_CAPTION` | Figure/table captions |
| `TABLE_FOOTNOTE` / `IMAGE_FOOTNOTE` | Table/image footnotes |
| `PHONETIC` | Pronunciation guides |
| `UNKNOWN` | Unclassified content |

#### ContentBlock Class
Dictionary-based container for document blocks with validation:

**Attributes:**
- `type` (str): Block type from BlockType constants
- `bbox` (list[float]): Normalized bounding box [x1, y1, x2, y2] in [0,1] range
- `angle` (Literal[None, 0, 90, 180, 270]): Rotation angle in degrees
- `content` (str | None): Extracted text/content

**Validation Rules:**
- Bounding box must have 4 coordinates
- All coordinates must be in [0, 1] range
- x1 < x2 and y1 < y2
- Angle must be one of: None, 0, 90, 180, 270

---

### 2. MinerUClient (`mineru_client.py`)

Main high-level API for document processing.

#### Supported Backends

| Backend | Description | Use Case |
|---------|-------------|----------|
| `http-client` | HTTP API calls to remote VLM server | Cloud/deployed models |
| `transformers` | HuggingFace transformers library | Local CPU/GPU inference |
| `mlx-engine` | Apple Silicon MLX framework | Mac optimization |
| `lmdeploy-engine` | LMDeploy VL async engine | High-performance serving |
| `vllm-engine` | vLLM synchronous engine | Batch processing |
| `vllm-async-engine` | vLLM V1 async engine | Concurrent requests |

#### Batching Modes

- **Concurrent Mode**: For `http-client`, `vllm-async-engine`, `lmdeploy-engine`
  - Processes pages independently with maximum parallelism
  - Better for I/O-bound operations

- **Stepping Mode**: For `transformers`, `vllm-engine`
  - Processes all pages through layout detection first, then extraction
  - Better for GPU memory management

#### Key Configuration Parameters

```python
layout_image_size: tuple[int, int] = (1036, 1036)  # Layout detection resolution
min_image_edge: int = 28                            # Minimum block size
max_image_edge_ratio: float = 50                    # Aspect ratio limit
handle_equation_block: bool = True                  # Merge multi-line equations
abandon_list: bool = False                          # Skip list blocks
abandon_paratext: bool = False                      # Skip header/footer
incremental_priority: bool = False                  # Auto-increment priorities
max_concurrency: int = 100                          # Concurrent request limit
batch_size: int = 0                                 # Batch size (0=auto)
http_timeout: int = 600                             # HTTP timeout in seconds
max_retries: int = 3                                # HTTP retry attempts
retry_backoff_factor: float = 0.5                   # Exponential backoff
```

#### Core Methods

##### Layout Detection

**`layout_detect(image, priority=None)`**
- Single image layout detection
- Returns: `list[ContentBlock]`

**`batch_layout_detect(images, priority=None)`**
- Batch layout detection
- Automatic batching based on backend

**`aio_layout_detect(image, priority, semaphore)`**
- Async single image layout detection

**`aio_batch_layout_detect(images, priority, semaphore)`**
- Async batch with concurrency control

##### Content Extraction

**`content_extract(image, type="text", priority=None)`**
- Extract content from entire image
- Returns: `str | None`

**`batch_content_extract(images, types="text", priority=None)`**
- Batch content extraction

##### Two-Step Extraction (Recommended)

**`two_step_extract(image, priority, not_extract_list)`**
1. Detect layout blocks
2. Extract content from each block
3. Returns: `list[ContentBlock]`

**`batch_two_step_extract(images, priority, not_extract_list)`**
- Automatic batching mode selection
- Most efficient for multi-page documents

---

### 3. VLM Client Backend (`vlm_client/`)

#### Base Client (`base_client.py`)

**SamplingParams Dataclass:**
```python
@dataclass
class SamplingParams:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    no_repeat_ngram_size: int | None = None
    max_new_tokens: int | None = None
```

**VlmClient Abstract Interface:**
- `predict(image, prompt, params, priority)` → str
- `batch_predict(images, prompts, params, priority)` → list[str]
- `aio_predict(image, prompt, params, priority)` → Awaitable[str]
- `aio_batch_predict(images, prompts, params, semaphore)` → Awaitable[list[str]]

#### HTTP Client (`http_client.py`)

**Features:**
- OpenAI-compatible API format
- Automatic model name detection
- Retry logic with exponential backoff
- Streaming support (SSE)
- Environment variable support: `MINERU_VL_SERVER`

**Request Format:**
```json
{
  "model": "model-name",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,...", "detail": "high"}},
      {"type": "text", "text": "..."}
    ]}
  ],
  "temperature": 0.0,
  "max_completion_tokens": 4096,
  "priority": 1
}
```

#### VLLM Engine Client (`vllm_engine_client.py`)

**Features:**
- Direct vLLM engine integration
- Batch processing with configurable batch size
- Automatic chat template application
- Custom logits processor support

**Flow:**
1. Convert PIL Images to RGB format
2. Apply chat template to messages
3. Build vLLM sampling params
4. Process in batches
5. Extract text from RequestOutput

---

### 4. Logits Processor (`logits_processor/`)

#### VllmV1NoRepeatNGramLogitsProcessor

**Purpose:** Prevent n-gram repetition in generated output

**Mechanism:**
- Tracks n-gram occurrences during generation
- Bans tokens that would repeat existing n-grams
- Configurable via `no_repeat_ngram_size` parameter

**Extra Args:**
- `no_repeat_ngram_size` (int): Size of n-grams to prevent repeating
- `debug` (bool): Enable debug logging

---

### 5. Post-Processing (`post_process/`)

#### Processing Pipeline

```
Raw VLM Output
    ↓
Table OTSL → HTML Conversion
    ↓
Equation Fixes (multiple passes)
    ↓
Equation Block Merging
    ↓
Add Equation Brackets
    ↓
Filter Unwanted Blocks
    ↓
Final Output
```

#### Equation Processing Steps

1. **`try_match_equation_left_right`**: Fix `\left` and `\right` matching
2. **`try_fix_equation_double_subscript`**: Fix double subscript issues
3. **`try_fix_equation_eqqcolon`**: Fix `=:` and similar patterns
4. **`try_fix_equation_big`**: Handle oversized equations
5. **`try_fix_equation_leq`**: Fix `\leq` and similar commands
6. **`try_fix_unbalanced_braces`**: Balance curly braces

#### Equation Block Merging

**Purpose:** Combine consecutive `equation` blocks into `equation_block`

**Logic:**
- Group adjacent equation blocks (vertical proximity)
- Merge content with newlines
- Replace original blocks with merged block

---

## Usage Patterns

### Basic Usage

```python
from mineru_vl_utils import MinerUClient
from PIL import Image

# Initialize client
client = MinerUClient(
    backend="http-client",
    server_url="http://localhost:8000"
)

# Single page processing
image = Image.open("page1.png")
blocks = client.two_step_extract(image)

for block in blocks:
    print(f"{block.type}: {block.bbox} -> {block.content[:50]}...")
```

### Batch Processing

```python
# Load multiple pages
images = [Image.open(f"page{i}.png") for i in range(1, 11)]

# Process with automatic batching
all_blocks = client.batch_two_step_extract(images)

# Access results
for page_blocks in all_blocks:
    for block in page_blocks:
        if block.type == "table":
            print(f"Table found: {block.content}")
```

### Async Processing

```python
import asyncio

async def process_pages():
    images = [Image.open(f"page{i}.png") for i in range(1, 11)]
    blocks = await client.aio_batch_two_step_extract(images)
    return blocks

results = asyncio.run(process_pages())
```

### Custom Prompts and Parameters

```python
client = MinerUClient(
    backend="vllm-engine",
    model_path="/path/to/model",
    prompts={
        "table": "\nTable Recognition:",
        "equation": "\nFormula Recognition:",
        "[layout]": "\nLayout Detection:",
    },
    sampling_params={
        "table": MinerUSamplingParams(
            temperature=0.0,
            presence_penalty=1.0,
            frequency_penalty=0.005
        ),
    }
)
```

---

## Design Strengths

1. **Backend Flexibility**: Easy switching between 6 different backends
2. **Lazy Loading**: Reduces initial import overhead
3. **Type Hints**: Comprehensive type annotations throughout
4. **Error Handling**: Custom exception types (RequestError, ServerError, UnsupportedError)
5. **Async Support**: Native async/await patterns for concurrent operations
6. **Batch Optimization**: Automatic batching strategy selection
7. **Post-Processing**: Comprehensive output cleanup pipeline
8. **Validation**: Strong data validation in ContentBlock
9. **Progress Tracking**: tqdm integration for long operations
10. **Retry Logic**: Built-in HTTP retry with exponential backoff

---

## Design Considerations for Redesign

### Potential Issues

1. **Tight Coupling**: `MinerUClient` tightly coupled to specific output format
2. **Large MinerUClient Class**: 788 lines, multiple responsibilities
3. **Post-Processing Complexity**: Chain of 7 equation fixers is fragile
4. **Global State**: Default prompts/params as module constants
5. **Limited Extensibility**: Adding new block types requires modifying multiple files
6. **No Streaming**: Async streaming commented out in http_client.py
7. **Mixed Abstractions**: Low-level bbox handling mixed with high-level API

### Suggested Improvements

1. **Separate Concerns**:
   - Extract layout parsing to separate class
   - Create pipeline for post-processing steps
   - Separate image preprocessing logic

2. **Plugin Architecture**:
   - Pluggable post-processing modules
   - Configurable block type handlers
   - Extensible prompt templates

3. **Configuration Management**:
   - Configuration class instead of many constructor args
   - YAML/JSON config file support
   - Profile-based configurations

4. **Error Recovery**:
   - Graceful degradation on post-processing failures
   - Fallback strategies for different backends
   - Better logging/debugging hooks

5. **Testing Support**:
   - Mock backends for testing
   - Deterministic sampling modes
   - Validation utilities

6. **Performance Optimization**:
   - Image caching strategies
   - Connection pooling
   - Better memory management for large batches

---

## Dependencies

### Required
- `PIL` (Pillow): Image processing
- `httpx`: HTTP client
- `httpx-retries`: Retry logic
- `aiofiles`: Async file operations
- `tqdm`: Progress bars

### Backend-Specific
- `transformers`: HuggingFace transformers
- `mlx-vlm`: Apple Silicon MLX
- `vllm`: vLLM inference engine
- `lmdeploy`: LMDeploy serving
- `torch`: PyTorch (for logits processor)

---

## API Reference Summary

### Exports (`__init__.py`)

- `MinerUClient`: Main client class
- `MinerUSamplingParams`: Sampling parameters dataclass
- `MinerULogitsProcessor`: Alias for VllmV1NoRepeatNGramLogitsProcessor
- `__version__`: Package version string

### Lazy Loading Pattern
The module uses Python's `__getattr__` for lazy loading, reducing import overhead:
- Main classes only loaded when first accessed
- TYPE_CHECKING provides type hints without actual imports

---

## Examples of Output Format

### Layout Detection Output
```
<|box_start|>150 200 500 800<|box_end|><|ref_start|>table<|ref_end|><|rotate_up|>
<|box_start|>100 100 900 150<|box_end|><|ref_start|>title<|ref_end|><|rotate_up|>
<|box_start|>100 150 900 200<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>
```

### ContentBlock JSON Representation
```json
{
  "type": "table",
  "bbox": [0.15, 0.2, 0.5, 0.8],
  "angle": 0,
  "content": "<table>...</table>"
}
```

---

## Conclusion

The `mineru_vl_utils` module provides a comprehensive, production-ready solution for document understanding using vision-language models. Its architecture supports multiple deployment scenarios from local development to cloud serving, with strong abstractions for backend independence. The main areas for improvement revolve around reducing coupling, improving extensibility, and simplifying the configuration model.
