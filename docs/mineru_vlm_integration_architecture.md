# MinerU VLM Integration Architecture

## Problem Statement

The current DocAI system uses a modular pipeline approach for document parsing with specialized components for layout detection, OCR, and table extraction. We need to integrate MinerU's Vision Language Model (VLM) capabilities as an alternative parsing method while maintaining backward compatibility and enabling easy comparison between approaches.

## Current Architecture

### Processing Pipeline

```
PDF Input → Page Conversion → Image Processing
                                    ↓
                    ┌───────────────────────────────┐
                    │  Layout Detection (YOLO)      │
                    │  - Text, Title, Table, Figure │
                    └───────────────────────────────┘
                                    ↓
                    ┌───────────────────────────────┐
                    │  Block Creation               │
                    │  - Bounding boxes             │
                    │  - Label assignment           │
                    └───────────────────────────────┘
                                    ↓
        ┌───────────────────────────┴───────────────────┐
        ↓                                               ↓
┌──────────────────┐                          ┌──────────────────┐
│  OCR/Text        │                          │  Table           │
│  - RapidOCR      │                          │  - Transformer   │
│  - Fitz API      │                          │  - Structure     │
└──────────────────┘                          └──────────────────┘
        │                                               │
        └───────────────────┬─────────────────────────┘
                            ↓
                    ┌───────────────────┐
                    │  Page Assembly    │
                    │  - Blocks         │
                    │  - Content        │
                    │  - Metadata       │
                    └───────────────────┘
                            ↓
                    ┌───────────────────┐
                    │  Output           │
                    │  - JSON           │
                    │  - Markdown       │
                    └───────────────────┘
```

### Key Components

**Location**: `process.py`

```python
class PDFProcessor:
    def __init__(self, zoom_factor=3, device="cpu", model_source="huggingface"):
        self.layout_detector = LayoutDetector(model_source, device=device)
        self.table_parser = TableExtractor(model_source=model_source, device=device)

    def process_one_page(self, doc, page_num):
        # 1. Convert page to image
        img, img_bytes = self.convert_page_to_image(page)

        # 2. Detect layout
        layout = self.layout_detector.detect(page.image)

        # 3. Build blocks
        page.build_blocks(layout)

        # 4. Fix with OCR
        page.fix_block_using_ocr()

        # 5. Recognize tables
        page.recognize_table(self.table_parser, save_path)

        # 6. Extract text
        page.extract_text()
```

**Entity Model**:

- `Page` (`entity/page.py`) - Container for page content
- `Block` (`entity/block.py`) - Individual content blocks (text, table, figure)
- `TableStructure` - Table cell structure

**Output Format**:

```json
{
  "id": 0,
  "x_1": 100.0,
  "y_1": 200.0,
  "x_2": 500.0,
  "y_2": 300.0,
  "label": "Table",
  "content": "...",
  "table_structure": [...]
}
```

## Proposed Solution: Strategy Pattern with Multi-Parser Support

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      PDFProcessor                            │
│  - Orchestrates document processing                         │
│  - Delegates to parsing strategy                            │
│  - Maintains unified output format                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  DocumentParserStrategy (ABC)    │
        │  + parse() → List[Page]          │
        │  + supports_file_type()          │
        │  + get_metadata()                │
        └──────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┬──────────────┐
        ▼                             ▼              ▼
┌──────────────────┐        ┌──────────────────┐ ┌────────────┐
│ LegacyParser     │        │  MinerUVLMParser │ │ Future...  │
│ Strategy         │        │  Strategy        │ │ Parser     │
├──────────────────┤        ├──────────────────┤ ├────────────┤
│ LayoutDetector   │        │ MinerUVLMRunner  │ │ ...        │
│ TextExtractor    │        │ Result Adapter   │ │            │
│ TableExtractor   │        │                  │ │            │
└──────────────────┘        └──────────────────┘ └────────────┘
        │                             │
        └─────────────┬───────────────┘
                      ▼
        ┌──────────────────────────────────┐
        │  ResultNormalizer                │
        │  - Converts to Page/Block        │
        │  - Unified data structure        │
        └──────────────────────────────────┘
                      │
                      ▼
              ┌──────────────┐
              │ Page Objects │
              │ + Blocks     │
              │ + Content    │
              └──────────────┘
```

### Component Design

#### 1. Abstract Parser Interface

**File**: `module/parser/base_parser.py`

```python
from abc import ABC, abstractmethod
from typing import List
from entity.page import Page

class DocumentParserStrategy(ABC):
    """Abstract base class for document parsing strategies."""

    @abstractmethod
    def parse_document(
        self,
        file_path: str,
        output_dir: str,
        **kwargs
    ) -> List[Page]:
        """Parse document and return normalized Page objects."""
        pass

    @abstractmethod
    def supports_file_type(self, file_type: str) -> bool:
        """Check if strategy supports the given file type."""
        pass

    def get_metadata(self) -> dict:
        """Return strategy metadata."""
        return {}
```

#### 2. Legacy Parser Strategy

**File**: `module/parser/legacy_parser.py`

```python
class LegacyParserStrategy(DocumentParserStrategy):
    """
    Wraps existing modular pipeline implementation.
    Preserves current functionality without changes.
    """

    def __init__(self, zoom_factor=3, device="cpu", model_source="huggingface"):
        from process import PDFProcessor
        self.processor = PDFProcessor(
            zoom_factor=zoom_factor,
            device=device,
            model_source=model_source
        )

    def parse_document(self, file_path: str, output_dir: str, **kwargs) -> List[Page]:
        return self.processor.process(file_path)

    def supports_file_type(self, file_type: str) -> bool:
        return file_type.lower() == 'pdf'
```

#### 3. MinerU VLM Parser Strategy

**File**: `module/parser/mineru_vlm_parser.py`

```python
class MinerUVLMParserStrategy(DocumentParserStrategy):
    """
    Uses MinerU's Vision Language Model for document parsing.
    Leverages VLM capabilities for comprehensive understanding.
    """

    def __init__(self, model_mode="full", device_mode="auto", parse_method="auto"):
        from miner.mineru_vlm_runner import MinerUVLMRunner
        self.runner = MinerUVLMRunner(
            model_mode=model_mode,
            device_mode=device_mode
        )
        self.adapter = MinerUResultAdapter()

    def parse_document(self, file_path: str, output_dir: str, **kwargs) -> List[Page]:
        # Process with MinerU
        results = self.runner.process_pdf(
            pdf_path=file_path,
            output_dir=output_dir,
            method=kwargs.get("method", "auto")
        )

        # Adapt results to Page objects
        return self.adapter.adapt_results(results, file_path, output_dir)

    def supports_file_type(self, file_type: str) -> bool:
        return file_type.lower() in ['pdf', 'jpg', 'jpeg', 'png', 'docx', 'pptx']
```

#### 4. Result Adapter

**File**: `util/mineru_adapter.py`

```python
class MinerUResultAdapter:
    """
    Converts MinerU VLM output to common Page/Block format.
    Ensures consistency between different parsing strategies.
    """

    def adapt_results(self, results: Dict, file_path: str, output_dir: str) -> List[Page]:
        """Convert MinerU results to Page objects."""
        content_list_path = results["outputs"]["content_list"]

        with open(content_list_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)

        pages = []
        for page_data in content_list:
            page = self._build_page_from_mineru(page_data)
            pages.append(page)

        return pages

    def _build_page_from_mineru(self, page_data: Dict) -> Page:
        """Build Page object from MinerU page data."""
        blocks = []
        for item in page_data.get("layout_dets", []):
            block = self._build_block_from_mineru_item(item)
            blocks.append(block)

        return Page(
            pdf_page=None,
            page_num=page_data.get("page_idx", 0),
            image=None,
            zoom_factor=1.0,
            items=blocks
        )
```

#### 5. Enhanced PDFProcessor

**File**: `process.py` (refactored)

```python
class PDFProcessor:
    """Enhanced PDF processor supporting multiple parsing strategies."""

    def __init__(
        self,
        parser_strategy: Optional[DocumentParserStrategy] = None,
        zoom_factor=3,
        device="cpu",
        model_source="huggingface",
        visualize=False
    ):
        # Use provided strategy or default to legacy
        if parser_strategy is None:
            parser_strategy = LegacyParserStrategy(
                zoom_factor=zoom_factor,
                device=device,
                model_source=model_source
            )

        self.parser_strategy = parser_strategy
        self.visualize = visualize

    def set_parser_strategy(self, strategy: DocumentParserStrategy):
        """Switch parsing strategy at runtime."""
        self.parser_strategy = strategy

    def process(self, file_path: str = None, document=None, **kwargs) -> List[Page]:
        """Process document using configured strategy."""
        output_dir = f"./results/{datetime.now().strftime('%Y%m%d%H%M%S')}"

        pages = self.parser_strategy.parse_document(
            file_path=file_path,
            output_dir=output_dir,
            **kwargs
        )

        if self.visualize:
            self._visualize_pages(pages, output_dir)

        return pages
```

#### 6. Configuration System

**File**: `config/parsing_config.py`

```python
from enum import Enum

class ParserType(Enum):
    LEGACY = "legacy"
    MINERU_VLM = "mineru_vlm"
    AUTO = "auto"

@dataclass
class ParserConfig:
    """Configuration for document parsing."""

    parser_type: ParserType = ParserType.LEGACY

    # Legacy parser config
    zoom_factor: int = 3
    device: str = "cpu"
    model_source: str = "huggingface"

    # MinerU VLM config
    mineru_model_mode: str = "full"
    mineru_device_mode: str = "auto"
    mineru_parse_method: str = "auto"

    @classmethod
    def from_env(cls) -> 'ParserConfig':
        """Load configuration from environment variables."""
        import os
        return cls(
            parser_type=ParserType(os.getenv("PARSER_TYPE", "legacy")),
            zoom_factor=int(os.getenv("ZOOM_FACTOR", "3")),
            device=os.getenv("DEVICE", "cpu"),
            mineru_model_mode=os.getenv("MINERU_MODEL_MODE", "full")
        )

    def create_parser(self) -> DocumentParserStrategy:
        """Create parser instance based on configuration."""
        if self.parser_type == ParserType.LEGACY:
            return LegacyParserStrategy(
                zoom_factor=self.zoom_factor,
                device=self.device
            )
        elif self.parser_type == ParserType.MINERU_VLM:
            return MinerUVLMParserStrategy(
                model_mode=self.mineru_model_mode
            )
```

### Updated API

**File**: `api/app.py`

```python
from fastapi import FastAPI, File, UploadFile
from config.parsing_config import ParserConfig, ParserType

app = FastAPI()

@app.post("/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    parser: str = "legacy"  # legacy, mineru_vlm
):
    """Upload and parse document using specified parser."""

    config = ParserConfig.from_env()
    config.parser_type = ParserType(parser)

    processor = PDFProcessor(
        parser_strategy=config.create_parser()
    )

    # Process file...
    output_pages = processor.process(temp_file_path)
    structured_content = processor.merge(output_pages)

    return JSONResponse(content={
        "result": structured_content,
        "parser": parser,
        "metadata": processor.parser_strategy.get_metadata()
    })

@app.post("/parse/compare")
async def compare_parsers(file: UploadFile = File(...)):
    """Parse document with both parsers and compare results."""
    # Run both parsers and return comparison
    pass
```

## Directory Structure

```
DocAI/
├── module/
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── base_parser.py           # Abstract interface
│   │   ├── legacy_parser.py         # Current implementation
│   │   └── mineru_vlm_parser.py     # MinerU VLM strategy
│   ├── layout/
│   ├── table/
│   └── text/
├── util/
│   ├── mineru_adapter.py            # Result adapter
│   ├── visualizer.py
│   └── ...
├── config/
│   ├── __init__.py
│   └── parsing_config.py            # Configuration management
├── entity/
│   ├── page.py                      # Unchanged
│   ├── block.py                     # Unchanged
│   └── box.py                       # Unchanged
├── miner/
│   ├── mineru_vlm_runner.py         # Existing MinerU wrapper
│   └── MinerU/                      # MinerU library
├── process.py                       # Enhanced orchestrator
├── api/
│   └── app.py                       # Updated API
└── requirements.txt
```

## Usage Examples

### Basic Usage

```python
# Use legacy parser (default - backward compatible)
processor = PDFProcessor()
pages = processor.process("document.pdf")

# Use MinerU VLM parser
from module.parser.mineru_vlm_parser import MinerUVLMParserStrategy

mineru_strategy = MinerUVLMParserStrategy(model_mode="full")
processor.set_parser_strategy(mineru_strategy)
pages = processor.process("document.pdf")
```

### Configuration-Based

```python
# Set via environment variables
# export PARSER_TYPE=mineru_vlm
# export MINERU_MODEL_MODE=full

from config.parsing_config import ParserConfig

config = ParserConfig.from_env()
processor = PDFProcessor(parser_strategy=config.create_parser())
pages = processor.process("document.pdf")
```

### API Usage

```bash
# Legacy parser
curl -X POST "http://localhost:8000/upload-file?parser=legacy" \
  -F "file=@document.pdf"

# MinerU VLM parser
curl -X POST "http://localhost:8000/upload-file?parser=mineru_vlm" \
  -F "file=@document.pdf"

# Compare both parsers
curl -X POST "http://localhost:8000/parse/compare" \
  -F "file=@document.pdf"
```

## Key Design Decisions

### 1. Strategy Pattern
**Why**: Allows runtime switching between parsers without changing client code

**Benefits**:
- Easy to add new parsers (AWS Textract, GCP Document AI, etc.)
- Compare parser performance side-by-side
- A/B testing different approaches
- Gradual migration from legacy to VLM

### 2. Adapter Pattern
**Why**: MinerU output format differs from existing Page/Block structure

**Benefits**:
- Maintains unified output format
- No changes to downstream code
- Clean separation of concerns
- Easy to modify mapping logic

### 3. Backward Compatibility
**Why**: Existing code and API consumers should continue working

**Approach**:
- Legacy parser is default
- Existing `PDFProcessor` API unchanged
- Optional opt-in to VLM parser
- Gradual migration path

### 4. Configuration-Driven
**Why**: Flexibility in deployment and testing

**Benefits**:
- Environment-based parser selection
- Easy to switch between environments
- No code changes for configuration
- Support for feature flags

## Benefits

### 1. Flexibility
- Choose best parser per use case
- Easy to experiment with new parsing technologies
- Runtime parser switching

### 2. Performance Comparison
- Side-by-side accuracy evaluation
- Benchmark processing speed
- Resource usage comparison
- Quality metrics collection

### 3. Gradual Migration
- No breaking changes
- Incremental VLM adoption
- Risk mitigation
- Rollback capability

### 4. Maintainability
- Clean separation of concerns
- Each parser is independent
- Easy to test
- Clear ownership boundaries

### 5. Extensibility
- Add new parsers without modifying existing code
- Plugin architecture
- Future-proof for new technologies

## Migration Path

### Phase 1: Foundation (No Breaking Changes)
- [ ] Create `DocumentParserStrategy` interface
- [ ] Implement `LegacyParserStrategy` wrapper
- [ ] Refactor `PDFProcessor` to use strategies
- [ ] Add comprehensive tests
- [ ] Ensure backward compatibility

### Phase 2: MinerU Integration
- [ ] Implement `MinerUVLMParserStrategy`
- [ ] Create `MinerUResultAdapter`
- [ ] Add configuration system
- [ ] Update API endpoints
- [ ] Add documentation

### Phase 3: Testing & Validation
- [ ] Unit tests for each parser
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Accuracy comparisons
- [ ] Load testing

### Phase 4: Gradual Rollout
- [ ] Enable VLM in development environment
- [ ] A/B testing in staging
- [ ] Monitor metrics and quality
- [ ] Gather user feedback

### Phase 5: Production Adoption
- [ ] Make VLM default if proven superior
- [ ] Deprecation timeline for legacy
- [ ] Migration guides
- [ ] Training materials

## Testing Strategy

### Unit Tests
```python
def test_legacy_parser():
    strategy = LegacyParserStrategy()
    pages = strategy.parse_document("test.pdf", "/tmp")
    assert len(pages) > 0
    assert all(isinstance(p, Page) for p in pages)

def test_mineru_vlm_parser():
    strategy = MinerUVLMParserStrategy()
    pages = strategy.parse_document("test.pdf", "/tmp")
    assert len(pages) > 0
    assert all(isinstance(p, Page) for p in pages)
```

### Comparison Tests
```python
def test_parser_output_format_consistency():
    """Ensure both parsers produce same output format."""
    legacy = LegacyParserStrategy()
    mineru = MinerUVLMParserStrategy()

    legacy_pages = legacy.parse_document("test.pdf", "/tmp/legacy")
    mineru_pages = mineru.parse_document("test.pdf", "/tmp/mineru")

    # Compare structure
    assert len(legacy_pages) == len(mineru_pages)

    for lp, mp in zip(legacy_pages, mineru_pages):
        assert len(lp.blocks) == len(mp.blocks)
        assert all(b.label in ["Text", "Table", "Figure"] for b in lp.blocks)
```

### Performance Tests
```python
def test_parser_performance():
    """Benchmark parsing speed and resource usage."""
    import time

    start = time.time()
    legacy.parse_document("large.pdf", "/tmp")
    legacy_time = time.time() - start

    start = time.time()
    mineru.parse_document("large.pdf", "/tmp")
    mineru_time = time.time() - start

    print(f"Legacy: {legacy_time:.2f}s, MinerU: {mineru_time:.2f}s")
```

## Comparison: Legacy vs MinerU VLM

| Aspect | Legacy Parser | MinerU VLM Parser |
|--------|--------------|-------------------|
| **Approach** | Modular pipeline | End-to-end VLM |
| **Layout Detection** | YOLO-based | VLM understanding |
| **OCR** | RapidOCR | Built-in VLM OCR |
| **Table Recognition** | Table Transformer | VLM table understanding |
| **Formula Extraction** | Not supported | Supported |
| **Reading Order** | Gap-tree algorithm | VLM natural order |
| **Supported Formats** | PDF | PDF, Images, Office docs |
| **Accuracy** | Good | Potentially better |
| **Speed** | Fast | Slower (VLM inference) |
| **Resource Usage** | Lower | Higher (GPU recommended) |
| **Maturity** | Production-tested | Newer, evolving |

## Risk Assessment

### Low Risk
- Adding new code without changing existing
- Backward compatibility maintained
- Legacy parser remains default

### Medium Risk
- MinerU result format may change
- Adapter logic complexity
- Increased codebase maintenance

### Mitigation
- Comprehensive test coverage
- Version pinning for MinerU
- Feature flags for easy rollback
- Monitoring and alerting

## Success Criteria

1. **Backward Compatibility**: All existing tests pass
2. **Output Consistency**: Both parsers produce identical output structures
3. **Performance**: VLM parser completes within acceptable time limits
4. **Accuracy**: VLM parser meets or exceeds legacy accuracy
5. **Maintainability**: Code is well-documented and testable
6. **Extensibility**: Easy to add new parsers in the future

## Future Enhancements

1. **Additional Parsers**
   - AWS Textract
   - Google Cloud Document AI
   - Azure Form Recognizer
   - Tesseract

2. **Hybrid Approach**
   - Use different parsers for different content types
   - Fall back mechanisms
   - Ensemble methods

3. **Caching**
   - Cache parsed results
   - Invalidate on document changes
   - Reduce redundant processing

4. **Async Processing**
   - Background job queue
   - Real-time status updates
   - Batch processing support

5. **Quality Metrics**
   - Accuracy scoring
   - Confidence intervals
   - Manual review workflows

## References

- **MinerU**: https://github.com/opendatalab/MinerU
- **Existing Code**: `process.py`, `entity/page.py`, `miner/mineru_vlm_runner.py`
- **Design Patterns**: Strategy Pattern, Adapter Pattern, Factory Pattern

## Questions & Considerations

1. **Accuracy Trade-offs**: Is VLM accuracy improvement worth the performance cost?
2. **Resource Requirements**: Do we have GPU capacity for VLM in production?
3. **Migration Timeline**: How quickly do we want to adopt VLM?
4. **Fallback Strategy**: What happens when VLM fails?
5. **Cost Analysis**: What are the infrastructure costs for each approach?
6. **User Preferences**: Should users be able to choose parser?

## Conclusion

This architecture provides a clean, extensible way to integrate MinerU VLM capabilities while maintaining backward compatibility and enabling easy comparison between parsing approaches. The strategy pattern allows for runtime flexibility, while the adapter pattern ensures unified output formats. The phased migration approach minimizes risk and allows for gradual adoption based on real-world performance metrics.
