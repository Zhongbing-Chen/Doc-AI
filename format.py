"""
Document Parser - JSON Format Classes

This module defines a unified JSON schema for all document types.
All documents are parsed into this uniform JSON format.
"""

import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


class DocumentType(Enum):
    """Supported document types"""
    PDF = "pdf"
    WORD = "word"
    EXCEL = "excel"
    POWERPOINT = "powerpoint"
    MARKDOWN = "markdown"
    IMAGE = "image"


class BlockType(Enum):
    """Block types for content classification"""
    TEXT = "text"
    TITLE = "title"
    HEADING_1 = "heading_1"
    HEADING_2 = "heading_2"
    HEADING_3 = "heading_3"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"
    CODE = "code"
    QUOTE = "quote"
    HEADER = "header"
    FOOTER = "footer"
    TOC = "toc"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    UNKNOWN = "unknown"


@dataclass
class BBox:
    """Bounding box representation"""
    x_1: float
    y_1: float
    x_2: float
    y_2: float
    coordinate_system: str = "pixels"
    origin: str = "top_left"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_list(cls, coords: list) -> 'BBox':
        return cls(x_1=coords[0], y_1=coords[1], x_2=coords[2], y_2=coords[3])


@dataclass
class TableCell:
    """Table cell representation"""
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    content: str = ""
    is_header: bool = False

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "col": self.col,
            "row_span": self.row_span,
            "col_span": self.col_span,
            "content": self.content,
            "is_header": self.is_header
        }


@dataclass
class Content:
    """Content representation for blocks"""
    text: str = ""
    markdown: str = ""
    html: str = ""
    raw: Any = None

    def to_dict(self) -> dict:
        result = {}
        if self.text:
            result["text"] = self.text
        if self.markdown:
            result["markdown"] = self.markdown
        if self.html:
            result["html"] = self.html
        if self.raw is not None:
            result["raw"] = self.raw
        return result


@dataclass
class Block:
    """Block representation - the basic unit of document content"""
    block_id: str
    type: str
    bbox: BBox
    content: Content = field(default_factory=Content)
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    table_data: Optional[Dict[str, Any]] = None
    figure_data: Optional[Dict[str, Any]] = None
    relationships: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {
            "block_id": self.block_id,
            "type": self.type,
            "bbox": self.bbox.to_dict(),
            "content": self.content.to_dict(),
            "confidence": self.confidence
        }
        if self.properties:
            result["properties"] = self.properties
        if self.table_data:
            result["table_data"] = self.table_data
        if self.figure_data:
            result["figure_data"] = self.figure_data
        if self.relationships:
            result["relationships"] = self.relationships
        return result


@dataclass
class Page:
    """Page representation"""
    page_num: int
    width: float = 0
    height: float = 0
    rotation: int = 0
    blocks: List[Block] = field(default_factory=list)
    page_content: str = ""
    image_data: Optional[str] = None

    def to_dict(self) -> dict:
        result = {
            "page_num": self.page_num,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
            "blocks": [b.to_dict() for b in self.blocks],
            "page_content": self.page_content
        }
        if self.image_data:
            result["image_data"] = self.image_data
        return result


@dataclass
class DocumentMetadata:
    """Document metadata"""
    file_name: str = ""
    file_path: str = ""
    file_size: int = 0
    file_type: str = ""
    file_extension: str = ""
    mime_type: str = ""
    created_at: Optional[str] = None
    processed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None and v != {} and v != ""}


@dataclass
class DocumentInfo:
    """Document information"""
    total_pages: int = 0
    language: str = ""
    title: str = ""
    author: str = ""
    subject: str = ""
    keywords: List[str] = field(default_factory=list)
    creator: str = ""
    producer: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}


@dataclass
class Document:
    """
    Uniform Document representation in JSON format

    This is the standard output format for all document parsing.
    All document types (PDF, Word, Excel, PPT, Markdown, Images)
    are converted to this uniform JSON structure.
    """
    version: str = "1.0"
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    document_info: DocumentInfo = field(default_factory=DocumentInfo)
    pages: List[Page] = field(default_factory=list)
    full_text: str = ""
    full_markdown: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "version": self.version,
            "metadata": self.metadata.to_dict(),
            "document_info": self.document_info.to_dict(),
            "pages": [p.to_dict() for p in self.pages],
            "full_text": self.full_text,
            "full_markdown": self.full_markdown
        }

    def to_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii)

    def save(self, file_path: Union[str, Path]) -> None:
        """Save to JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

    @classmethod
    def from_json(cls, json_str: str) -> 'Document':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls._from_dict(data)

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'Document':
        """Load from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())

    @classmethod
    def _from_dict(cls, data: dict) -> 'Document':
        """Create from dictionary"""
        doc = cls(version=data.get("version", "1.0"))
        if "metadata" in data:
            doc.metadata = DocumentMetadata(**data["metadata"])
        if "document_info" in data:
            doc.document_info = DocumentInfo(**data["document_info"])
        if "pages" in data:
            doc.pages = [cls._page_from_dict(p) for p in data["pages"]]
        doc.full_text = data.get("full_text", "")
        doc.full_markdown = data.get("full_markdown", "")
        return doc

    @classmethod
    def _page_from_dict(cls, data: dict) -> Page:
        """Create Page from dictionary"""
        page = Page(
            page_num=data.get("page_num", 0),
            width=data.get("width", 0),
            height=data.get("height", 0),
            rotation=data.get("rotation", 0),
            page_content=data.get("page_content", "")
        )
        if "blocks" in data:
            page.blocks = [cls._block_from_dict(b) for b in data["blocks"]]
        return page

    @classmethod
    def _block_from_dict(cls, data: dict) -> Block:
        """Create Block from dictionary"""
        bbox = BBox(**data.get("bbox", {}))
        content = Content(
            text=data.get("content", {}).get("text", ""),
            markdown=data.get("content", {}).get("markdown", ""),
            html=data.get("content", {}).get("html", "")
        )
        return Block(
            block_id=data.get("block_id", ""),
            type=data.get("type", "unknown"),
            bbox=bbox,
            content=content,
            confidence=data.get("confidence", 1.0),
            properties=data.get("properties", {}),
            table_data=data.get("table_data"),
            figure_data=data.get("figure_data"),
            relationships=data.get("relationships", {})
        )
