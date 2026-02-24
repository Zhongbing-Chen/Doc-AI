"""
Coordinate Mapping Module for MinerU
Provides bidirectional mapping between original document coordinates and parsed markdown offsets.
"""

import json
import uuid
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class BoundingBox:
    """Bounding box in PDF coordinates [x0, y0, x1, y1]"""
    x0: float
    y0: float
    x1: float
    y1: float

    def to_list(self) -> List[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    @classmethod
    def from_list(cls, bbox: List[float]) -> 'BoundingBox':
        if len(bbox) != 4:
            raise ValueError("Bounding box must have 4 values [x0, y0, x1, y1]")
        return cls(bbox[0], bbox[1], bbox[2], bbox[3])

    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if this bounding box intersects with another"""
        return not (self.x1 < other.x0 or self.x0 > other.x1 or
                   self.y1 < other.y0 or self.y0 > other.y1)

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point (x, y) is inside this bounding box"""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box"""
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def area(self) -> float:
        """Calculate area of bounding box"""
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def intersection_area(self, other: 'BoundingBox') -> float:
        """Calculate intersection area with another bounding box"""
        x_overlap = max(0, min(self.x1, other.x1) - max(self.x0, other.x0))
        y_overlap = max(0, min(self.y1, other.y1) - max(self.y0, other.y0))
        return x_overlap * y_overlap


@dataclass
class MarkdownRange:
    """Range in markdown content"""
    offset: int
    length: int

    def to_tuple(self) -> Tuple[int, int]:
        return (self.offset, self.length)

    @classmethod
    def from_tuple(cls, range_tuple: Tuple[int, int]) -> 'MarkdownRange':
        return cls(range_tuple[0], range_tuple[1])

    def contains(self, offset: int) -> bool:
        """Check if offset is within this range"""
        return self.offset <= offset < self.offset + self.length

    def end(self) -> int:
        """Get end offset (exclusive)"""
        return self.offset + self.length


@dataclass
class CoordinateMapping:
    """
    Maps coordinates between original document and parsed markdown.

    Attributes:
        page_num: Page number in the original document
        bbox: Bounding box in PDF coordinates
        markdown_range: Range in markdown content
        element_type: Type of element (text, table, image, formula)
        element_id: Unique identifier for this element
        confidence: Confidence score (0-1) if available
        metadata: Additional metadata
    """
    page_num: int
    bbox: BoundingBox
    markdown_range: MarkdownRange
    element_type: str
    element_id: str
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'page_num': self.page_num,
            'bbox': self.bbox.to_list(),
            'markdown_offset': self.markdown_range.offset,
            'markdown_length': self.markdown_range.length,
            'element_type': self.element_type,
            'element_id': self.element_id,
            'confidence': self.confidence,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoordinateMapping':
        """Create from dictionary"""
        return cls(
            page_num=data['page_num'],
            bbox=BoundingBox.from_list(data['bbox']),
            markdown_range=MarkdownRange(
                data['markdown_offset'],
                data['markdown_length']
            ),
            element_type=data['element_type'],
            element_id=data['element_id'],
            confidence=data.get('confidence'),
            metadata=data.get('metadata')
        )


class CoordinateMapper:
    """
    Provides bidirectional coordinate mapping between original document and parsed markdown.

    This class maintains a list of coordinate mappings and provides methods to query
    mappings in both directions (original document → markdown and markdown → original document).
    """

    def __init__(self):
        self.mappings: List[CoordinateMapping] = []
        self.page_count: int = 0
        self.markdown_length: int = 0

    def add_mapping(self, mapping: CoordinateMapping) -> None:
        """Add a coordinate mapping"""
        self.mappings.append(mapping)

    def add_mappings(self, mappings: List[CoordinateMapping]) -> None:
        """Add multiple coordinate mappings"""
        self.mappings.extend(mappings)

    def clear(self) -> None:
        """Clear all mappings"""
        self.mappings.clear()
        self.page_count = 0
        self.markdown_length = 0

    def markdown_to_original(
        self,
        markdown_offset: int,
        page_num: Optional[int] = None
    ) -> Optional[CoordinateMapping]:
        """
        Find original document coordinates for a markdown offset.

        Args:
            markdown_offset: Character offset in markdown content
            page_num: Optional page number to filter results

        Returns:
            CoordinateMapping if found, None otherwise
        """
        candidates = []

        for mapping in self.mappings:
            if page_num is not None and mapping.page_num != page_num:
                continue

            if mapping.markdown_range.contains(markdown_offset):
                candidates.append(mapping)

        if not candidates:
            return None

        # Return the mapping with the smallest area (most precise)
        candidates.sort(key=lambda m: m.bbox.area())
        return candidates[0]

    def original_to_markdown(
        self,
        page_num: int,
        x: float,
        y: float
    ) -> Optional[CoordinateMapping]:
        """
        Find markdown offset for original document coordinates.

        Args:
            page_num: Page number in the original document
            x: X coordinate in PDF space
            y: Y coordinate in PDF space

        Returns:
            CoordinateMapping if found, None otherwise
        """
        candidates = []
        point = (x, y)

        for mapping in self.mappings:
            if mapping.page_num != page_num:
                continue

            if mapping.bbox.contains_point(x, y):
                candidates.append(mapping)

        if not candidates:
            return None

        # Return the mapping with the smallest area (most precise)
        candidates.sort(key=lambda m: m.bbox.area())
        return candidates[0]

    def original_bbox_to_markdown(
        self,
        page_num: int,
        bbox: BoundingBox
    ) -> List[CoordinateMapping]:
        """
        Find all markdown offsets that intersect with a bounding box.

        Args:
            page_num: Page number in the original document
            bbox: Bounding box to search

        Returns:
            List of intersecting CoordinateMappings, sorted by intersection area
        """
        candidates = []

        for mapping in self.mappings:
            if mapping.page_num != page_num:
                continue

            if mapping.bbox.intersects(bbox):
                intersection_area = mapping.bbox.intersection_area(bbox)
                candidates.append((intersection_area, mapping))

        # Sort by intersection area (descending)
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [mapping for _, mapping in candidates]

    def get_elements_by_type(
        self,
        element_type: str,
        page_num: Optional[int] = None
    ) -> List[CoordinateMapping]:
        """
        Get all mappings of a specific element type.

        Args:
            element_type: Type of element (text, table, image, formula)
            page_num: Optional page number to filter results

        Returns:
            List of CoordinateMappings of the specified type
        """
        mappings = self.mappings

        if page_num is not None:
            mappings = [m for m in mappings if m.page_num == page_num]

        return [m for m in mappings if m.element_type == element_type]

    def get_page_elements(self, page_num: int) -> List[CoordinateMapping]:
        """
        Get all elements on a specific page.

        Args:
            page_num: Page number

        Returns:
            List of CoordinateMappings on the page
        """
        return [m for m in self.mappings if m.page_num == page_num]

    def to_json(self) -> str:
        """Export mappings to JSON string"""
        data = {
            'mappings': [m.to_dict() for m in self.mappings],
            'page_count': self.page_count,
            'markdown_length': self.markdown_length
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'CoordinateMapper':
        """Import mappings from JSON string"""
        data = json.loads(json_str)
        mapper = cls()
        mapper.page_count = data.get('page_count', 0)
        mapper.markdown_length = data.get('markdown_length', 0)

        for mapping_data in data.get('mappings', []):
            mapper.add_mapping(CoordinateMapping.from_dict(mapping_data))

        return mapper

    @classmethod
    def from_mineru_json(cls, json_content: Dict[str, Any]) -> 'CoordinateMapper':
        """
        Create CoordinateMapper from MinerU JSON output.

        Args:
            json_content: Parsed JSON content from MinerU

        Returns:
            CoordinateMapper with extracted mappings
        """
        mapper = cls()

        # Process each page
        for page in json_content.get('pages', []):
            page_num = page.get('page_no', 0)

            # Process layout items
            for layout in page.get('layouts', []):
                bbox_data = layout.get('bbox', [])
                if len(bbox_data) == 4:
                    mapping = CoordinateMapping(
                        page_num=page_num,
                        bbox=BoundingBox.from_list(bbox_data),
                        markdown_range=MarkdownRange(
                            layout.get('markdown_offset', 0),
                            layout.get('markdown_length', 0)
                        ),
                        element_type=layout.get('type', 'text'),
                        element_id=layout.get('id', str(uuid.uuid4())),
                        confidence=layout.get('confidence'),
                        metadata={'layout': layout}
                    )
                    mapper.add_mapping(mapping)

            # Process tables
            for table in page.get('tables', []):
                bbox_data = table.get('bbox', [])
                if len(bbox_data) == 4:
                    mapping = CoordinateMapping(
                        page_num=page_num,
                        bbox=BoundingBox.from_list(bbox_data),
                        markdown_range=MarkdownRange(
                            table.get('markdown_offset', 0),
                            table.get('markdown_length', 0)
                        ),
                        element_type='table',
                        element_id=table.get('id', str(uuid.uuid4())),
                        confidence=table.get('confidence'),
                        metadata={'table': table}
                    )
                    mapper.add_mapping(mapping)

            # Process images
            for image in page.get('images', []):
                bbox_data = image.get('bbox', [])
                if len(bbox_data) == 4:
                    mapping = CoordinateMapping(
                        page_num=page_num,
                        bbox=BoundingBox.from_list(bbox_data),
                        markdown_range=MarkdownRange(
                            image.get('markdown_offset', 0),
                            image.get('markdown_length', 0)
                        ),
                        element_type='image',
                        element_id=image.get('id', str(uuid.uuid4())),
                        confidence=image.get('confidence'),
                        metadata={'image': image}
                    )
                    mapper.add_mapping(mapping)

        return mapper


if __name__ == "__main__":
    # Example usage
    mapper = CoordinateMapper()

    # Add some example mappings
    mapping1 = CoordinateMapping(
        page_num=1,
        bbox=BoundingBox(100, 200, 400, 250),
        markdown_range=MarkdownRange(0, 100),
        element_type='text',
        element_id='txt1'
    )
    mapper.add_mapping(mapping1)

    # Query markdown → original
    result = mapper.markdown_to_original(50)
    if result:
        print(f"Found at page {result.page_num}, bbox: {result.bbox.to_list()}")

    # Query original → markdown
    result = mapper.original_to_markdown(1, 250, 225)
    if result:
        print(f"Found at markdown offset {result.markdown_range.offset}")

    # Export to JSON
    json_str = mapper.to_json()
    print(json_str)

    # Import from JSON
    mapper2 = CoordinateMapper.from_json(json_str)
    print(f"Imported {len(mapper2.mappings)} mappings")
