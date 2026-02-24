"""
Direct VLM Client Utility for MinerU

This utility calls the MinerU VLM server directly (OpenAI-compatible API)
and parses the raw output to get ALL elements with bbox coordinates.

Unlike mineru-vl-utils which only provides high-level blocks,
this gives us access to ALL detected elements including individual text lines.
"""

import os
import re
import io
from typing import List, Dict, Optional, Tuple
from PIL import Image
import base64
import json

# Disable proxy for localhost
os.environ['NO_PROXY'] = 'localhost'

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests not installed. pip install requests")


class VLMElement:
    """Represents a single element detected by VLM with bbox and content."""

    def __init__(self, bbox: List[float], element_type: str, content: str = ""):
        """
        Initialize VLM element.

        Args:
            bbox: Normalized bbox [x1, y1, x2, y2] where coordinates are 0-1
            element_type: Type of element (text, title, table, etc.)
            content: Text content
        """
        self.bbox = bbox
        self.type = element_type
        self.content = content

    def to_pixel_bbox(self, img_width: int, img_height: int) -> List[float]:
        """Convert normalized bbox to pixel coordinates."""
        x1, y1, x2, y2 = self.bbox
        return [
            x1 * img_width,
            y1 * img_height,
            x2 * img_width,
            y2 * img_height
        ]

    def __repr__(self):
        content_preview = self.content[:30] if self.content else ""
        return f"VLMElement(type={self.type}, bbox={self.bbox}, content='{content_preview}...')"


class DirectVLMClient:
    """
    Direct client for MinerU VLM server (OpenAI-compatible API).

    Parses raw VLM output to extract ALL elements with bbox coordinates.
    """

    # Regex pattern to parse VLM output
    # Format: <|box_start|>x1 y1 x2 y2<|box_end|><|ref_start|>type<|ref_end|>content
    VLM_OUTPUT_RE = re.compile(
        r'<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|>'
        r'<\|ref_start\|>(\w+?)<\|ref_end\|>(.*)$'
    )

    def __init__(self, server_url: str = "http://localhost:8000"):
        """
        Initialize Direct VLM Client.

        Args:
            server_url: URL of the VLM server
        """
        self.server_url = server_url
        self.api_url = f"{server_url}/v1/chat/completions"

        if not HAS_REQUESTS:
            raise ImportError("requests library is required. Install with: pip install requests")

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _create_payload(self, image_base64: str, prompt: str = "") -> Dict:
        """Create API payload for VLM server."""
        return {
            "model": "mineru",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 4096
        }

    def _parse_bbox(self, x1: str, y1: str, x2: str, y2: str) -> Optional[List[float]]:
        """
        Parse bbox coordinates from VLM output (0-1000 range).

        Args:
            x1, y1, x2, y2: Bbox coordinates as strings

        Returns:
            Normalized bbox [x1, y1, x2, y2] in range 0-1, or None if invalid
        """
        try:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            # Validate bbox
            if any(coord < 0 or coord > 1000 for coord in (x1, y1, x2, y2)):
                return None

            # Normalize to 0-1
            bbox = [coord / 1000.0 for coord in (x1, y1, x2, y2)]

            # Ensure x1 < x2 and y1 < y2
            if bbox[0] > bbox[2]:
                bbox[0], bbox[2] = bbox[2], bbox[0]
            if bbox[1] > bbox[3]:
                bbox[1], bbox[3] = bbox[3], bbox[1]

            # Check for valid size
            if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
                return None

            return bbox
        except ValueError:
            return None

    def _parse_vlm_output(self, output: str) -> List[VLMElement]:
        """
        Parse raw VLM output to extract all elements with bbox.

        Args:
            output: Raw text output from VLM

        Returns:
            List of VLMElement objects
        """
        elements = []

        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue

            match = self.VLM_OUTPUT_RE.match(line)
            if not match:
                # Skip lines that don't match the expected format
                continue

            x1, y1, x2, y2, element_type, content = match.groups()

            # Parse bbox
            bbox = self._parse_bbox(x1, y1, x2, y2)
            if bbox is None:
                print(f"Warning: Invalid bbox in line: {line}")
                continue

            # Create element
            element = VLMElement(
                bbox=bbox,
                element_type=element_type.lower(),
                content=content
            )
            elements.append(element)

        return elements

    def extract_elements(self, image: Image.Image, prompt: str = "") -> List[VLMElement]:
        """
        Extract all elements with bbox from image using VLM.

        Args:
            image: PIL Image
            prompt: Optional text prompt

        Returns:
            List of VLMElement objects with bbox and content
        """
        # Encode image
        image_base64 = self._encode_image(image)

        # Create payload
        payload = self._create_payload(image_base64, prompt)

        # Call VLM server
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error calling VLM server: {e}")
            return []

        # Parse response
        try:
            result = response.json()
            output_text = result['choices'][0]['message']['content']
        except (KeyError, IndexError) as e:
            print(f"Error parsing VLM response: {e}")
            return []

        # Parse VLM output to get elements
        elements = self._parse_vlm_output(output_text)

        return elements

    def extract_table_cells(self, image: Image.Image) -> Tuple[List[VLMElement], List[VLMElement]]:
        """
        Extract tables and table cells from image.

        This method extracts all elements, then separates tables from other elements.
        For tables, you can later parse the HTML content to get cell structure.

        Args:
            image: PIL Image

        Returns:
            Tuple of (all elements, table elements only)
        """
        # Extract all elements
        all_elements = self.extract_elements(image)

        # Filter tables
        table_elements = [e for e in all_elements if e.type == 'table']
        other_elements = [e for e in all_elements if e.type != 'table']

        return all_elements, table_elements


# Test function
def test_direct_vlm_client():
    """Test the Direct VLM Client."""
    import fitz

    print("=" * 80)
    print("Testing Direct VLM Client")
    print("=" * 80)

    # Load test image (page 3 from BOC PDF)
    pdf_path = "/home/zhongbing/Projects/MLE/DocAI/pdf/boc_first_10_pages.pdf"
    doc = fitz.open(pdf_path)
    page = doc[2]  # Page 3 with honors table
    pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
    img_bytes = pix.tobytes("ppm")
    image = Image.open(io.BytesIO(img_bytes))
    doc.close()

    print(f"\n1. Loaded image: {image.size[0]} x {image.size[1]}")

    # Create client
    client = DirectVLMClient(server_url="http://localhost:8000")

    # Extract elements
    print(f"\n2. Extracting elements from VLM...")
    elements = client.extract_elements(image)

    print(f"   ✓ Found {len(elements)} elements")

    # Group by type
    by_type = {}
    for elem in elements:
        if elem.type not in by_type:
            by_type[elem.type] = []
        by_type[elem.type].append(elem)

    print(f"\n3. Elements by type:")
    for elem_type, elems in sorted(by_type.items()):
        print(f"   {elem_type}: {len(elems)}")

    # Show first few elements
    print(f"\n4. First 5 elements:")
    for i, elem in enumerate(elements[:5]):
        pixel_bbox = elem.to_pixel_bbox(image.size[0], image.size[1])
        print(f"   Element {i+1}:")
        print(f"     Type: {elem.type}")
        print(f"     BBox (normalized): {elem.bbox}")
        print(f"     BBox (pixels): [{pixel_bbox[0]:.1f}, {pixel_bbox[1]:.1f}, {pixel_bbox[2]:.1f}, {pixel_bbox[3]:.1f}]")
        content_preview = elem.content[:50] if elem.content else ""
        print(f"     Content: {content_preview}...")

    print(f"\n{'=' * 80}")
    print("✓ Test complete!")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    test_direct_vlm_client()
