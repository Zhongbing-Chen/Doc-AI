"""
Get raw VLM output to see ALL elements with bbox.

This bypasses the post-processing in mineru-vl-utils to see
the raw VLM output with all detected elements.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz
import io
from PIL import Image

# Set NO_PROXY
os.environ['NO_PROXY'] = 'localhost'

from mineru_vl_utils import MinerUClient


def get_raw_vlm_output():
    """Get raw VLM output to see all elements."""

    print("=" * 80)
    print("Getting Raw VLM Output - All Elements with BBox")
    print("=" * 80)

    # Load image (page 3 from BOC PDF - has table)
    pdf_path = "/home/zhongbing/Projects/MLE/DocAI/pdf/boc_first_10_pages.pdf"
    doc = fitz.open(pdf_path)
    page = doc[2]  # Page 3
    pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
    img_bytes = pix.tobytes("ppm")
    image = Image.open(io.BytesIO(img_bytes))
    doc.close()

    print(f"\n1. Loaded image: {image.size[0]} x {image.size[1]}")

    # Get the internal HTTP client
    client = MinerUClient(backend="http-client", server_url="http://localhost:8000")
    http_client = client.client

    print(f"\n2. HTTP Client server URL: {http_client.server_url}")

    # Use the helper to prepare the image for layout detection
    print(f"\n3. Preparing image for layout detection...")
    prepared_image = client.helper.prepare_for_layout(image)
    print(f"   ✓ Prepared image type: {type(prepared_image)}")

    # Call the VLM client directly with layout detection prompt
    print(f"\n4. Calling VLM server for layout detection...")

    # Get the prompt for layout detection
    layout_prompt = "\nLayout Detection:"

    try:
        # Get sampling params from helper
        sampling_params = client.helper.sampling_params.get("[layout]")
        if sampling_params is None:
            from mineru_vl_utils import MinerUSamplingParams
            sampling_params = MinerUSamplingParams()

        # Make the API call
        output = http_client.predict(
            prepared_image,
            layout_prompt,
            sampling_params,
            priority=None
        )

        print(f"   ✓ Got VLM output ({len(output)} characters)")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        print(f"\n   Please ensure VLM server is running on port 8000")
        return

    # Show raw output
    print(f"\n5. Raw VLM Output:")
    print(f"   {'=' * 76}")
    lines = output.split('\n')
    print(f"   Total lines: {len(lines)}")
    print(f"\n   First 20 lines:")
    for i, line in enumerate(lines[:20], 1):
        print(f"   {i:2d}. {line}")

    # Parse all elements
    print(f"\n6. Parsing elements from VLM output...")

    # Use the helper to parse layout output
    try:
        blocks = client.helper.parse_layout_output(output)
        print(f"   ✓ Parsed {len(blocks)} blocks")
    except Exception as e:
        print(f"   ✗ Parse error: {e}")
        return

    # Group by type
    by_type = {}
    for block in blocks:
        if block.type not in by_type:
            by_type[block.type] = []
        by_type[block.type].append(block)

    print(f"\n7. Blocks by type:")
    for block_type, blocks_list in sorted(by_type.items()):
        print(f"   {block_type}: {len(blocks_list)}")

    # Show all blocks with bbox
    print(f"\n8. All blocks with bbox:")
    print(f"   {'=' * 76}")

    for i, block in enumerate(blocks, 1):
        print(f"\n   Block {i}:")
        print(f"     Type: {block.type}")
        print(f"     BBox (normalized): {block.bbox}")

        # Convert to pixel coordinates manually
        bbox = block.bbox
        pixel_bbox = [bbox[0] * image.size[0], bbox[1] * image.size[1],
                      bbox[2] * image.size[0], bbox[3] * image.size[1]]
        print(f"     BBox (pixels): [{pixel_bbox[0]:.1f}, {pixel_bbox[1]:.1f}, {pixel_bbox[2]:.1f}, {pixel_bbox[3]:.1f}]")

        if block.angle is not None:
            print(f"     Angle: {block.angle}")

        if block.content:
            content_preview = block.content[:100].replace('\n', ' ')
            print(f"     Content: {content_preview}...")

    print(f"\n{'=' * 80}")
    print("✓ Raw VLM output retrieved!")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    get_raw_vlm_output()
