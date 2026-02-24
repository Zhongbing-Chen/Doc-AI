"""
Table utility functions for converting between table formats.

Provides HTML to Markdown conversion with proper handling of merged cells.
"""


import re
from typing import List, Dict, Optional


def html_table_to_markdown(html_table: str) -> str:
    """
    Convert HTML table to Markdown format with proper handling of merged cells.

    Merged cells (rowspan/colspan) are replicated across rows/columns.
    Produces clean Markdown tables compatible with MD readers.

    Args:
        html_table: HTML table string

    Returns:
        Markdown table string

    Example:
        >>> html = '<table><tr><td>A</td><td>B</td></tr></table>'
        >>> md = html_table_to_markdown(html)
        >>> print(md)
        | A | B |
        |---|---|
    """
    if not html_table or not html_table.strip():
        return ""

    # Clean up HTML: remove newlines within tags
    html_table = re.sub(r'\n+', ' ', html_table)

    # Parse HTML table
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html_table, re.DOTALL)
    if not rows:
        return ""

    # Track actual column count and merged cells
    table_data = []
    max_cols = 0

    for row_idx, row in enumerate(rows):
        # Extract cells with their tags
        cell_matches = re.findall(r'<(t[hd])([^>]*)>(.*?)</\1>', row, re.DOTALL)

        cells_in_row = []
        col_offset = 0

        for cell_type, attrs, content in cell_matches:
            # Parse rowspan and colspan
            rowspan_match = re.search(r'rowspan=["\'](\d+)["\']', attrs)
            colspan_match = re.search(r'colspan=["\'](\d+)["\']', attrs)

            rowspan = int(rowspan_match.group(1)) if rowspan_match else 1
            colspan = int(colspan_match.group(1)) if colspan_match else 1

            # Clean cell content: remove HTML tags but preserve text
            content_clean = re.sub(r'<[^>]+>', '', content)
            content_clean = content_clean.strip()
            # Replace multiple spaces with single space
            content_clean = re.sub(r'\s+', ' ', content_clean)

            cells_in_row.append({
                'content': content_clean,
                'colspan': colspan,
                'rowspan': rowspan,
                'row': row_idx,
                'col_start': col_offset,
                'col_end': col_offset + colspan
            })

            col_offset += colspan

        max_cols = max(max_cols, col_offset)
        table_data.append(cells_in_row)

    # Build markdown rows with proper cell expansion
    markdown_rows = []

    for row_idx, cells_in_row in enumerate(table_data):
        expanded_cells = []

        for cell in cells_in_row:
            # Expand cells with colspan
            for _ in range(cell['colspan']):
                expanded_cells.append(cell['content'])

        # Pad row to max_cols if needed
        while len(expanded_cells) < max_cols:
            expanded_cells.append('')

        # Create markdown row
        md_row = '| ' + ' | '.join(expanded_cells) + ' |'
        markdown_rows.append(md_row)

    # Add separator after first row (header row)
    if len(markdown_rows) > 1:
        separator = '|' + '|'.join(['---' for _ in range(max_cols)]) + '|'
        markdown_rows.insert(1, separator)

    # Add blank lines before and after table for MD reader compatibility
    result = '\n'.join(markdown_rows)
    return '\n' + result + '\n'


def parse_html_table_structure(html_table: str, table_bbox: List[float], img_width: int, img_height: int,
                               ocr_boxes: List[dict] = None) -> List[dict]:
    """
    Parse HTML table to extract cell-level information with bbox coordinates.

    Can use OCR text boxes to refine cell boundaries for more accurate positioning.

    Args:
        html_table: HTML table string
        table_bbox: Bounding box of the entire table [x_1, y_1, x_2, y_2]
        img_width: Image width
        img_height: Image height
        ocr_boxes: Optional list of OCR text boxes with 'text' and 'bbox' keys
                  If provided, will be used to refine cell boundaries

    Returns:
        List of cell dictionaries with content, row, col, and bbox (estimated or OCR-refined)
    """
    if not html_table or not html_table.strip():
        return []

    # Clean up HTML
    html_table = re.sub(r'\n+', ' ', html_table)

    # Parse HTML table
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html_table, re.DOTALL)
    if not rows:
        return []

    # Track table structure
    table_data = []
    max_cols = 0

    for row_idx, row in enumerate(rows):
        cell_matches = re.findall(r'<(t[hd])([^>]*)>(.*?)</\1>', row, re.DOTALL)

        cells_in_row = []
        col_offset = 0

        for cell_type, attrs, content in cell_matches:
            rowspan_match = re.search(r'rowspan=["\'](\d+)["\']', attrs)
            colspan_match = re.search(r'colspan=["\'](\d+)["\']', attrs)

            rowspan = int(rowspan_match.group(1)) if rowspan_match else 1
            colspan = int(colspan_match.group(1)) if colspan_match else 1

            content_clean = re.sub(r'<[^>]+>', '', content)
            content_clean = content_clean.strip()
            content_clean = re.sub(r'\s+', ' ', content_clean)

            cells_in_row.append({
                'content': content_clean,
                'colspan': colspan,
                'rowspan': rowspan,
                'row': row_idx,
                'col_start': col_offset,
                'col_end': col_offset + colspan
            })

            col_offset += colspan

        max_cols = max(max_cols, col_offset)
        table_data.append(cells_in_row)

    # Calculate cell bboxes
    table_x1, table_y1, table_x2, table_y2 = table_bbox
    table_width = table_x2 - table_x1
    table_height = table_y2 - table_y1

    num_rows = len(table_data)
    if num_rows == 0 or max_cols == 0:
        return []

    cell_height = table_height / num_rows
    cell_width = table_width / max_cols

    # Generate cell list with estimated bboxes
    cells = []

    for row_idx, cells_in_row in enumerate(table_data):
        for cell in cells_in_row:
            # Calculate estimated bbox for this cell
            cell_x1 = table_x1 + (cell['col_start'] * cell_width)
            cell_y1 = table_y1 + (row_idx * cell_height)
            cell_x2 = table_x1 + (cell['col_end'] * cell_width)
            cell_y2 = cell_y1 + (cell['rowspan'] * cell_height)

            estimated_bbox = [cell_x1, cell_y1, cell_x2, cell_y2]

            # If OCR boxes provided, try to refine the bbox
            if ocr_boxes:
                refined_bbox = refine_bbox_with_ocr(estimated_bbox, cell['content'], ocr_boxes)
                final_bbox = refined_bbox if refined_bbox else estimated_bbox
            else:
                final_bbox = estimated_bbox

            cells.append({
                'content': cell['content'],
                'row': cell['row'],
                'col_start': cell['col_start'],
                'col_end': cell['col_end'],
                'colspan': cell['colspan'],
                'rowspan': cell['rowspan'],
                'bbox': final_bbox,
                'bbox_source': 'ocr_refined' if (ocr_boxes and final_bbox != estimated_bbox) else 'estimated'
            })

    return cells


def refine_bbox_with_ocr(cell_bbox: List[float], cell_content: str, ocr_boxes: List[dict]) -> Optional[List[float]]:
    """
    Refine cell bbox using OCR text boxes that match the cell content.

    Finds OCR boxes within the estimated cell bbox that match the cell content,
    then expands the cell bbox to fit the actual text positions.

    Args:
        cell_bbox: Estimated cell bbox [x_1, y_1, x_2, y_2]
        cell_content: Text content of the cell
        ocr_boxes: List of OCR boxes with 'text' and 'bbox' [x_1, y_1, x_2, y_2] keys

    Returns:
        Refined bbox [x_1, y_1, x_2, y_2] or None if no matching OCR boxes found
    """
    if not ocr_boxes or not cell_content:
        return None

    cell_x1, cell_y1, cell_x2, cell_y2 = cell_bbox

    # Find OCR boxes that are within the estimated cell bbox
    matching_boxes = []
    for ocr_box in ocr_boxes:
        ocr_x1, ocr_y1, ocr_x2, ocr_y2 = ocr_box['bbox']

        # Check if OCR box is within cell bbox (with some tolerance)
        if (ocr_x1 >= cell_x1 - 10 and ocr_x2 <= cell_x2 + 10 and
            ocr_y1 >= cell_y1 - 10 and ocr_y2 <= cell_y2 + 10):

            # Check if OCR text matches cell content (partial match is OK)
            ocr_text = ocr_box.get('text', '')
            if ocr_text and (ocr_text in cell_content or cell_content in ocr_text):
                matching_boxes.append(ocr_box)

    if not matching_boxes:
        return None

    # Expand cell bbox to fit all matching OCR boxes
    refined_x1 = min(box['bbox'][0] for box in matching_boxes)
    refined_y1 = min(box['bbox'][1] for box in matching_boxes)
    refined_x2 = max(box['bbox'][2] for box in matching_boxes)
    refined_y2 = max(box['bbox'][3] for box in matching_boxes)

    # Add small padding
    padding = 5
    return [
        max(cell_x1, refined_x1 - padding),
        max(cell_y1, refined_y1 - padding),
        min(cell_x2, refined_x2 + padding),
        min(cell_y2, refined_y2 + padding)
    ]


if __name__ == '__main__':
    # Test HTML to Markdown conversion
    test_html = """
    <table>
        <tr>
            <th colspan="2">Header 1</th>
            <th>Header 2</th>
        </tr>
        <tr>
            <td>Cell 1</td>
            <td>Cell 2</td>
            <td>Cell 3</td>
        </tr>
    </table>
    """

    markdown = html_table_to_markdown(test_html)
    print("HTML to Markdown conversion:")
    print(markdown)
