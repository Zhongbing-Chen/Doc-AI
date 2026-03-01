"""
Export functions for document parser

Provides markdown and JSON export functionality.
"""
import json
from typing import List


def convert_to_markdown(pages: List, output_file: str = "list_output.md") -> str:
    """
    Convert pages to markdown format

    Args:
        pages: List of Page objects
        output_file: Output file path

    Returns:
        Markdown content string
    """
    md_content = []

    for page in pages:
        # Add page header
        md_content.append(f"--- Page {page.page_num + 1} ---\n")

        # Add block content
        for block in page.blocks:
            if block.content:
                md_content.append(block.content)
                md_content.append("\n\n")

    # Join all content
    full_content = ''.join(md_content)

    # Write to file
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_content)

    return full_content


def merge(pages: List, output_file: str = "json_output.json") -> str:
    """
    Merge pages to JSON format

    Args:
        pages: List of Page objects
        output_file: Output file path

    Returns:
        JSON content string
    """
    result = []

    for page in pages:
        page_data = {
            "page_num": page.page_num,
            "blocks": []
        }

        for block in page.blocks:
            block_data = {
                "id": block.block_id,
                "label": block.label,
                "bbox": [block.x_1, block.y_1, block.x_2, block.y_2],
                "content": block.content,
            }

            # Add table structure if available
            if block.table_structure:
                block_data["table_structure"] = [
                    {
                        "bbox": cell.bbox,
                        "content": cell.content,
                        "row_nums": cell.row_nums,
                        "column_nums": cell.column_nums,
                        "column_header": cell.column_header,
                        "projected_row_header": cell.projected_row_header,
                    }
                    for cell in block.table_structure
                ]

            page_data["blocks"].append(block_data)

        result.append(page_data)

    # Convert to JSON
    json_content = json.dumps(result, indent=2, ensure_ascii=False)

    # Write to file
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_content)

    return json_content
