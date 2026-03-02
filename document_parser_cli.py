"""
Standalone Document Parser Script

Usage:
    python document_parser_cli.py <file_path> [--output <output.md>]

Supported formats:
    - Word: .doc, .docx
    - Excel: .xls, .xlsx
    - PowerPoint: .ppt, .pptx
    - Markdown: .md, .markdown
    - Images: .png, .jpg, .jpeg
"""

import argparse
import sys
from pathlib import Path

from universal_parser import UniversalDocumentParser, parse_document


def main():
    parser = argparse.ArgumentParser(
        description='Parse various document formats to markdown'
    )
    parser.add_argument(
        'file_path',
        type=str,
        nargs='?',
        default=None,
        help='Path to the document file to parse'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path (default: print to stdout)'
    )
    parser.add_argument(
        '-l', '--list-formats',
        action='store_true',
        help='List supported file formats'
    )
    parser.add_argument(
        '--enable-vlm',
        action='store_true',
        help='Enable VLM for image parsing'
    )
    parser.add_argument(
        '--vlm-url',
        type=str,
        default=None,
        help='VLM server URL'
    )

    args = parser.parse_args()

    if args.list_formats:
        print("Supported file formats:")
        formats = UniversalDocumentParser.SUPPORTED_FORMATS
        for ext, file_type in sorted(formats.items()):
            print(f"  {ext}: {file_type}")
        return 0

    if not args.file_path:
        parser.print_help()
        return 1

    file_path = Path(args.file_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    try:
        # Parse the document
        result = parse_document(
            file_path,
            enable_vlm=args.enable_vlm,
            vlm_url=args.vlm_url
        )

        # Output results
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.markdown)
            print(f"Output written to: {output_path}")
        else:
            print(result.markdown)

        # Print metadata to stderr if not outputting to file
        if not args.output:
            print(f"\n--- Metadata ---", file=sys.stderr)
            print(f"File: {result.file_path}", file=sys.stderr)
            print(f"Type: {result.file_type}", file=sys.stderr)
            print(f"Extension: {result.file_ext}", file=sys.stderr)
            if result.title:
                print(f"Title: {result.title}", file=sys.stderr)

        return 0

    except Exception as e:
        print(f"Error parsing document: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
