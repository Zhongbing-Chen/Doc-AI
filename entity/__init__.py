"""Entity module for document parser"""
from .box import Box, OcrBlock, BoxUtil
from .block import Block, TableStructure

__all__ = ['Box', 'OcrBlock', 'BoxUtil', 'Block', 'TableStructure']
