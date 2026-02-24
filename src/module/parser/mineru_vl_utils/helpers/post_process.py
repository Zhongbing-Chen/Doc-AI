"""Post-processing - COPY from MinerUClientHelper, keep equations as plain text"""
import asyncio
from concurrent.futures import Executor

from ..structs import ContentBlock
from ..table_parser import otsl2html

PARATEXT_TYPES = {"header", "footer", "page_number", "aside_text", "page_footnote", "unknown"}


def post_process(
    blocks: list[ContentBlock],
    handle_equation_block: bool = False,  # Keep param but don't process
    abandon_list: bool = False,
    abandon_paratext: bool = False,
    debug: bool = False,
) -> list[ContentBlock]:
    """
    COPY from original post_process, keep equations as plain text

    Changes from original:
    - Keep equation blocks as plain text (no special processing)
    - Remove equation processing code (7 different equation fixers)
    - Keep table processing (COPY AS IS)
    - Keep filtering logic (COPY AS IS)
    """
    for block in blocks:
        # Table processing (COPY AS IS)
        if block.type == "table" and block.content:
            try:
                block.content = otsl2html.convert_otsl_to_html(block.content)
            except Exception as e:
                print("Warning: Failed to convert OTSL to HTML: ", e)
                print("Content: ", block.content)

        # Equation: Keep as plain text, no processing
        # if block.type == "equation" or block.type == "equation_block":
        #     Keep block.content as-is (plain text from VLM)
        #     No LaTeX bracket wrapping
        #     No equation fixing
        #     Just pass through

    # Removed: all equation processing (7 different equation fixers)

    # Removed: equation_block merging logic

    # Removed: add equation brackets (\\[ ... \\])

    # Filter unwanted blocks (COPY AS IS, but keep equations)
    out_blocks: list[ContentBlock] = []
    for block in blocks:
        # Keep equation blocks (don't drop them)
        if block.type == "equation_block":
            # Convert equation_block to equation for consistency
            block["type"] = "equation"
        if abandon_list and block.type == "list":
            continue
        if abandon_paratext and block.type in PARATEXT_TYPES:
            continue
        out_blocks.append(block)

    return out_blocks


class PostProcessHelper:
    """Helper class for batch/async post-processing (COPY from MinerUClientHelper)"""

    def __init__(
        self,
        abandon_list: bool = False,
        abandon_paratext: bool = False,
        debug: bool = False,
    ):
        self.abandon_list = abandon_list
        self.abandon_paratext = abandon_paratext
        self.debug = debug

    def post_process(self, blocks: list[ContentBlock]) -> list[ContentBlock]:
        """Original method (COPY AS IS, equations as plain text)"""
        try:
            return post_process(
                blocks,
                handle_equation_block=False,  # Don't process equations, keep as-is
                abandon_list=self.abandon_list,
                abandon_paratext=self.abandon_paratext,
                debug=self.debug,
            )
        except Exception as e:
            print(f"Warning: post-processing failed with error: {e}")
            return blocks

    # Batch method (COPY AS IS)
    def batch_post_process(
        self,
        executor: Executor | None,
        blocks_list: list[list[ContentBlock]],
    ) -> list[list[ContentBlock]]:
        """Original method (COPY AS IS)"""
        if executor is None:
            return [self.post_process(blocks) for blocks in blocks_list]
        return list(executor.map(self.post_process, blocks_list))

    # Async method (COPY AS IS)
    async def aio_post_process(
        self,
        executor: Executor | None,
        blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        """Original method (COPY AS IS)"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.post_process, blocks)
