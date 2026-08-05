"""
Shared token-bounded packing helpers, used by both docs_chunker.py (##/###
sections) and chunker.py (discussion answers) so any long piece of source
text is packed into embeddable chunks the same way.

Extracted verbatim from docs_chunker.py -- see that module's docstring for
the packing strategy this implements: fenced ```code``` blocks are never
split; a blank-line-free block that alone exceeds the cap is split by line;
blocks are then greedily packed into chunks that prioritize reaching the
400-token floor over respecting the 600-token ceiling.
"""

from __future__ import annotations

TARGET_MIN_TOKENS = 400
TARGET_MAX_TOKENS = 600
CHARS_PER_TOKEN = 4

_FENCE_PREFIX = "```"


def _estimate_tokens(text: str) -> int:
    """Rough token count, ~4 chars/token (no exact tokenizer for the local
    embedding model)."""
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def _split_into_blocks(lines: list[str]) -> list[tuple[str, bool]]:
    """Split a section's lines into (text, is_code) blocks: each fenced
    ```code``` block is one atomic block, each blank-line-delimited chunk
    of prose (which may itself be a long, blank-line-free list) is another."""
    blocks: list[tuple[str, bool]] = []
    buf: list[str] = []
    in_fence = False

    def flush(is_code: bool = False) -> None:
        if buf and any(line.strip() for line in buf):
            blocks.append(("\n".join(buf).strip("\n"), is_code))
        buf.clear()

    for line in lines:
        if line.strip().startswith(_FENCE_PREFIX):
            if not in_fence:
                flush()
                buf.append(line)
                in_fence = True
            else:
                buf.append(line)
                flush(is_code=True)
                in_fence = False
            continue

        if in_fence:
            buf.append(line)
            continue

        if not line.strip():
            flush()
            continue

        buf.append(line)

    flush()
    return blocks


def _split_oversized_block(block: str) -> list[str]:
    """Split a non-code block that alone exceeds the chunk cap (e.g. a long,
    blank-line-free bullet list of PR links in release-notes.md) by line, so
    no single unit blows past the target -- unlike fenced code, prose/list
    text has no atomicity requirement to protect."""
    parts: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for line in block.splitlines():
        line_tokens = _estimate_tokens(line)
        if buf and buf_tokens + line_tokens > TARGET_MAX_TOKENS:
            parts.append("\n".join(buf))
            buf = [line]
            buf_tokens = line_tokens
        else:
            buf.append(line)
            buf_tokens += line_tokens

    if buf:
        parts.append("\n".join(buf))
    return parts


def _atomic_blocks(lines: list[str]) -> list[str]:
    """The final list of unsplittable text units for a section: fenced code
    blocks stay whole no matter their size; oversized prose/list blocks are
    broken up by line first."""
    result: list[str] = []
    for text, is_code in _split_into_blocks(lines):
        if not is_code and _estimate_tokens(text) > TARGET_MAX_TOKENS:
            result.extend(_split_oversized_block(text))
        else:
            result.append(text)
    return result


def _pack_blocks(blocks: list[str]) -> list[str]:
    """Pack atomic blocks into chunks, prioritizing the 400-token floor
    over the 600-token ceiling (see module docstring). A fenced code block
    that alone exceeds the cap still forms its own oversized chunk, since
    it's never split."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = _estimate_tokens(block)
        if current and current_tokens >= TARGET_MIN_TOKENS and (
            current_tokens + block_tokens > TARGET_MAX_TOKENS
        ):
            chunks.append("\n\n".join(current))
            current = [block]
            current_tokens = block_tokens
        else:
            current.append(block)
            current_tokens += block_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks
