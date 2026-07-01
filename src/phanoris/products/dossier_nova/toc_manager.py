# toc_manager.py

"""
toc_manager.py

Manages the TOC markdown file (typically toc/full.md) that now has the form:

    # [H1] Origins of the Theory
    ## [H1.1] Early Online Discussions
    ### [H1.1.1] Specific Forum Threads

This class provides:

- render_outline_for_prompt():
    Returns only the heading lines (with IDs) as a string, suitable
    for injecting into the LLM prompt.

- insert_block_under_heading_id(heading_id, block):
    Inserts an arbitrary markdown block immediately *after* the
    content for a given heading, but before the next heading of the
    same or higher level.

- save():
    Writes the updated markdown back to disk.

Notes:
- We treat headings as the spine; any non-heading lines in the file
  are considered content under the most recent heading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Regex for headings of the form:
#   ## [H1.2] Some Title
HEADING_RE = re.compile(
    r"^(#{1,6})\s+\[(H[0-9\.]+)\]\s+(.*\S.*)$"
)


@dataclass
class HeadingInfo:
    heading_id: str
    level: int
    title: str
    line_index: int  # index in self.lines


class TocManager:
    def __init__(self, path: str = "toc/full.md") -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"TOC file not found: {self.path}")

        text = self.path.read_text(encoding="utf-8")
        # Keep the raw lines in memory so we can insert content.
        self.lines: List[str] = text.splitlines(keepends=True)
        self._rebuild_heading_index()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str = "toc/full.md") -> "TocManager":
        """
        Convenience constructor; some legacy code may expect this.
        """
        return cls(path)

    def render_outline_for_prompt(self) -> str:
        """
        Return ONLY the heading lines (with IDs) as a single string.

        Example:

            # [H1] Origins of the Theory
            ## [H1.1] Early Online Discussions
            ## [H1.2] Historical Parallels
        """
        outline_lines: List[str] = []

        for line in self.lines:
            if HEADING_RE.match(line):
                # Strip trailing newline for prompt clarity
                outline_lines.append(line.rstrip("\n"))

        return "\n".join(outline_lines)

    def insert_block_under_heading_id(self, heading_id: str, block: str) -> None:
        """
        Insert a markdown block under the heading with the given ID.

        - We find the heading line for `heading_id`.
        - Then we scan forward until we hit:
            - EOF, or
            - The next heading with level <= this heading's level.
        - We insert the block *just before* that boundary.

        If the heading_id is unknown, we silently ignore the request.
        """
        info = self._headings_by_id.get(heading_id)
        if info is None:
            print(f"[TOC] Warning: heading ID '{heading_id}' not found; skipping insert.")
            return

        insert_at = self._find_insertion_index(info)
        # Ensure block ends with a newline
        if not block.endswith("\n"):
            block = block + "\n"

        block_lines = block.splitlines(keepends=True)

        # Insert the new content
        self.lines[insert_at:insert_at] = block_lines

        # After modifying the line list, we must rebuild heading indexes
        self._rebuild_heading_index()

    def save(self) -> None:
        """
        Persist the updated markdown back to the same file.
        """
        text = "".join(self.lines)
        self.path.write_text(text, encoding="utf-8")
        print(f"[TOC] Saved updated TOC/content to {self.path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_heading_index(self) -> None:
        """
        Scan self.lines and rebuild:
          - self._headings: a list of HeadingInfo objects in order
          - self._headings_by_id: mapping of heading_id -> HeadingInfo
        """
        self._headings: List[HeadingInfo] = []
        self._headings_by_id: Dict[str, HeadingInfo] = {}

        for idx, raw_line in enumerate(self.lines):
            line = raw_line.rstrip("\n")
            m = HEADING_RE.match(line)
            if not m:
                continue

            hashes, hid, title = m.groups()
            level = len(hashes)

            info = HeadingInfo(
                heading_id=hid,
                level=level,
                title=title.strip(),
                line_index=idx,
            )
            self._headings.append(info)
            self._headings_by_id[hid] = info

        print(f"[TOC] Indexed {len(self._headings)} headings from {self.path}")

    def _find_insertion_index(self, info: HeadingInfo) -> int:
        """
        Determine the line index where new content under `info` should be inserted.

        Strategy:
        - Start from the line *after* the heading.
        - Walk forward until we hit:
            - EOF, or
            - A heading whose level <= info.level (i.e., next sibling or parent).
        - Return the index of that boundary (we insert just before it).
        """
        start_idx = info.line_index + 1
        total_lines = len(self.lines)

        # Precompute a mapping from line index -> (level or None)
        heading_levels_by_line: Dict[int, int] = {
            h.line_index: h.level for h in self._headings
        }

        i = start_idx
        while i < total_lines:
            level_at_i = heading_levels_by_line.get(i)
            if level_at_i is not None and level_at_i <= info.level:
                # Next heading at same or higher level; insert before this line.
                return i
            i += 1

        # If we got here, we hit EOF; insert at end.
        return total_lines

    # ------------------------------------------------------------------
    # Optional helpers (if you ever need them later)
    # ------------------------------------------------------------------

    def get_heading_titles(self) -> List[Tuple[str, str]]:
        """
        Return a list of (heading_id, title) tuples in document order.
        """
        return [(h.heading_id, h.title) for h in self._headings]

    def get_title_for_id(self, heading_id: str) -> Optional[str]:
        """Return the title for a heading ID, or None if unknown."""
        info = self._headings_by_id.get(heading_id)
        return info.title if info else None

    def find_heading_by_title(self, title: str) -> Optional[HeadingInfo]:
        """
        Very simple search by exact title match.
        """
        title = title.strip()
        for h in self._headings:
            if h.title == title:
                return h
        return None
