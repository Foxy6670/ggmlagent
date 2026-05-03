"""
Memory management.

Context memory  — stored in a list of strings, serialised into the
                  KoboldCPP `memory` field.  Token-budget enforced by
                  dropping oldest lines when over limit.

Persistent mem  — plain text file (memory.md), paged for reading.
"""

from pathlib import Path
from config import PERSISTENT_MEMORY_FILE, MEMORY_TOKEN_BUDGET
from kcpp_client import KoboldClient

_PAGE_LINES = 128  # lines per page when scrolling persistent memory


class ContextMemory:
    """In-prompt volatile memory (top of context)."""

    def __init__(self, client: KoboldClient):
        self._client = client
        self._lines: list[str] = []

    # --- public interface -------------------------------------------------

    def read(self, line: int) -> str:
        """Return line N (1-indexed). Returns error string on bad index."""
        idx = line - 1
        if idx < 0 or idx >= len(self._lines):
            return f"[cmem] No line {line} (total lines: {len(self._lines)})"
        return f"[cmem:{line}] {self._lines[idx]}"

    def write(self, line: int, text: str) -> str:
        """Write/overwrite line N. Appends if line > current length."""
        idx = line - 1
        if idx < 0:
            return "[cmem] Line number must be >= 1"
        # Pad with empty lines if needed
        while len(self._lines) < idx:
            self._lines.append("")
        if idx < len(self._lines):
            self._lines[idx] = text
        else:
            self._lines.append(text)
        self._enforce_budget()
        return f"[cmem:{line}] written."

    def delete(self, line: int) -> str:
        """Delete line N (1-indexed)."""
        idx = line - 1
        if idx < 0 or idx >= len(self._lines):
            return f"[cmem] No line {line} (total lines: {len(self._lines)})"
        self._lines.pop(idx)
        return f"[cmem:{line}] deleted."

    def render(self) -> str:
        """Serialise to a string suitable for the `memory` API field."""
        if not self._lines:
            return ""
        return "\n".join(self._lines) + "\n"

    # --- internals --------------------------------------------------------

    def _enforce_budget(self):
        """Drop oldest lines until we're within the token budget."""
        while self._lines:
            token_count = self._client.tokenize(self.render())
            if token_count <= MEMORY_TOKEN_BUDGET:
                break
            self._lines.pop(0)  # drop oldest


class PersistentMemory:
    """File-backed memory with paged reading."""

    def __init__(self):
        self._path = Path(PERSISTENT_MEMORY_FILE)
        self._page = 0  # current page index (0-based)

    # --- public interface -------------------------------------------------

    def read_page(self) -> str:
        lines = self._all_lines()
        total_pages = max(1, -(-len(lines) // _PAGE_LINES))  # ceil div
        start = self._page * _PAGE_LINES
        chunk = lines[start : start + _PAGE_LINES]
        if not chunk:
            return f"[pmem] Empty. (page {self._page + 1}/{total_pages})"
        header = f"[pmem page {self._page + 1}/{total_pages}]\n"
        numbered = "\n".join(
            f"{start + i + 1:4d}: {line.rstrip()}"
            for i, line in enumerate(chunk)
        )
        return header + numbered

    def write(self, text: str) -> str:
        """
        Prepend a new memory entry to the top of the file.

        Memory access is recency-weighted — the agent almost always cares
        more about "what did I learn recently?" than "what did I learn
        first?". Storing newest-first means /pmem r shows current context
        immediately rather than requiring /pgdown to find recent entries.
        Older memories live below, reachable via /pgdown.

        Costs O(N) per write (full-file rewrite), but the file is small
        and writes are infrequent — not a perf concern.
        """
        existing = self._path.read_text(encoding="utf-8") if self._path.exists() else ""
        new_line = text.rstrip("\n") + "\n"
        self._path.write_text(new_line + existing, encoding="utf-8")
        return "[pmem] Memory saved."

    def delete(self, line: int) -> str:
        lines = self._all_lines()
        idx = line - 1
        if idx < 0 or idx >= len(lines):
            return f"[pmem] No line {line} (total lines: {len(lines)})"
        lines.pop(idx)
        self._path.write_text("".join(lines), encoding="utf-8")
        return f"[pmem:{line}] deleted."

    def page_up(self) -> str:
        if self._page > 0:
            self._page -= 1
        return self.read_page()

    def page_down(self) -> str:
        lines = self._all_lines()
        total_pages = max(1, -(-len(lines) // _PAGE_LINES))
        if self._page < total_pages - 1:
            self._page += 1
        return self.read_page()

    # --- internals --------------------------------------------------------

    def _all_lines(self) -> list[str]:
        if not self._path.exists():
            return []
        return self._path.read_text(encoding="utf-8").splitlines(keepends=True)
