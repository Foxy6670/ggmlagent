"""
Multi-file patch parser and applier.

Ported from openclaw/src/agents/apply-patch.ts and apply-patch-update.ts.

Patch format:
    *** Begin Patch
    *** Add File: path/to/new.txt
    +line 1
    +line 2
    *** Update File: src/foo.py
    @@ optional anchor (e.g. function name)
    -old line
    +new line
     unchanged context line
    *** Delete File: obsolete.txt
    *** End Patch

Update hunks use unified-diff-style markers:
  '+'  adds the line
  '-'  removes the line
  ' '  (single leading space) keeps the line as context
  ''   blank line is treated as a blank context line on both sides

Search is exact first, then progressively more lenient (rstrip, strip, then
unicode-punctuation normalization) so small whitespace mismatches still apply.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Union


# ── Markers ──────────────────────────────────────────────────────────────

BEGIN_PATCH      = "*** Begin Patch"
END_PATCH        = "*** End Patch"
ADD_FILE         = "*** Add File: "
DELETE_FILE      = "*** Delete File: "
UPDATE_FILE      = "*** Update File: "
MOVE_TO          = "*** Move to: "
EOF_MARKER       = "*** End of File"
CTX_MARKER       = "@@ "
EMPTY_CTX_MARKER = "@@"


class PatchError(Exception):
    pass


# ── Hunk types ───────────────────────────────────────────────────────────

@dataclass
class AddHunk:
    path:     str
    contents: str


@dataclass
class DeleteHunk:
    path: str


@dataclass
class UpdateChunk:
    change_context: str | None = None
    old_lines:      list[str]   = field(default_factory=list)
    new_lines:      list[str]   = field(default_factory=list)
    is_eof:         bool        = False


@dataclass
class UpdateHunk:
    path:      str
    move_path: str | None = None
    chunks:    list[UpdateChunk] = field(default_factory=list)


Hunk = Union[AddHunk, DeleteHunk, UpdateHunk]


@dataclass
class ApplyResult:
    added:    list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted:  list[str] = field(default_factory=list)


# ── Parser ───────────────────────────────────────────────────────────────

def parse_patch(text: str) -> list[Hunk]:
    """Parse patch text into hunks. Raises PatchError on malformed input."""
    trimmed = text.strip()
    if not trimmed:
        raise PatchError("Empty patch.")
    lines = trimmed.split("\n")
    if lines[0].strip() != BEGIN_PATCH:
        raise PatchError("First line of patch must be '*** Begin Patch'.")
    if lines[-1].strip() != END_PATCH:
        raise PatchError("Last line of patch must be '*** End Patch'.")

    hunks: list[Hunk] = []
    remaining = lines[1:-1]
    line_no = 2
    while remaining:
        hunk, consumed = _parse_one_hunk(remaining, line_no)
        hunks.append(hunk)
        line_no += consumed
        remaining = remaining[consumed:]
    return hunks


def _parse_one_hunk(lines: list[str], line_no: int) -> tuple[Hunk, int]:
    if not lines:
        raise PatchError(f"Line {line_no}: empty hunk")
    first = lines[0].strip()

    if first.startswith(ADD_FILE):
        path = first[len(ADD_FILE):]
        contents = ""
        consumed = 1
        for line in lines[1:]:
            if line.startswith("+"):
                contents += line[1:] + "\n"
                consumed += 1
            else:
                break
        return AddHunk(path=path, contents=contents), consumed

    if first.startswith(DELETE_FILE):
        return DeleteHunk(path=first[len(DELETE_FILE):]), 1

    if first.startswith(UPDATE_FILE):
        path = first[len(UPDATE_FILE):]
        remaining = lines[1:]
        consumed = 1
        move_path: str | None = None

        if remaining and remaining[0].strip().startswith(MOVE_TO):
            move_path = remaining[0].strip()[len(MOVE_TO):]
            remaining = remaining[1:]
            consumed += 1

        chunks: list[UpdateChunk] = []
        while remaining:
            if remaining[0].strip() == "":
                remaining = remaining[1:]
                consumed += 1
                continue
            if remaining[0].startswith("***"):
                break
            chunk, used = _parse_chunk(
                remaining,
                line_no + consumed,
                allow_missing_context=(len(chunks) == 0),
            )
            chunks.append(chunk)
            remaining = remaining[used:]
            consumed += used

        if not chunks:
            raise PatchError(f"Line {line_no}: update hunk for '{path}' is empty")
        return UpdateHunk(path=path, move_path=move_path, chunks=chunks), consumed

    raise PatchError(
        f"Line {line_no}: '{lines[0]}' is not a valid hunk header. "
        "Use '*** Add File:', '*** Delete File:', or '*** Update File:'."
    )


def _parse_chunk(
    lines: list[str], line_no: int, allow_missing_context: bool,
) -> tuple[UpdateChunk, int]:
    if not lines:
        raise PatchError(f"Line {line_no}: empty chunk")

    change_context: str | None = None
    start_idx = 0
    if lines[0] == EMPTY_CTX_MARKER:
        start_idx = 1
    elif lines[0].startswith(CTX_MARKER):
        change_context = lines[0][len(CTX_MARKER):]
        start_idx = 1
    elif not allow_missing_context:
        raise PatchError(
            f"Line {line_no}: expected '@@' context marker, got: {lines[0]!r}"
        )

    if start_idx >= len(lines):
        raise PatchError(f"Line {line_no + 1}: chunk has no content")

    chunk = UpdateChunk(change_context=change_context)
    parsed = 0
    for line in lines[start_idx:]:
        if line == EOF_MARKER:
            if parsed == 0:
                raise PatchError(f"Line {line_no + 1}: chunk has no content")
            chunk.is_eof = True
            parsed += 1
            break

        if not line:
            chunk.old_lines.append("")
            chunk.new_lines.append("")
            parsed += 1
            continue

        marker = line[0]
        if marker == " ":
            content = line[1:]
            chunk.old_lines.append(content)
            chunk.new_lines.append(content)
            parsed += 1
        elif marker == "+":
            chunk.new_lines.append(line[1:])
            parsed += 1
        elif marker == "-":
            chunk.old_lines.append(line[1:])
            parsed += 1
        else:
            if parsed == 0:
                raise PatchError(
                    f"Line {line_no + 1}: chunk lines must begin with ' ', '+', or '-'; "
                    f"got: {line!r}"
                )
            break

    return chunk, parsed + start_idx


# ── Update application ───────────────────────────────────────────────────

def apply_update(content: str, chunks: list[UpdateChunk], file_path: str) -> str:
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    replacements = _compute_replacements(lines, file_path, chunks)
    new_lines = _apply_replacements(lines, replacements)
    if not new_lines or new_lines[-1] != "":
        new_lines.append("")
    return "\n".join(new_lines)


def _compute_replacements(
    lines: list[str], file_path: str, chunks: list[UpdateChunk],
) -> list[tuple[int, int, list[str]]]:
    replacements: list[tuple[int, int, list[str]]] = []
    line_idx = 0

    for chunk in chunks:
        if chunk.change_context:
            ctx_idx = _seek_sequence(lines, [chunk.change_context], line_idx, False)
            if ctx_idx is None:
                raise PatchError(
                    f"Failed to find context '{chunk.change_context}' in {file_path}"
                )
            line_idx = ctx_idx + 1

        if not chunk.old_lines:
            insertion = (
                len(lines) - 1 if lines and lines[-1] == "" else len(lines)
            )
            replacements.append((insertion, 0, list(chunk.new_lines)))
            continue

        pattern   = list(chunk.old_lines)
        new_slice = list(chunk.new_lines)
        found = _seek_sequence(lines, pattern, line_idx, chunk.is_eof)

        if found is None and pattern and pattern[-1] == "":
            pattern = pattern[:-1]
            if new_slice and new_slice[-1] == "":
                new_slice = new_slice[:-1]
            found = _seek_sequence(lines, pattern, line_idx, chunk.is_eof)

        if found is None:
            raise PatchError(
                f"Failed to find expected lines in {file_path}:\n"
                + "\n".join(chunk.old_lines)
            )
        replacements.append((found, len(pattern), new_slice))
        line_idx = found + len(pattern)

    replacements.sort(key=lambda r: r[0])
    return replacements


def _apply_replacements(
    lines: list[str], replacements: list[tuple[int, int, list[str]]],
) -> list[str]:
    result = list(lines)
    for start_idx, old_len, new_lines in reversed(replacements):
        for _ in range(old_len):
            if start_idx < len(result):
                del result[start_idx]
        for i, new_line in enumerate(new_lines):
            result.insert(start_idx + i, new_line)
    return result


def _seek_sequence(
    lines: list[str], pattern: list[str], start: int, eof: bool,
) -> int | None:
    if not pattern:
        return start
    if len(pattern) > len(lines):
        return None

    max_start = len(lines) - len(pattern)
    search_start = max_start if (eof and len(lines) >= len(pattern)) else start
    if search_start > max_start:
        return None

    # Try progressively more lenient matching: exact → rstrip → strip →
    # unicode-punctuation normalized strip.
    for normalize in (
        lambda v: v,
        lambda v: v.rstrip(),
        lambda v: v.strip(),
        lambda v: _normalize_punct(v.strip()),
    ):
        for i in range(search_start, max_start + 1):
            if all(
                normalize(lines[i + j]) == normalize(pattern[j])
                for j in range(len(pattern))
            ):
                return i
    return None


_PUNCT_MAP = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "　": " ",
})


def _normalize_punct(s: str) -> str:
    return s.translate(_PUNCT_MAP)


# ── Top-level apply ──────────────────────────────────────────────────────

def apply_patch(text: str, *, safe_path: Callable[[str], Path]) -> ApplyResult:
    """
    Apply a patch.  *safe_path* validates each path from the patch and
    returns an absolute Path object — typically the dispatcher's _safe_path.
    """
    hunks = parse_patch(text)
    if not hunks:
        raise PatchError("No files were modified.")

    result = ApplyResult()
    seen   = {"added": set(), "modified": set(), "deleted": set()}

    def _record(bucket: str, name: str):
        if name in seen[bucket]:
            return
        seen[bucket].add(name)
        getattr(result, bucket).append(name)

    for hunk in hunks:
        if isinstance(hunk, AddHunk):
            target = safe_path(hunk.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(hunk.contents, encoding="utf-8")
            _record("added", hunk.path)

        elif isinstance(hunk, DeleteHunk):
            target = safe_path(hunk.path)
            target.unlink()
            _record("deleted", hunk.path)

        else:  # UpdateHunk
            target = safe_path(hunk.path)
            content = target.read_text(encoding="utf-8")
            new_content = apply_update(content, hunk.chunks, hunk.path)
            if hunk.move_path:
                move_target = safe_path(hunk.move_path)
                move_target.parent.mkdir(parents=True, exist_ok=True)
                move_target.write_text(new_content, encoding="utf-8")
                target.unlink()
                _record("modified", hunk.move_path)
            else:
                target.write_text(new_content, encoding="utf-8")
                _record("modified", hunk.path)

    return result


def format_summary(result: ApplyResult) -> str:
    lines = ["[patch] Success."]
    for f in result.added:    lines.append(f"  A {f}")
    for f in result.modified: lines.append(f"  M {f}")
    for f in result.deleted:  lines.append(f"  D {f}")
    return "\n".join(lines)
