"""
Command parser and dispatcher.

Commands are matched against completed lines from the LLM output.
Each handler returns a result string that gets injected back into
the prompt as an observation.

Interactive operations (/appendlines, /edit) use a PendingEdit state
machine.  While pending is not None, the agent loop feeds each
completed non-blank line to handle_pending_input() instead of
treating it as a command.
"""

import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from memory import ContextMemory, PersistentMemory
import web as web_mod
import moltbook as mb_mod
from apply_patch import (
    apply_patch          as _apply_patch,
    format_summary       as _format_patch_summary,
    is_unified_diff      as _is_unified_diff,
    unified_diff_to_patch as _unified_diff_to_patch,
    PatchError        as _PatchError,
    BEGIN_PATCH, END_PATCH,
)

_WORK_DIR = Path(".").resolve()
_FILE_PAGE_LINES = 100
_frwx_enabled = False   # set True by CommandDispatcher when --frwx is active
_chroot_root: "Path | None" = None  # set by CommandDispatcher when --chroot is active

_PAGE_SIZE      = 6000  # chars per web page; ~1 500 tokens — fits comfortably in context
_TG_HISTORY_N   = 30    # recent Telegram messages shown by bare /telegram

_SHELL_TIMEOUT  = 30    # seconds before a shell command is killed
_SHELL_MAX_OUT  = 4000  # chars; longer output gets head+tail trimmed
_SHELL_HEAD     = 3000
_SHELL_TAIL     = 800


class CommandError(Exception):
    pass


# ---------------------------------------------------------------------------
# Pending-edit state machine
# ---------------------------------------------------------------------------

@dataclass
class PendingEdit:
    """
    Tracks an in-progress interactive session.

    mode      : "appendlines" | "edit_old" | "edit_new"
    file_path : path relative to _WORK_DIR
    old_lines : accumulated during "edit_old" phase
    new_lines : accumulated during "edit_new" phase
    """
    mode:      str
    file_path: str
    old_lines: list[str] = field(default_factory=list)
    new_lines: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Paginated file reader
# ---------------------------------------------------------------------------

class FileReader:
    def __init__(self):
        self._files: dict[str, tuple[list[str], int]] = {}

    def read(self, path_str: str) -> str:
        p = _safe_path(path_str)
        if not p.exists():
            return f"[file] Not found: {path_str}"
        if not p.is_file():
            return f"[file] Not a file: {path_str}"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        self._files[path_str] = (lines, 0)
        return self._render(path_str)

    def page_up(self, path_str: str) -> str:
        if path_str not in self._files:
            return f"[file] Not open: {path_str} — use /read first"
        lines, page = self._files[path_str]
        if page > 0:
            self._files[path_str] = (lines, page - 1)
        return self._render(path_str)

    def page_down(self, path_str: str) -> str:
        if path_str not in self._files:
            return f"[file] Not open: {path_str} — use /read first"
        lines, page = self._files[path_str]
        total = max(1, -(-len(lines) // _FILE_PAGE_LINES))
        if page < total - 1:
            self._files[path_str] = (lines, page + 1)
        return self._render(path_str)

    def _render(self, path_str: str) -> str:
        lines, page = self._files[path_str]
        total = max(1, -(-len(lines) // _FILE_PAGE_LINES))
        start = page * _FILE_PAGE_LINES
        chunk = lines[start : start + _FILE_PAGE_LINES]
        header = f"[file:{path_str} page {page + 1}/{total}]\n"
        numbered = "".join(
            f"{start + i + 1:4d}: {line}" for i, line in enumerate(chunk)
        )
        return header + numbered


_JAIL_BLOCKED_NAMES = frozenset({".secrets", "hosts.yml"})

def _safe_path(path_str: str) -> Path:
    if Path(path_str).name in _JAIL_BLOCKED_NAMES:
        raise CommandError(f"[file] Access denied (credential file): {path_str}")
    if _chroot_root is not None:
        raw = Path(path_str)
        if raw.is_absolute():
            # Absolute paths are jailed: /etc/foo → <jail>/etc/foo
            p = (_chroot_root / raw.relative_to("/")).resolve()
            if not str(p).startswith(str(_chroot_root)):
                raise CommandError(f"[file] Access denied (outside jail): {path_str}")
        else:
            # Relative paths resolve from the workspace (harness-managed files)
            p = (_WORK_DIR / raw).resolve()
            if not str(p).startswith(str(_WORK_DIR)):
                raise CommandError(f"[file] Access denied (outside workspace): {path_str}")
        return p
    p = (Path(".").resolve() / path_str).resolve()
    if not _frwx_enabled and not str(p).startswith(str(_WORK_DIR)):
        raise CommandError(f"[file] Access denied (outside working directory): {path_str}")
    return p


def _read_lines(path_str: str) -> list[str]:
    """Read a file as lines; return [] if it doesn't exist yet."""
    p = _safe_path(path_str)
    if not p.exists():
        return []
    return p.read_text(encoding="utf-8", errors="replace").splitlines()


def _write_lines(path_str: str, lines: list[str]) -> None:
    p = _safe_path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_TG_COOLDOWN = 60  # seconds between /telegram commands

_MB_ERROR_THRESHOLD = 3     # consecutive 5xx errors before cooling off
_MB_COOLOFF_SECS    = 300   # 5 minutes

class CommandDispatcher:
    def __init__(
        self,
        cmem: ContextMemory,
        pmem: PersistentMemory,
        frwx: bool = False,
        telegram: bool = False,
        monero: bool = False,
        sim=None,
        chroot: str = "",
    ):
        global _frwx_enabled, _chroot_root
        _frwx_enabled = frwx
        self._frwx = frwx
        self._chroot: str = chroot  # jail path for shell dispatch; "" = disabled
        _chroot_root = Path(chroot).resolve() if chroot else None
        self.cmem = cmem
        self.pmem = pmem
        self._file_reader = FileReader()
        self.pending: PendingEdit | None = None
        self._last_tg_text: str = ""
        self._last_tg_time: float = 0.0
        self._mb_consec_errors: int = 0
        self._mb_cooloff_until: float = 0.0
        self._page_buf:    list[str] = []  # pages of the last /search or /goto result
        self._page_cur:    int = 0
        self._sim = sim
        # When chroot is active the persistent CWD is jail-relative; start at jail root.
        self._cwd: str = "/" if chroot else str(Path.cwd())

        # When simulating, the sim object replaces the real Telegram handler —
        # it duck-types `send`/`drain_inbox`/`history`. Moltbook write actions
        # are intercepted in `_mb` rather than via a handler swap.
        if sim is not None:
            self._tg = sim
        elif telegram:
            import telegram_handler as _tg_mod
            self._tg = _tg_mod
        else:
            self._tg = None

        if monero:
            import monero_wallet as _xmr_mod
            self._xmr = _xmr_mod
        else:
            self._xmr = None

    # -----------------------------------------------------------------------
    # Body-block dispatch (multiline command content)
    # -----------------------------------------------------------------------

    def dispatch_block(self, line: str, body: str) -> str:
        """
        Dispatch a command that arrived inside a \"\"\" / ``` body block.

        *line* is the first line of the block (command + args).
        *body* is everything between line 1 and the closing delimiter.

        Commands that understand multiline content (/mb post, /mb comment,
        /mb reply, /telegram) use *body* as their content.  Everything else
        falls back to normal single-line dispatch and ignores the body.
        """
        parts = line.split(None, 1)
        if not parts:
            return "[block] Empty — put the command on the first line inside the block."
        cmd  = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if cmd == "/mb":
            return self._mb_block(rest, body)
        if cmd == "/telegram":
            text = (rest.strip() + "\n" + body).strip() if body.strip() else rest.strip()
            return self._telegram(text.split(" ", 1) if " " in text else ([text] if text else []))
        if cmd in ("$", "#") and self._frwx:
            return self._shell_block(cmd, rest, body)
        if cmd in ("/pmem", "/pm"):
            if rest.lower().startswith("w") and body.strip():
                inline = rest[1:].strip()
                body_text = " ".join(l.strip() for l in body.strip().splitlines() if l.strip())
                text = (inline + " " + body_text).strip() if inline else body_text
                return self._pmem("/pmem", ["w", text])
            return self.dispatch(line)
        if cmd == "/patch":
            if body.strip():
                # Feed body through the streaming accumulator all at once.
                # _patch_input returns "" for each batch line and the result
                # string when *** End Patch is reached.
                self.pending = PendingEdit(mode="patch", file_path="")
                for ln in body.rstrip().splitlines():
                    result = self._patch_input(ln)
                    if result:  # *** End Patch was in the body
                        return result
                # Model omitted the sentinel — fire it now.
                return self._patch_input(END_PATCH)
            return self._begin_patch([])
        if cmd in ("/appendlines", "/edit"):
            # Single-shot: the whole multi-line payload arrives in *body*.  Begin
            # the interactive session, then drive it line-by-line through the same
            # handler the streaming loop used, so the protocol stays identical.
            begin = self.dispatch(line)
            if self.pending is None:
                return begin  # begin failed (bad path / file not found)
            if not body.strip():
                self.pending = None
                return f"[block] {cmd} needs its content in the body."
            last = begin
            for ln in body.rstrip("\n").splitlines():
                last = self.handle_pending_input(ln)
                if self.pending is None:
                    break  # terminator reached ('done' / edit applied)
            if self.pending is not None:
                mode = self.pending.mode
                if mode == "appendlines":
                    # No 'done' in body — lines were still written; finalize.
                    last = self.handle_pending_input("done")
                else:  # edit_old / edit_new — incomplete, nothing applied
                    self.pending = None
                    last = (
                        "[edit] Not applied — the body must contain the exact old "
                        "text, a line with only ---, the new text, then a line with "
                        "only done."
                    )
            return last
        # Fall back: dispatch the command line as-is, body ignored.
        result = self.dispatch(line)
        if result:
            return result
        stripped = line.strip()
        if stripped and not stripped.startswith("/"):
            return (
                "[block] Missing prefix — did you forget `$` or `#`? "
                "Use `$ <cmd>` to run as user, `# <cmd>` to run as root, "
                "or `/command` for harness commands."
            )
        return "[block] Command not recognised."

    def _mb_block(self, rest: str, body: str) -> str:
        """Handle /mb commands that arrived with a multiline body."""
        try:
            return self._mb_block_inner(rest, body)
        except mb_mod.MoltbookError as e:
            return f"[mb] Error: {e}"
        except Exception as e:
            return f"[mb] Unexpected error: {type(e).__name__}: {e}"

    def _mb_block_inner(self, rest: str, body: str) -> str:
        args = rest.split()
        if not args:
            return '[mb] Usage in block:\n  """\n  /mb post <submolt> <title>\n  <body>\n  """'
        sub = args[0].lower()
        rest_args = args[1:]

        if sub == "post":
            if not rest_args:
                return "[mb] Usage: /mb post <submolt> <title> (body follows in block)"
            submolt    = rest_args[0]
            title_args = rest_args[1:]
            if "|" in title_args:
                pipe         = title_args.index("|")
                title        = " ".join(title_args[:pipe])
                inline_body  = " ".join(title_args[pipe + 1:])
                body         = (inline_body + "\n" + body).strip() if body else inline_body
            else:
                title = " ".join(title_args)
            return mb_mod.create_post(submolt, title, body)

        if sub == "comment":
            if not rest_args:
                return "[mb] Usage: /mb comment <post_id> (body follows in block)"
            return mb_mod.comment(rest_args[0], body)

        if sub == "reply":
            if len(rest_args) < 2:
                return "[mb] Usage: /mb reply <post_id> <comment_id> (body follows in block)"
            return mb_mod.comment(rest_args[0], body, parent_id=rest_args[1])

        # Not a body-aware subcommand — fall back to normal dispatch.
        return self._mb_dispatch(args)

    def _shell_block(self, prefix: str, cmd: str, body: str) -> str:
        """Run a shell block: pipe *body* as stdin to *cmd*, or run as bash script."""
        import shlex, tempfile, os
        cmd = cmd.strip()
        # When chroot is active temp files must live inside the jail so the chrooted
        # shell can reach them.  Write to <jail>/tmp/ on the host; reference as
        # /tmp/<name> from inside the jail.
        tmp_dir = os.path.join(self._chroot, "tmp") if self._chroot else None
        if cmd:
            # Write body to a temp file and pipe it as stdin.
            with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp",
                                             delete=False, encoding="utf-8",
                                             dir=tmp_dir) as f:
                f.write(body)
                fname_host = f.name
            fname_ref = os.path.join("/tmp", os.path.basename(fname_host)) if self._chroot else fname_host
            actual = f"{prefix} {cmd} < {shlex.quote(fname_ref)}"
        else:
            # No command specified — execute body as a bash script.
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh",
                                             delete=False, encoding="utf-8",
                                             dir=tmp_dir) as f:
                f.write(body)
                fname_host = f.name
            fname_ref = os.path.join("/tmp", os.path.basename(fname_host)) if self._chroot else fname_host
            actual = f"{prefix} bash {shlex.quote(fname_ref)}"
        try:
            return self.dispatch(actual) or "[block] Shell block executed."
        finally:
            try:
                os.unlink(fname_host)
            except OSError:
                pass

    # -----------------------------------------------------------------------
    # Normal command dispatch
    # -----------------------------------------------------------------------

    def dispatch(self, line: str) -> str | None:
        """
        Try to parse *line* as a command.
        Returns the result string, or None if the line isn't a command.
        /appendlines and /edit set self.pending and return a prompt string.
        """
        stripped = line.strip()
        if stripped.startswith("$ ") or (stripped.startswith("# ") and not stripped.startswith("##")):
            if not self._frwx:
                return "[error] Shell access requires --frwx flag."
            root = stripped.startswith("# ")
            cmd_str = stripped[2:].strip()
            placeholders = re.findall(r'<[a-zA-Z_][a-zA-Z0-9_\-]*>', cmd_str)
            if placeholders:
                return (
                    f"[system] Command contains unsubstituted placeholder(s): "
                    f"{', '.join(placeholders)}. Replace each with an actual value before running."
                )
            return self._shell(cmd_str, root=root)
        if not stripped.startswith("/"):
            return None
        try:
            return self._route(stripped)
        except CommandError as e:
            return str(e)
        except Exception as e:
            return f"[error] {type(e).__name__}: {e}"

    def _route(self, line: str) -> str:
        try:
            parts = shlex.split(line)
        except ValueError:
            parts = line.split()

        cmd  = parts[0].lower()
        args = parts[1:]

        if cmd == "/cmem":         return self._cmem(args)
        # '/cm' is a single BPE token in Qwen3's vocabulary.  The model often
        # emits it alone when it intends '/cmem r 1' — the ChatML end-of-turn
        # token (<|im_end|>) wins the probability race immediately after '/cm',
        # so generation stops before 'em r 1' can be produced.  Alias it.
        if cmd == "/cm":
            if args:
                # /cm w|r|d ... — pass subcommands straight through
                return self._cmem(args)
            # Bare /cm — shorthand for reading line 1
            result = self._cmem(["r", "1"])
            return f"[cmem] (alias /cm → /cmem r 1) {result}"
        if cmd in ("/pmem", "/pgup", "/pgdown"):
                                   return self._pmem(cmd, args)
        if cmd in ("/pm", "/pmew"):
            # /pm  — shorthand for /pmem r
            # /pmew — model often emits this instead of "/pmem w" (BPE fusion)
            if args:
                return self._pmem("/pmem", args)
            return self._pmem("/pmem", ["r"])
        if cmd == "/dir":          return self._dir(args)
        if cmd == "/read":         return self._read(args)
        if cmd == "/append":       return self._append(args)
        if cmd == "/appendlines":  return self._begin_appendlines(args)
        if cmd == "/edit":         return self._begin_edit(args)
        if cmd == "/patch":        return self._begin_patch(args)
        if cmd == "/dellines":     return self._dellines(args)
        if cmd == "/del":          return self._del(args)
        if cmd == "/search":       return self._search(args)
        if cmd == "/goto":         return self._goto(args)
        if cmd == "/next":         return self._next()
        if cmd == "/back":         return self._back()
        if cmd == "/mb":           return self._mb(args)
        if cmd == "/telegram":     return self._telegram(args)
        if cmd == "/wallet":       return self._wallet(args)

        return f"[error] Unknown command: {cmd}"

    # -----------------------------------------------------------------------
    # Pending-edit input handler
    # -----------------------------------------------------------------------

    def handle_pending_input(self, text: str) -> str:
        """
        Called by the agent loop when self.pending is set.
        *text* is a single completed line from the model with leading
        whitespace preserved (only trailing newlines stripped) — handlers
        that look for keywords (done, ---, *** End Patch) call .strip().
        Returns the next observation string.
        Clears self.pending when the session ends.
        """
        if self.pending is None:
            return "[error] handle_pending_input called with no pending state"

        if self.pending.mode == "appendlines":
            return self._appendlines_input(text)
        if self.pending.mode == "patch":
            return self._patch_input(text)
        if self.pending.mode in ("edit_old", "edit_new"):
            return self._edit_input(text)

        self.pending = None
        return "[error] Unknown pending mode"

    # -----------------------------------------------------------------------
    # Context memory
    # -----------------------------------------------------------------------

    def _cmem(self, args: list[str]) -> str:
        if not args:
            raise CommandError("[cmem] Usage: /cmem r|w|d <line> [text]")
        sub = args[0].lower()
        if sub == "r":
            return "[cmem] Context memory is already visible in your prompt — you do not need to read it. Use /cmem w <line> <text> to write or /cmem d <line> to delete."
        if sub == "w":
            line = _int_arg(args, 1, "/cmem w <line> <text>")
            return self.cmem.write(line, " ".join(args[2:]))
        if sub == "d":
            return self.cmem.delete(_int_arg(args, 1, "/cmem d <line>"))
        raise CommandError(f"[cmem] Unknown subcommand: {sub}")

    # -----------------------------------------------------------------------
    # Persistent memory
    # -----------------------------------------------------------------------

    def _pmem(self, cmd: str, args: list[str]) -> str:
        if cmd == "/pgup":   return self.pmem.page_up()
        if cmd == "/pgdown": return self.pmem.page_down()
        if not args:
            raise CommandError("[pmem] Usage: /pmem r | /pmem w <text> | /pmem w <line> <text> | /pmem d <line>")
        sub = args[0].lower()
        if sub == "r": return self.pmem.read_page()
        if sub == "w":
            # /pmem w <n> <text>  — update line n in place
            if len(args) >= 3 and args[1].isdigit():
                line = int(args[1])
                text = " ".join(args[2:])
                return self.pmem.update(line, text)
            # /pmem w <text>  — prepend new entry (no line number needed)
            text = " ".join(args[1:])
            if not text.strip():
                raise CommandError("[pmem] Usage: /pmem w <text>")
            if len(text) > 300:
                text = text[:297] + "…"
                result = self.pmem.write(text)
                return result + "\n[pmem] Note: entry was truncated to 300 chars. Use shorter entries."
            return self.pmem.write(text)
        if sub == "d": return self.pmem.delete(_int_arg(args, 1, "/pmem d <line>"))
        raise CommandError(f"[pmem] Unknown subcommand: {sub}")

    # -----------------------------------------------------------------------
    # Files
    # -----------------------------------------------------------------------

    def _dir(self, args: list[str]) -> str:
        path_str = args[0] if args else "."
        p = _safe_path(path_str)
        if not p.exists():   return f"[dir] Not found: {path_str}"
        if not p.is_dir():   return f"[dir] Not a directory: {path_str}"
        entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        lines = []
        for e in entries:
            kind = "" if e.is_dir() else "  "
            size = f"  {e.stat().st_size:>10} B" if e.is_file() else ""
            lines.append(f"  {kind}{e.name}{size}")
        return f"[dir:{path_str}]\n" + "\n".join(lines)

    def _read(self, args: list[str]) -> str:
        if not args:
            raise CommandError("[file] Usage: /read <file>")
        return self._file_reader.read(args[0])

    def _append(self, args: list[str]) -> str:
        if len(args) < 2:
            raise CommandError("[file] Usage: /append <file> <content>")
        path_str, content = args[0], " ".join(args[1:])
        p = _safe_path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(content + "\n")
        return f"[file] Appended to: {path_str}"

    def _begin_appendlines(self, args: list[str]) -> str:
        """
        /appendlines <file>  — append multiple lines interactively.

        The agent types one line of content per response; each is appended
        immediately.  Type 'done' to finish.  Designed for writing structured
        multi-line entries (e.g. the events.md format).
        """
        if not args:
            raise CommandError("[file] Usage: /appendlines <file>")
        path_str = args[0]
        _safe_path(path_str)  # validate path before setting pending
        self.pending = PendingEdit(mode="appendlines", file_path=path_str)
        return (
            f"[appendlines:{path_str}] Ready. Write all your lines now "
            "(the system reads each line as you go). "
            "Type 'done' alone on its own line when finished."
        )

    def _appendlines_input(self, text: str) -> str:
        p = self.pending
        assert p is not None
        if text.strip().lower() in ("done", "exit", "quit"):
            self.pending = None
            return f"[appendlines:{p.file_path}] Done."
        path_obj = _safe_path(p.file_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
        return f"[appendlines:{p.file_path}] Appended. Next line (or 'done'):"

    def _begin_edit(self, args: list[str]) -> str:
        """
        /edit <file>  — find-and-replace (or delete) a block of text.

        Write old text, then --- alone on its own line, then replacement text,
        then done alone on its own line.  Leave replacement empty (--- then done
        immediately) to delete the old text.  Use /read first to copy exact text.
        """
        if not args:
            raise CommandError("[file] Usage: /edit <file>")
        path_str = args[0]
        p = _safe_path(path_str)
        if not p.exists():
            return f"[file] Not found: {path_str}"
        self.pending = PendingEdit(mode="edit_old", file_path=path_str)
        return (
            f"[edit:{path_str}] In THIS block, before you close it, write: the exact "
            f"text to find, then --- alone on a line, then the replacement text, then "
            f"done alone on a line (empty replacement = delete). Do not close the block "
            f"until after 'done'."
        )

    def _edit_input(self, text: str) -> str:
        p = self.pending
        assert p is not None

        if p.mode == "edit_old":
            if text.strip() == "---":
                if not p.old_lines:
                    self.pending = None
                    return "[edit] Cancelled — no text to find."
                p.mode = "edit_new"
                return f"[edit:{p.file_path}] Separator. Write replacement, then done:"
            p.old_lines.append(text)
            return f"[edit:{p.file_path}] Line {len(p.old_lines)} recorded."

        if p.mode == "edit_new":
            if text.strip().lower() == "done":
                return self._apply_edit()
            p.new_lines.append(text)
            return f"[edit:{p.file_path}] Replacement line {len(p.new_lines)} recorded."

        self.pending = None
        return "[error] Unexpected edit state."

    def _apply_edit(self) -> str:
        p = self.pending
        assert p is not None
        self.pending = None

        file_path = _safe_path(p.file_path)
        content   = file_path.read_text(encoding="utf-8", errors="replace")

        old_text = "\n".join(p.old_lines)
        new_text = "\n".join(p.new_lines)

        # Prefer matching with a trailing newline so that deleting a block of
        # lines also consumes the line terminator (avoids leaving a blank line).
        if old_text + "\n" in content:
            replacement = (new_text + "\n") if new_text else ""
            updated = content.replace(old_text + "\n", replacement, 1)
        elif old_text in content:
            updated = content.replace(old_text, new_text, 1)
        else:
            return (
                f"[edit:{p.file_path}] ERROR — text not found in file.\n"
                "Make sure you copied the exact text from /read output (including spacing).\n"
                "Use /edit again with the correct text."
            )

        file_path.write_text(updated, encoding="utf-8")

        if p.new_lines:
            return f"[edit:{p.file_path}] Done. Replaced {len(p.old_lines)} line(s) with {len(p.new_lines)} line(s)."
        else:
            return f"[edit:{p.file_path}] Done. Deleted {len(p.old_lines)} line(s)."

    # -----------------------------------------------------------------------
    # Multi-file patch (apply_patch format)
    # -----------------------------------------------------------------------

    def _begin_patch(self, args: list[str]) -> str:
        """/patch — apply a multi-file patch. Accepts unified diff or apply_patch format."""
        # old_lines is reused as the line buffer; file_path is unused for /patch.
        self.pending = PendingEdit(mode="patch", file_path="")
        return (
            "[patch] Paste your patch, then write '*** End Patch' on its own line to apply.\n"
            "Accepts standard unified diff (diff -u / git diff) OR the custom format:\n"
            "  *** Begin Patch\n"
            "  *** Update File: path/to/file\n"
            "  @@ optional anchor\n"
            "  -old line\n"
            "  +new line\n"
            "  *** End Patch"
        )

    def _patch_input(self, text: str) -> str:
        p = self.pending
        assert p is not None
        p.old_lines.append(text)

        if text.strip() != END_PATCH:
            return ""  # batch mode — result is discarded

        # End-of-patch reached — locate start of patch content and apply.
        buf = p.old_lines
        self.pending = None

        # Find the begin marker if present (custom format); otherwise use the
        # first non-blank line (unified diff has no begin marker).
        begin_idx = next(
            (i for i, line in enumerate(buf) if line.strip() == BEGIN_PATCH),
            None,
        )
        if begin_idx is None:
            # No custom-format begin marker — treat entire buffer as unified diff.
            raw = "\n".join(buf)
            if not _is_unified_diff(raw):
                return (
                    "[patch] Error: unrecognised patch format. Use standard unified diff "
                    "(diff -u / git diff) or the custom format starting with '*** Begin Patch'."
                )
        else:
            raw = "\n".join(buf[begin_idx:])

        # Auto-convert unified diff to apply_patch format if needed.
        if _is_unified_diff(raw):
            try:
                raw = _unified_diff_to_patch(raw)
            except Exception as e:
                return f"[patch] Error converting unified diff: {e}"

        try:
            result = _apply_patch(raw, safe_path=_safe_path)
        except _PatchError as e:
            return f"[patch] Error: {e}"
        except FileNotFoundError as e:
            return (
                f"[patch] Error: file/directory specified in patch header does not exist — {e}\n"
                f"[patch] Note: patch paths resolve relative to the harness working directory "
                f"({_WORK_DIR}), not the shell CWD."
            )
        return _format_patch_summary(result)

    def _dellines(self, args: list[str]) -> str:
        """
        /dellines <file> <N>      delete line N
        /dellines <file> <N>-<M>  delete lines N through M (inclusive)
        """
        if len(args) < 2:
            raise CommandError("[file] Usage: /dellines <file> <N> or <N>-<M>")
        path_str  = args[0]
        range_str = args[1]

        if "-" in range_str:
            parts = range_str.split("-", 1)
            try:
                n1, n2 = int(parts[0]), int(parts[1])
            except ValueError:
                raise CommandError(f"[file] Invalid range: {range_str!r}. Use N or N-M.")
        else:
            try:
                n1 = n2 = int(range_str)
            except ValueError:
                raise CommandError(f"[file] Invalid line number: {range_str!r}.")

        lines = _read_lines(path_str)
        total = len(lines)

        if n1 < 1 or n2 > total or n1 > n2:
            return (
                f"[file] Line range {n1}–{n2} is out of bounds "
                f"(file has {total} line(s)). Use /read to check."
            )

        del lines[n1 - 1 : n2]
        _write_lines(path_str, lines)
        count = n2 - n1 + 1
        return f"[file] Deleted {count} line(s) ({n1}–{n2}) from {path_str}. File now has {len(lines)} line(s)."

    def _del(self, args: list[str]) -> str:
        if not args:
            raise CommandError("[file] Usage: /del <file>")
        p = _safe_path(args[0])
        if not p.exists():
            return f"[file] Not found: {args[0]}"
        p.unlink()
        return f"[file] Deleted: {args[0]}"

    # -----------------------------------------------------------------------
    # Web
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Moltbook
    # -----------------------------------------------------------------------

    def _mb(self, args: list[str]) -> str:
        """Backoff wrapper around _mb_dispatch. Tracks consecutive 5xx errors."""
        now = time.time()
        if self._mb_cooloff_until > now:
            wait = int(self._mb_cooloff_until - now)
            return (
                f"[mb] Moltbook is cooling off after {_MB_ERROR_THRESHOLD} consecutive "
                f"server errors — {wait}s remaining. Use this time to write a post draft "
                f"in /cmem, check /pmem, browse the web, or check your wallet."
            )
        result = self._mb_dispatch(args)
        is_server_error = any(f"API error {c}" in result for c in ("500", "502", "503", "504"))
        if is_server_error:
            self._mb_consec_errors += 1
            if self._mb_consec_errors >= _MB_ERROR_THRESHOLD:
                self._mb_cooloff_until = time.time() + _MB_COOLOFF_SECS
                result += (
                    f"\n[mb] {_MB_ERROR_THRESHOLD} consecutive server errors — "
                    f"cooling off for {_MB_COOLOFF_SECS // 60} minutes. "
                    "Do something else and try /mb again later."
                )
        else:
            self._mb_consec_errors = 0
        return result

    def _mb_dispatch(self, args: list[str]) -> str:
        """
        /mb <subcommand> [args...]

        Subcommands:
          register <name> <description>   — register a new agent account
          me                              — show own profile
          home                            — dashboard (start here each check-in)
          feed [hot|new|top]              — browse the feed
          read <post_id>                  — read a post and its comments
          post <submolt> <title> [body]   — create a post (handles verification)
          comment <post_id> <text>        — comment on a post
          reply <post_id> <cmt_id> <txt>  — reply to a comment
          upvote <post_id>                — upvote a post
          upvote-comment <cmt_id>         — upvote a comment
          verify <code> <answer>          — complete a verification challenge
          search <query>                  — semantic search
          dm                              — check DM activity
          dm list                         — list active conversations
          dm read <conv_id>               — read a conversation (marks as read)
          dm send <conv_id> <message>     — send a DM
          dm approve <conv_id>            — approve a DM request
          dm reject <conv_id>             — reject a DM request
        """
        if not args:
            return "[mb] Usage: /mb <subcommand> — try /mb home to start."
        sub  = args[0].lower()
        rest = args[1:]

        # Simulation: intercept write actions before they hit the real API.
        # Reads (home/feed/read/search/me/dm-list/dm-read) pass through —
        # the agent should see real Moltbook state when deciding what to do.
        if self._sim is not None:
            from sim import is_mb_write
            dsub = rest[1].lower() if (sub == "dm" and len(rest) >= 2) else None
            if is_mb_write(sub, dsub):
                # For dm writes, the "args" the sim formats are the trailing
                # tokens (conv_id, message, ...) — strip the dsub off the front.
                payload = rest[1:] if dsub else rest
                return self._sim.mb_write(sub, dsub, payload)

        try:
            if sub == "register":
                if len(rest) < 2:
                    return "[mb] Usage: /mb register <name> <description>"
                return mb_mod.register(rest[0], " ".join(rest[1:]))

            if sub == "me":
                return mb_mod.me()

            if sub == "home":
                return mb_mod.home()

            if sub == "notifications" and rest and rest[0] == "clear":
                return mb_mod.clear_all_notifications()

            if sub == "feed":
                # /mb feed [sort] [submolt=<name>] [next=<cursor>] [filter=following]
                sort     = "new"
                submolt  = ""
                cursor   = ""
                filter_  = ""
                for tok in rest:
                    if tok.startswith("next="):
                        cursor = tok[5:]
                    elif tok.startswith("submolt="):
                        submolt = tok[8:]
                    elif tok.startswith("filter="):
                        filter_ = tok[7:]
                    elif tok in ("hot", "new", "top", "rising"):
                        sort = tok
                return mb_mod.feed(sort=sort, cursor=cursor, submolt=submolt, filter_=filter_)

            if sub == "read":
                if not rest:
                    return "[mb] Usage: /mb read <post_id>"
                return mb_mod.read_post(rest[0])

            if sub == "submolts":
                return mb_mod.list_submolts()

            if sub == "post":
                if len(rest) < 2:
                    return (
                        '[mb] Usage: /mb post <submolt> <title> | <body>\n'
                        '  Title-only:  /mb post m/general My Title\n'
                        '  With body:   /mb post m/general My Title | Body text here.'
                    )
                submolt = rest[0]
                title_and_body = rest[1:]
                if "|" in title_and_body:
                    pipe = title_and_body.index("|")
                    title = " ".join(title_and_body[:pipe])
                    body  = " ".join(title_and_body[pipe + 1:])
                else:
                    title = " ".join(title_and_body)
                    body  = ""
                return mb_mod.create_post(submolt, title, body)

            if sub == "comment":
                if len(rest) < 2:
                    return "[mb] Usage: /mb comment <post_id> <text>"
                return mb_mod.comment(rest[0], " ".join(rest[1:]))

            if sub == "reply":
                if len(rest) < 3:
                    return "[mb] Usage: /mb reply <post_id> <comment_id> <text>"
                return mb_mod.comment(rest[0], " ".join(rest[2:]), parent_id=rest[1])

            if sub == "upvote":
                if not rest:
                    return "[mb] Usage: /mb upvote <post_id>"
                return mb_mod.upvote(rest[0])

            if sub == "upvote-comment":
                if not rest:
                    return "[mb] Usage: /mb upvote-comment <comment_id>"
                return mb_mod.upvote_comment(rest[0])

            if sub == "follow":
                if not rest:
                    return "[mb] Usage: /mb follow <username>"
                return mb_mod.follow(rest[0])

            if sub == "unfollow":
                if not rest:
                    return "[mb] Usage: /mb unfollow <username>"
                return mb_mod.unfollow(rest[0])

            if sub == "subscribe":
                if not rest:
                    return "[mb] Usage: /mb subscribe <submolt>"
                return mb_mod.subscribe(rest[0])

            if sub == "unsubscribe":
                if not rest:
                    return "[mb] Usage: /mb unsubscribe <submolt>"
                return mb_mod.unsubscribe(rest[0])

            if sub == "verify":
                if len(rest) < 2:
                    return "[mb] Usage: /mb verify <code> <answer>"
                return mb_mod.verify(rest[0], rest[1])

            if sub == "search":
                if not rest:
                    return "[mb] Usage: /mb search <query>"
                return mb_mod.search(" ".join(rest))

            if sub == "dm":
                if not rest:
                    return mb_mod.dm_check()
                dsub = rest[0].lower()
                if dsub == "list":
                    return mb_mod.dm_list()
                if dsub == "read":
                    if len(rest) < 2:
                        return "[mb] Usage: /mb dm read <conv_id>"
                    return mb_mod.dm_read(rest[1])
                if dsub == "send":
                    if len(rest) < 3:
                        return "[mb] Usage: /mb dm send <conv_id> <message>"
                    return mb_mod.dm_send(rest[1], " ".join(rest[2:]))
                if dsub == "approve":
                    if len(rest) < 2:
                        return "[mb] Usage: /mb dm approve <conv_id>"
                    return mb_mod.dm_approve(rest[1])
                if dsub == "reject":
                    if len(rest) < 2:
                        return "[mb] Usage: /mb dm reject <conv_id>"
                    return mb_mod.dm_reject(rest[1])
                return f"[mb] Unknown dm subcommand: {dsub}"

            return f"[mb] Unknown subcommand: {sub}. Try /mb home."

        except mb_mod.MoltbookError as e:
            return f"[mb] Error: {e}"
        except Exception as e:
            return f"[mb] Unexpected error: {type(e).__name__}: {e}"

    # -----------------------------------------------------------------------
    # Telegram
    # -----------------------------------------------------------------------

    def _telegram_history(self) -> str:
        entries = self._tg.history() if self._tg else []
        if not entries:
            return "[telegram] No conversation history yet."
        recent = entries[-_TG_HISTORY_N:]
        lines = [f"[telegram] Last {len(recent)} messages:"]
        for e in recent:
            direction = e.get("direction", "in")
            sender    = e.get("from", "Foxo" if direction == "in" else "Boonie")
            text      = e.get("text", "")
            try:
                dt = datetime.fromtimestamp(e.get("ts", 0)).strftime("%d %b %H:%M")
            except Exception:
                dt = "?"
            arrow = "←" if direction == "in" else "→"
            lines.append(f"  {dt} {arrow} {sender}: {text}")
        return "\n".join(lines)

    def _telegram(self, args: list[str]) -> str:
        """
        /telegram                    — show recent conversation history
        /telegram <message>          — reply to the last sender
        /telegram @foxo <message>    — send directly to Foxo regardless of last sender
        """
        if self._tg is None:
            return "[telegram] Telegram is not enabled — start the harness with --telegram / -tg."
        if not args:
            return self._telegram_history()

        direct_foxo = args[0].lower() == "@foxo"
        if direct_foxo:
            args = args[1:]
            if not args:
                return "[telegram] Usage: /telegram @foxo <message>"

        text = " ".join(args).replace("<|eoc|>", "").strip()
        if not text:
            return "[telegram] Empty message after stripping internal tokens."
        if text == self._last_tg_text:
            return "[telegram] Duplicate — you already sent that exact message. Continue your task."
        elapsed = time.time() - self._last_tg_time
        if elapsed < _TG_COOLDOWN:
            wait = int(_TG_COOLDOWN - elapsed)
            return f"[telegram] Too soon — sent a message {int(elapsed)}s ago. Wait {wait}s, then continue your task."
        result = self._tg.send_foxo(text) if direct_foxo else self._tg.send(text)
        self._last_tg_text = text
        self._last_tg_time = time.time()
        return result

    # -----------------------------------------------------------------------
    # Monero wallet
    # -----------------------------------------------------------------------

    def _wallet(self, args: list[str]) -> str:
        if self._xmr is None:
            return "[wallet] Monero wallet is not enabled — start the harness with --monero / -xmr."
        sub = args[0].lower() if args else "help"
        if sub == "address":
            return self._xmr.address()
        if sub == "balance":
            return self._xmr.balance()
        if sub == "send":
            if len(args) < 3:
                return "[wallet] Usage: /wallet send <address> <amount_xmr>"
            return self._xmr.send(args[1], float(args[2]))
        return "[wallet] Commands: /wallet address | /wallet balance | /wallet send <addr> <xmr>"

    # -----------------------------------------------------------------------
    # Web
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Web + pagination
    # -----------------------------------------------------------------------

    def _paginate(self, content: str) -> str:
        """Split *content* into _PAGE_SIZE chunks; store all; return page 1."""
        pages: list[str] = []
        remaining = content
        while remaining:
            if len(remaining) <= _PAGE_SIZE:
                pages.append(remaining)
                break
            chunk = remaining[:_PAGE_SIZE]
            # Prefer splitting at a paragraph or line boundary
            split = chunk.rfind("\n\n")
            if split < _PAGE_SIZE // 2:
                split = chunk.rfind("\n")
            if split <= 0:
                split = _PAGE_SIZE
            pages.append(remaining[:split].rstrip())
            remaining = remaining[split:].lstrip("\n")
        self._page_buf = pages
        self._page_cur = 0
        return self._format_page(0)

    def _format_page(self, idx: int) -> str:
        n = len(self._page_buf)
        text = self._page_buf[idx]
        if n == 1:
            return text
        nav: list[str] = []
        if idx > 0:
            nav.append("/back")
        if idx < n - 1:
            nav.append("/next")
        suffix = " — " + " · ".join(nav) if nav else ""
        return text + f"\n\n[Page {idx + 1}/{n}{suffix}]"

    def _next(self) -> str:
        if not self._page_buf:
            return "[web] No page loaded — use /search or /goto first."
        if self._page_cur >= len(self._page_buf) - 1:
            return f"[web] End of content (page {len(self._page_buf)}/{len(self._page_buf)})."
        self._page_cur += 1
        return self._format_page(self._page_cur)

    def _back(self) -> str:
        if not self._page_buf:
            return "[web] No page loaded."
        if self._page_cur <= 0:
            return "[web] Already at the first page."
        self._page_cur -= 1
        return self._format_page(self._page_cur)

    def _search(self, args: list[str]) -> str:
        query = " ".join(args).strip('"\'')
        if not query:
            raise CommandError('[web] Usage: /search "<query>"')
        return self._paginate(web_mod.search(query))

    def _goto(self, args: list[str]) -> str:
        if not args:
            raise CommandError("[web] Usage: /goto <url>")
        return self._paginate(web_mod.fetch(args[0]))

    def _shell(self, cmd_str: str, root: bool = False) -> str:
        """Run a shell command (--frwx mode only). stdin closed, 30 s timeout.

        CWD persists across calls: 'cd' updates the stored directory and every
        subsequent command starts from there, just like a real interactive shell.
        """
        if not cmd_str:
            return "[shell] Empty command."

        _CWD_MARKER = "__CWD__:"
        # Append CWD capture using ';' so it runs regardless of exit code.
        cmd_with_cwd = cmd_str + f'; printf "\\n{_CWD_MARKER}%s\\n" "$(pwd)"'

        if self._chroot:
            # Run inside the jail.  The cd ensures the persistent CWD carries over
            # since we can't use cwd= across the chroot boundary.
            # Requires: <user> ALL=(root) NOPASSWD: /usr/sbin/chroot <jail> *
            inner = f"cd {shlex.quote(self._cwd)} 2>/dev/null; {cmd_with_cwd}"
            full_cmd = f"sudo chroot {shlex.quote(self._chroot)} bash -c {shlex.quote(inner)}"
            cwd_arg = "/"
        elif root:
            full_cmd = f"sudo -n bash -c {shlex.quote(cmd_with_cwd)}"
            cwd_arg = self._cwd
        else:
            full_cmd = cmd_with_cwd
            cwd_arg = self._cwd

        try:
            proc = subprocess.run(
                full_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=_SHELL_TIMEOUT,
                stdin=subprocess.DEVNULL,
                cwd=cwd_arg,
            )
            raw = proc.stdout + proc.stderr

            # Extract CWD marker and update stored directory.
            lines = raw.splitlines()
            filtered, new_cwd = [], self._cwd
            for ln in lines:
                if ln.startswith(_CWD_MARKER):
                    new_cwd = ln[len(_CWD_MARKER):]
                else:
                    filtered.append(ln)
            self._cwd = new_cwd
            out = "\n".join(filtered).strip()

            if not out:
                out = "(no output)"
            elif len(out) > _SHELL_MAX_OUT:
                omitted = len(out) - _SHELL_HEAD - _SHELL_TAIL
                out = (
                    out[:_SHELL_HEAD]
                    + f"\n[…{omitted} chars omitted…]\n"
                    + out[-_SHELL_TAIL:]
                )
            return f"[shell cwd={self._cwd}] exit={proc.returncode}\n{out}"
        except subprocess.TimeoutExpired:
            return f"[shell cwd={self._cwd}] Timeout after {_SHELL_TIMEOUT}s — command killed."
        except Exception as e:
            return f"[shell] Error: {e}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _int_arg(args: list[str], idx: int, usage: str) -> int:
    if idx >= len(args):
        raise CommandError(f"[error] Expected integer argument. Usage: {usage}")
    try:
        return int(args[idx])
    except ValueError:
        raise CommandError(f"[error] Expected integer, got '{args[idx]}'. Usage: {usage}")


_COMMAND_RE = re.compile(
    r"^/(cmem|cm|pmem|pm|pgup|pgdown|dir|read|append|appendlines|edit|patch|dellines|del|search|goto|next|back|mb|telegram|wallet|bg|fg|jobs)\b",
    re.IGNORECASE,
)

# "$ cmd" or "# cmd" — but not "## markdown header"
_SHELL_CMD_RE = re.compile(r"^(\$ \S|# [^#\s])")

# Markdown fenced code blocks. Modern instruct models default to ```bash / ```sh
# for shell commands; treat those (and bare ``` with no language) as shell
# containers. Other languages (```python, ```json, ...) stay prose for now.
_SHELL_FENCE_OPEN_RE = re.compile(r"^```\s*(bash|sh|shell)\s*$", re.IGNORECASE)
_FENCE_CLOSE_RE      = re.compile(r"^```\s*$")


def is_command_line(line: str, frwx: bool = False) -> bool:
    s = line.strip()
    if frwx and _SHELL_CMD_RE.match(s):
        return True
    return bool(_COMMAND_RE.match(s))


def is_shell_fence_open(line: str) -> bool:
    """A line that opens a shell-flavored markdown fence."""
    return bool(_SHELL_FENCE_OPEN_RE.match(line.strip()))


def is_fence_close(line: str) -> bool:
    """A line that's a bare closing fence."""
    return bool(_FENCE_CLOSE_RE.match(line.strip()))
