"""
Session logger.

Creates a timestamped log file in logs/ for each run.
Every event is written verbosely with a timestamp and type tag.
Streaming tokens are buffered per-line for readability, but each
individual token is also written raw so nothing is lost.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path("logs")

# Tag widths padded for alignment
_TAGS = {
    "AGENT":   "[AGENT  ]",
    "THINK":   "[THINK  ]",
    "CMD":     "[CMD    ]",
    "OBS":     "[OBS    ]",
    "PENDING": "[PENDING]",
    "SYS":     "[SYS    ]",
    "RAW":     "[RAW    ]",
}


class SessionLogger:
    """
    One instance per agent run.  Thread-safety not required — single-threaded.
    """

    def __init__(self):
        _LOG_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._path = _LOG_DIR / f"session_{ts}.log"
        self._fh = self._path.open("w", encoding="utf-8", buffering=1)  # line-buffered
        self._token_buf: dict[str, str] = {}  # kind -> accumulated line so far
        self._write_header(ts)
        print(f"\033[2m[log] Writing to {self._path}\033[0m", file=sys.stderr)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def token(self, text: str, kind: str = "AGENT") -> None:
        """
        Log a streaming token.  Buffers until a newline is seen, then
        flushes the completed line with a timestamp.  Each raw token is
        also appended to a RAW line for full fidelity.
        """
        # Raw token log (every token, no newlines substituted)
        self._raw(f"<{kind}>{repr(text)}")

        buf = self._token_buf.get(kind, "")
        buf += text

        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            self._log(kind, line)

        self._token_buf[kind] = buf

    def flush_token_buf(self, kind: str = "AGENT") -> None:
        """Flush any incomplete buffered line (call at end of generation)."""
        buf = self._token_buf.pop(kind, "")
        if buf.strip():
            self._log(kind, buf)

    def command(self, line: str) -> None:
        self.flush_token_buf("AGENT")
        self._log("CMD", line)

    def observation(self, text: str) -> None:
        # Observations can be multi-line; log each line separately
        for line in text.splitlines():
            self._log("OBS", line)

    def pending_input(self, text: str) -> None:
        self.flush_token_buf("AGENT")
        self._log("PENDING", text)

    def system(self, text: str) -> None:
        self._log("SYS", text)

    def close(self) -> None:
        for kind in list(self._token_buf):
            self.flush_token_buf(kind)
        self._log("SYS", "=== SESSION END ===")
        self._fh.close()

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_header(self, ts: str) -> None:
        self._fh.write(f"=== KoboldCPP Agent Session — {ts} ===\n\n")

    def _log(self, kind: str, text: str) -> None:
        tag = _TAGS.get(kind, f"[{kind:<7}]")
        ts  = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._fh.write(f"{ts} {tag} {text}\n")

    def _raw(self, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._fh.write(f"{ts} {_TAGS['RAW']} {text}\n")
