"""Command loop detection — ported from openclaw/src/agents/tool-loop-detection.ts."""

from collections import deque
import hashlib


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class CommandLoopDetector:
    """Detects when the agent is stuck in a repetitive command loop.

    Ported from OpenClaw's ToolLoopDetector. Tracks the last N commands
    and their results to catch four failure modes:
      - generic_repeat: same command fired REPEAT_WARN+ times in a row
      - poll_no_progress: same command+result (i.e. nothing changed) POLL_WARN+ times
      - ping_pong: alternating between two commands with no progress
      - global_circuit_breaker: GLOBAL_LIMIT total commands this session
    """

    HISTORY_SIZE = 20
    REPEAT_WARN = 3    # consecutive same command → warn
    REPEAT_STOP = 6    # consecutive same command → hard stop
    POLL_WARN = 2      # same command+result → warn
    POLL_STOP = 4      # same command+result → hard stop
    PING_PONG_WARN = 4 # A/B alternation cycles → warn
    PING_PONG_STOP = 8 # A/B alternation cycles → hard stop
    GLOBAL_LIMIT = 150 # fires every GLOBAL_LIMIT commands as a sanity check

    def __init__(self) -> None:
        # Each entry: (cmd_key, result_hash)
        self._history: deque[tuple[str, str]] = deque(maxlen=self.HISTORY_SIZE)
        self._total = 0

    def _cmd_key(self, cmd: str) -> str:
        """Normalize command to a stable key (strip whitespace, lowercase prefix)."""
        return cmd.strip()

    def record(self, cmd: str, result: str) -> str | None:
        """Record a command+result pair. Returns a warning string if a loop is detected."""
        key = self._cmd_key(cmd)
        rhash = _hash(result)
        self._history.append((key, rhash))
        self._total += 1

        warning = (
            self._check_global_circuit_breaker()
            or self._check_poll_no_progress(key, rhash)
            or self._check_generic_repeat(key)
            or self._check_ping_pong()
        )
        return warning

    # ── detectors ──────────────────────────────────────────────────────────

    def _check_global_circuit_breaker(self) -> str | None:
        # Fire only at exact multiples so it triggers once per milestone,
        # not on every command after the first threshold is crossed.
        if self._total > 0 and self._total % self.GLOBAL_LIMIT == 0:
            return (
                f"[system] Loop guard: {self._total} commands issued this session. "
                "If you are stuck, review your task and cmem, then try a different "
                "approach or use /telegram to ask Foxo for direction."
            )
        return None

    def _check_generic_repeat(self, key: str) -> str | None:
        """Consecutive identical commands (regardless of result)."""
        history = list(self._history)
        run = 0
        for k, _ in reversed(history):
            if k == key:
                run += 1
            else:
                break
        if run >= self.REPEAT_STOP:
            return (
                f"[system] Loop guard: you have issued `{key}` {run} times in a row. "
                "Stop repeating this command — try a different approach."
            )
        if run >= self.REPEAT_WARN:
            return (
                f"[system] Loop guard: `{key}` repeated {run} consecutive times. "
                "If you are waiting for a result to change, try a different command first."
            )
        return None

    def _check_poll_no_progress(self, key: str, rhash: str) -> str | None:
        """Same command AND same result — nothing is changing."""
        history = list(self._history)
        run = 0
        for k, r in reversed(history):
            if k == key and r == rhash:
                run += 1
            else:
                break
        if run >= self.POLL_STOP:
            return (
                f"[system] Loop guard: `{key}` returned the same result {run} times. "
                "The state is not changing. Take a different action."
            )
        if run >= self.POLL_WARN:
            return (
                f"[system] Loop guard: `{key}` returned the same result {run} times in a row. "
                "Consider whether waiting longer will help."
            )
        return None

    def _check_ping_pong(self) -> str | None:
        """Alternating A/B/A/B pattern with no progress."""
        history = list(self._history)
        if len(history) < 4:
            return None

        keys = [k for k, _ in history]

        # Detect strict A/B alternation in the last N entries.
        # Need at least 4 entries to see A/B/A/B.
        n = len(keys)
        # Walk backwards and count how long the A/B pattern holds.
        if n < 4:
            return None

        a, b = keys[-1], keys[-2]
        if a == b:
            return None  # not alternating, generic_repeat will catch it

        cycles = 1  # we already have one A/B pair
        # From the end: ..., a, b, a, b (a is newest).
        # Pair at offset n-3, n-4 should be a, b again.
        i = n - 3
        while i >= 1:
            if keys[i] == a and keys[i - 1] == b:
                cycles += 1
                i -= 2
            else:
                break

        if cycles >= self.PING_PONG_STOP:
            return (
                f"[system] Loop guard: alternating between `{a}` and `{b}` for "
                f"{cycles} cycles — no progress. Break the pattern."
            )
        if cycles >= self.PING_PONG_WARN:
            return (
                f"[system] Loop guard: ping-pong detected between `{a}` and `{b}` "
                f"({cycles} cycles). Make sure each step is actually making progress."
            )
        return None
