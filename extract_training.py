#!/usr/bin/env python3
"""
Extract training data from raw session logs.

Parses [THINK], [AGENT], [CMD], [OBS] lines from *.log files and
reconstructs OpenAI-format JSONL, filtering out correction turns.

Usage:
  python3 extract_training.py moltbot/logs/session_*.log
  python3 extract_training.py moltbot/logs/session_*.log --out dataset.jsonl
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Sync with agent.py _BAD_OBS_PREFIXES
_BAD_OBS_PREFIXES = (
    "[system] You generated a fake",
    "[system] Your /telegram message",
    "[system] Blank-line stall",
    "[system] You are generating blank",
    "[system] Commands use a leading slash",
    "[system] Use /telegram to send",
    "[system] You wrote what a command response",
    "[system] You wrote a long response without",
    "[system] Response completed without issuing",
    "[system] Your response was cut off by the token limit",
    "[system] Loop guard:",
    "[system] The <|eoc|>",
    "[error] Unrecognised command:",
    "[error] Command cut off mid-token",  # legacy — pre-2026-05-03 wording
)

_LINE_RE     = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d+ \[(\w+)\s*\] (.*)")
_ABORT_RE    = re.compile(r"\[SYS\s*\] Abort sent \(genkey=[A-Z0-9]+\), (.+?)\s*$")
_MODEL_RE    = re.compile(r'"model"\s*:\s*"([^"]+)"')
_SELF_REF_RE = re.compile(r'\bthe user\b', re.IGNORECASE)


def _session_health(path: Path) -> tuple[int, int]:
    """Return (executing_aborts, total_aborts) for a session log.

    A session whose aborts are dominated by 'prose monologue', 'hallucinated
    <|eoc|>', or 'observation-echo' is one where the model has lost the
    plot — its turns are not safe training data even if individual lines
    don't trip _BAD_OBS_PREFIXES. The ratio between these two numbers is
    the cheapest health signal we have.
    """
    exec_n = total = 0
    for raw in path.read_text(errors="replace").splitlines():
        m = _ABORT_RE.search(raw)
        if not m:
            continue
        total += 1
        if "executing command" in m.group(1):
            exec_n += 1
    return exec_n, total


def _extract_model(path: Path) -> str:
    """Scan the first ~200 SSE lines of a log for the model name."""
    scanned = 0
    for raw in path.read_text(errors="replace").splitlines():
        if "[SSE]" not in raw:
            continue
        m = _MODEL_RE.search(raw)
        if m:
            return m.group(1)
        scanned += 1
        if scanned > 200:
            break
    return "unknown"


@dataclass
class Turn:
    think: str = ""
    agent: str = ""
    observations: list[str] = field(default_factory=list)


def _parse_log(path: Path) -> list[Turn]:
    turns: list[Turn] = []
    current = Turn()
    in_think = False
    in_obs_block = False
    obs_lines: list[str] = []

    def _flush_obs():
        nonlocal obs_lines, in_obs_block
        if obs_lines:
            current.observations.append("\n".join(obs_lines).strip())
            obs_lines = []
        in_obs_block = False

    def _commit():
        nonlocal current
        if current.agent.strip() or current.observations:
            turns.append(current)
        current = Turn()

    for raw in path.read_text(errors="replace").splitlines():
        m = _LINE_RE.match(raw)
        if not m:
            # Continuation of a multi-line OBS block
            if in_obs_block:
                obs_lines.append(raw)
            continue

        tag, content = m.group(1).upper(), m.group(2)

        if tag == "THINK":
            if content.strip() in ("<think>", "</think>"):
                in_think = content.strip() == "<think>"
                continue
            if in_think:
                current.think += content + "\n"

        elif tag == "AGENT":
            current.agent += content + "\n"

        elif tag == "CMD":
            # A command starts a new turn boundary — flush previous obs
            _flush_obs()
            # CMD line is recorded as the first obs line of the new obs block
            in_obs_block = True
            obs_lines = [f"> {content}"]

        elif tag == "OBS":
            if not in_obs_block:
                in_obs_block = True
                obs_lines = []
            obs_lines.append(content)

        elif tag in ("SYS", "RAW"):
            # SYS lines that aren't generation markers mean a new turn starts
            if "Generation started" in content:
                _flush_obs()
                _commit()

    _flush_obs()
    _commit()
    return turns


def _is_bad(turn: Turn) -> bool:
    for obs in turn.observations:
        for prefix in _BAD_OBS_PREFIXES:
            if prefix.lower() in obs.lower():
                return True
    if _SELF_REF_RE.search(turn.think):
        return True
    return False


def _build_messages(turns: list[Turn], system_prompt: str) -> list[dict]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": "Begin. Read your task file first."},
    ]
    for turn in turns:
        if _is_bad(turn) or not (turn.agent.strip() or turn.think.strip()):
            continue
        think = turn.think.strip()
        agent = turn.agent.strip()
        content = f"<think>\n{think}\n</think>\n{agent}" if think else agent
        if not content.strip():
            continue
        messages.append({"role": "assistant", "content": content})
        for obs in turn.observations:
            messages.append({"role": "user", "content": obs})
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="session log files")
    parser.add_argument("--out", default="-", help="output JSONL file (default: stdout)")
    parser.add_argument(
        "--system",
        default=None,
        help="path to a file containing the system prompt (default: import from config.py)",
    )
    parser.add_argument(
        "--min-turns", type=int, default=3,
        help="minimum good turns to include a session (default: 3)",
    )
    parser.add_argument(
        "--min-exec-ratio", type=float, default=0.5,
        help="drop sessions where executing-command aborts / total aborts < this "
             "ratio (default: 0.5). Catches sessions where the model lost command "
             "discipline even if individual turns look OK.",
    )
    parser.add_argument(
        "--exclude", action="append", default=[],
        help="skip sessions whose filename contains this substring; repeatable",
    )
    parser.add_argument(
        "--model-tag", default=None,
        help="override model name in output (default: auto-detect from log SSE lines)",
    )
    args = parser.parse_args()

    if args.system:
        system_prompt = Path(args.system).read_text()
    else:
        sys.path.insert(0, str(Path(__file__).parent))
        from config import SYSTEM_PROMPT
        system_prompt = SYSTEM_PROMPT

    out = open(args.out, "w", encoding="utf-8") if args.out != "-" else sys.stdout

    total_sessions = 0
    total_turns = 0
    for log_path in sorted(args.logs):
        p = Path(log_path)
        if not p.exists() or p.suffix != ".log":
            continue

        if any(pat in p.name for pat in args.exclude):
            print(f"  skip {p.name} (excluded)", file=sys.stderr)
            continue

        exec_n, total_aborts = _session_health(p)
        if total_aborts > 0:
            ratio = exec_n / total_aborts
            if ratio < args.min_exec_ratio:
                print(
                    f"  skip {p.name} (exec ratio {exec_n}/{total_aborts}="
                    f"{ratio:.0%} < {args.min_exec_ratio:.0%})",
                    file=sys.stderr,
                )
                continue

        turns = _parse_log(p)
        good = [t for t in turns if not _is_bad(t) and (t.agent.strip() or t.think.strip())]
        if len(good) < args.min_turns:
            print(f"  skip {p.name} ({len(good)} good turns)", file=sys.stderr)
            continue
        model = args.model_tag or _extract_model(p)
        messages = _build_messages(turns, system_prompt)
        out.write(json.dumps({"model": model, "messages": messages}, ensure_ascii=False) + "\n")
        total_sessions += 1
        total_turns += len(good)
        ratio_str = f", exec {exec_n}/{total_aborts}" if total_aborts else ""
        print(f"  {p.name} [{model}]: {len(good)} good turns{ratio_str}", file=sys.stderr)

    if args.out != "-":
        out.close()
    print(f"\nTotal: {total_sessions} sessions, {total_turns} turns → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
