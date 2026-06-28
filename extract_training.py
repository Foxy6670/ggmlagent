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

# Sync with agent.py _BAD_OBS_PREFIXES (tool-call era).  A few legacy entries are
# kept so this parser still cleanly filters pre-migration logs from the old
# triple-backtick corpus.
_BAD_OBS_PREFIXES = (
    # current (tool-call format)
    "[system] You wrote",
    "[system] Your /telegram message",
    "[system] Your tool call",
    "[system] Unknown tool ",
    "[system] Response completed without",
    "[system] Your response was cut off",
    "[system] Loop guard:",
    "[error] Unrecognised command:",
    # legacy (old codeblock format) — keep for back-compat parsing
    "[system] You generated a fake",
    "[system] Blank-line stall",
    "[system] You are generating blank",
    "[system] Commands use a leading slash",
    "[system] Use /telegram to send",
    "[system] You wrote what a command response",
    "[system] You wrote a long response without",
    "[system] Response completed without issuing",
    "[system] The <|eoc|>",
    "[error] Command cut off mid-token",
)


_TOOL_CALL_OPEN = "<tool_call>"


def _normalize_tool_call_text(text: str) -> str:
    """
    Rebuild assistant text as [prose]\\n<tool_call>\\n{json}\\n</tool_call>.

    Logs capture the streamed tokens, but </tool_call> is a special token that
    KCPP renders as empty in the content stream — so the reconstructed agent text
    has the opening tag and JSON but no closing tag.  Re-attach it (the JSON is
    self-delimiting).  Text without a <tool_call> (old codeblock logs) is returned
    unchanged so this parser still handles the pre-migration corpus.
    """
    idx = text.find(_TOOL_CALL_OPEN)
    if idx == -1:
        return text.rstrip()
    prose = text[:idx].rstrip()
    after = text[idx + len(_TOOL_CALL_OPEN):]
    brace = after.find("{")
    if brace == -1:
        return text.rstrip()
    try:
        _obj, end = json.JSONDecoder().raw_decode(after[brace:])
    except json.JSONDecodeError:
        return text.rstrip()
    json_str = after[brace:brace + end]
    block = f"{_TOOL_CALL_OPEN}\n{json_str}\n</tool_call>"
    return f"{prose}\n{block}" if prose else block

_LINE_RE     = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d+ \[(\w+)\s*\] (.*)")
_ABORT_RE    = re.compile(r"\[SYS\s*\] Abort sent \(genkey=[A-Z0-9]+\), (.+?)\s*$")
_MODEL_RE    = re.compile(r'"model"\s*:\s*"([^"]+)"')
_SELF_REF_RE = re.compile(r'\bthe user\b', re.IGNORECASE)

# Ordered substitutions for fixing third-person "the user ..." framing in think
# blocks.  Verb-form pairs come first so the catch-all ("the user" → "I") doesn't
# consume matches that need a different verb conjugation.
_THIRD_PERSON_SUBS = [
    (re.compile(r'\bthe user is\b',     re.IGNORECASE), 'I am'),
    (re.compile(r'\bthe user was\b',    re.IGNORECASE), 'I was'),
    (re.compile(r'\bthe user has\b',    re.IGNORECASE), 'I have'),
    (re.compile(r'\bthe user had\b',    re.IGNORECASE), 'I had'),
    (re.compile(r'\bthe user needs\b',  re.IGNORECASE), 'I need'),
    (re.compile(r'\bthe user wants\b',  re.IGNORECASE), 'I want'),
    (re.compile(r'\bthe user should\b', re.IGNORECASE), 'I should'),
    (re.compile(r'\bthe user will\b',   re.IGNORECASE), 'I will'),
    (re.compile(r'\bthe user can\b',    re.IGNORECASE), 'I can'),
    (re.compile(r'\bthe user might\b',  re.IGNORECASE), 'I might'),
    (re.compile(r'\bthe user could\b',  re.IGNORECASE), 'I could'),
    (re.compile(r'\bthe user would\b',  re.IGNORECASE), 'I would'),
    (re.compile(r'\bthe user\'s\b',     re.IGNORECASE), 'my'),
    (re.compile(r'\bthe user\b',        re.IGNORECASE), 'I'),
]


def _fix_third_person(text: str) -> str:
    """Rewrite third-person 'the user ...' framing to first-person 'I ...'."""
    for pattern, replacement in _THIRD_PERSON_SUBS:
        text = pattern.sub(replacement, text)
    # Re-capitalise 'my' / 'i' at sentence boundaries (after '.', '!', '?', or
    # at the very start of the block).
    text = re.sub(r'(?:^|(?<=[.!?])\s+)(i)\b', lambda m: m.group(0)[:-1] + 'I', text)
    text = re.sub(r'(?:^|(?<=[.!?])\s+)(my)\b', lambda m: m.group(0)[:-2] + 'My', text)
    return text


# Third-person verbs that, immediately after the agent's own name, mark it being
# treated as an external subject ("Boonie is running ...", "Boonie tried ...").
_DISSOC_VERBS = (
    r"is|was|are|were|has|had|does|did|will|won't|would|can|can't|could|should|"
    r"must|needs?|wants?|tried|tries|keeps?|seems?|appears?|ran|runs?|began|"
    r"started|stopped|attempted|noticed|realized|realised|decided|wrote|made|"
    r"set|gets?|got|said|thinks?|thought"
)

_DISSOC_RE_CACHE: dict[str, list] = {}


def _dissoc_patterns(name: str):
    """Build (and cache) the dissociation regexes for an agent *name*."""
    if name not in _DISSOC_RE_CACHE:
        n = re.escape(name)
        _DISSOC_RE_CACHE[name] = [
            # 1. Own name as the subject of a third-person verb.  First-person and
            #    appositive uses ("I am Boonie", "as Boonie", "named Boonie") don't
            #    put a finite verb right after the name, so they don't match.
            ("name-as-subject",
             re.compile(r"\b" + n + r"\s+(?:" + _DISSOC_VERBS + r")\b", re.IGNORECASE)),
            # 2. Explicit self-as-user apposition: "the user, Boonie" / "Boonie, the user".
            ("user-apposition",
             re.compile(r"\bthe user,?\s+" + n + r"\b|\b" + n + r",?\s+the user\b",
                        re.IGNORECASE)),
        ]
    return _DISSOC_RE_CACHE[name]


# A single "the user" is rewritten to "I" by _fix_third_person; this many or more
# means the whole block is framed as narrating someone else ("the user is trying
# to... they tried...") — the dominant live drift form, which rewriting only
# mangles ("I is trying..."), so dense blocks are dropped instead.
_USER_DENSITY = 2


def _dissociation_signals(think: str, name: str) -> list[str]:
    """Markers that the agent is narrating itself in the third person.

    These are self-references _fix_third_person can't safely repair — the agent
    talking about itself by name as an external subject, casting itself as 'the
    user', or narrating the whole block in dense third-person 'the user' framing.
    Such turns teach spectator framing, so they're dropped rather than
    cosmetically rewritten.  Identity-affirming uses ("I am Boonie", "my task as
    Boonie") and a single clean "the user" (left to _fix_third_person) are NOT
    flagged.
    """
    hits = [label for label, pat in _dissoc_patterns(name) if pat.search(think)]
    if len(_SELF_REF_RE.findall(think)) >= _USER_DENSITY:
        hits.append("user-density")
    return hits


def _is_dissociated(think: str, name: str) -> bool:
    return bool(_dissociation_signals(think, name))


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
            # A command marks the start of a result block — flush previous obs.
            # The command itself is NOT echoed into the result: it already lives
            # in the assistant turn's <tool_call> block (reconstructed from AGENT
            # lines), and the result is injected back as a bare role:"tool".
            _flush_obs()
            in_obs_block = True
            obs_lines = []

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
    return False


def _build_messages(turns: list[Turn], system_prompt: str, agent_name: str = "Boonie") -> list[dict]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": "Begin. Read your task file first."},
    ]
    for turn in turns:
        if _is_bad(turn) or not (turn.agent.strip() or turn.think.strip()):
            continue
        if _is_dissociated(turn.think, agent_name):
            continue
        think = _fix_third_person(turn.think.strip())
        # Re-attach the </tool_call> the stream drops, so the assistant turn is a
        # complete native tool call (no-op for old codeblock logs).
        agent = _normalize_tool_call_text(turn.agent)
        content = f"<think>\n{think}\n</think>\n{agent}" if think else agent
        if not content.strip():
            continue
        messages.append({"role": "assistant", "content": content})
        # Command results inject back as role:"tool" (Qwen3 wraps them in
        # <tool_response>); the dispatched command is in the assistant turn above.
        for obs in turn.observations:
            messages.append({"role": "tool", "content": obs})
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
    parser.add_argument(
        "--agent-name", default="Boonie",
        help="agent's own name, used to drop turns that narrate it in the third "
             "person ('Boonie is...', 'the user, Boonie') (default: Boonie)",
    )
    parser.add_argument(
        "--dropped-out", default=None,
        help="quarantine file: write every dissociation-dropped turn here (with "
             "session + which signal fired) instead of discarding it, so useful "
             "content in the dropped ~5%% can be repaired and reincorporated later",
    )
    args = parser.parse_args()

    if args.system:
        system_prompt = Path(args.system).read_text()
    else:
        sys.path.insert(0, str(Path(__file__).parent))
        from config import SYSTEM_PROMPT
        system_prompt = SYSTEM_PROMPT

    out = open(args.out, "w", encoding="utf-8") if args.out != "-" else sys.stdout
    dropped_out = open(args.dropped_out, "w", encoding="utf-8") if args.dropped_out else None

    total_sessions = 0
    total_turns = 0
    total_dissoc = 0
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
        present = [t for t in turns if not _is_bad(t) and (t.agent.strip() or t.think.strip())]
        dissoc = [t for t in present if _is_dissociated(t.think, args.agent_name)]
        good = [t for t in present if t not in dissoc]
        if len(good) < args.min_turns:
            print(f"  skip {p.name} ({len(good)} good turns)", file=sys.stderr)
            continue
        model = args.model_tag or _extract_model(p)
        if dropped_out:
            for t in dissoc:
                dropped_out.write(json.dumps({
                    "session": p.name,
                    "model": model,
                    "signals": _dissociation_signals(t.think, args.agent_name),
                    "think": t.think,
                    "agent": t.agent,
                    "observations": t.observations,
                }, ensure_ascii=False) + "\n")
        messages = _build_messages(turns, system_prompt, args.agent_name)
        out.write(json.dumps({"model": model, "messages": messages}, ensure_ascii=False) + "\n")
        total_sessions += 1
        total_turns += len(good)
        total_dissoc += len(dissoc)
        ratio_str = f", exec {exec_n}/{total_aborts}" if total_aborts else ""
        dissoc_str = f", {len(dissoc)} dissociated dropped" if dissoc else ""
        print(f"  {p.name} [{model}]: {len(good)} good turns{ratio_str}{dissoc_str}", file=sys.stderr)

    if args.out != "-":
        out.close()
    if dropped_out:
        dropped_out.close()
    dissoc_total_str = f" ({total_dissoc} dissociated turns dropped)" if total_dissoc else ""
    print(f"\nTotal: {total_sessions} sessions, {total_turns} turns{dissoc_total_str} → {args.out}",
          file=sys.stderr)
    if args.dropped_out and total_dissoc:
        print(f"Quarantined {total_dissoc} dropped turns → {args.dropped_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
