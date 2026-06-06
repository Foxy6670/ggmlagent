#!/usr/bin/env python3
"""
watch_furrow.py — live Furrow session monitor.

Connects to the Gateway via LAN SSH, tails the latest session log,
and formats output with colors. Filters out RAW/SSE noise.

Usage:
    python3 watch_furrow.py           # follow latest session
    python3 watch_furrow.py --list    # list available sessions
    python3 watch_furrow.py <N>       # follow Nth most recent session (1=latest)
"""

import re
import subprocess
import sys

HOST   = "furrow@stt7f7qmyesgdy4tya2sq6trqhzw5b35ihwgcnvfxwkkuw2wgmxdywqd.onion"
SSH    = ["ssh", "-o", "ProxyCommand=nc -x 127.0.0.1:9050 %h %p", HOST]
LOGDIR = "ggmlagent/furrow/logs"

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
GRAY   = "\033[90m"
RED    = "\033[31m"
BLUE   = "\033[34m"
PURPLE = "\033[35m"


def ssh(cmd: str, timeout: int = 30) -> str:
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def list_sessions() -> list[str]:
    out = ssh(f"ls -t {LOGDIR}/session_*.log 2>/dev/null")
    return [l.strip() for l in out.splitlines() if l.strip()]


def format_line(line: str) -> str | None:
    # Drop raw token stream and SSE wire noise
    if "[RAW    ]" in line:
        return None
    if "[SYS    ]" in line and "[SSE]" in line:
        return None

    # Redact credential values — replace anything that looks like KEY=value
    line = re.sub(
        r'(?i)([A-Z_]*(TOKEN|KEY|SECRET|PASSWORD|API_KEY|BOT_TOKEN)[A-Z_]*=)\S+',
        r'\1[REDACTED]',
        line,
    )
    # Redact bare credential values by known format (e.g. from echo $VAR or printenv)
    line = re.sub(r'moltbook_sk_[A-Za-z0-9]+', '[REDACTED:moltbook_key]', line)
    line = re.sub(r'\d{8,10}:[A-Za-z0-9_-]{35,}', '[REDACTED:tg_token]', line)

    # Drop bare <|eoc|> agent lines
    if "[AGENT  ]" in line and line.strip().endswith("<|eoc|>") and \
            not re.search(r"\[AGENT  \] .{5,}", line):
        return None

    m = re.match(r"^(\d{2}:\d{2}:\d{2}\.\d{3}) \[(\w+)\s*\] (.*)$", line)
    if not m:
        return DIM + line + RESET

    ts, tag, content = m.group(1), m.group(2).strip(), m.group(3)
    ts_str = GRAY + ts + RESET

    if tag == "AGENT":
        stripped = content.strip()
        if not stripped or stripped in ("```", '"""', "'''"):
            return None
        if stripped.startswith(("/", "$", "#")) and stripped != "<|eoc|>":
            return f"{ts_str} {PURPLE}{BOLD}[agent]{RESET} {BOLD}{content}{RESET}"
        return f"{ts_str} {PURPLE}[agent]{RESET} {content}"

    elif tag == "THINK":
        if content.strip() in ("<think>", "</think>", ""):
            return None
        return f"{ts_str} {YELLOW}[think]{RESET} {DIM}{content}{RESET}"

    elif tag == "CMD":
        return f"{ts_str} {BOLD}{CYAN}[ cmd ]{RESET} {BOLD}{content}{RESET}"

    elif tag == "OBS":
        if not content.strip():
            return None
        return f"{ts_str} {CYAN}[ obs ]{RESET} {content}"

    elif tag == "SYS":
        if "SESSION START" in content:
            return f"\n{BOLD}{PURPLE}{'═'*60}{RESET}\n{ts_str} {BOLD}{PURPLE}[sys  ] {content}{RESET}"
        return f"{ts_str} {GRAY}[ sys ] {content}{RESET}"

    else:
        return f"{ts_str} [{tag}] {content}"


def tail_session(log_path: str) -> None:
    print(f"{BOLD}Watching:{RESET} {GRAY}{log_path}{RESET}\n", flush=True)
    proc = subprocess.Popen(
        SSH + [f"tail -n 80 -f {log_path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    try:
        for raw in proc.stdout:
            formatted = format_line(raw.rstrip())
            if formatted is not None:
                print(formatted, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        print(f"\n{GRAY}Stopped.{RESET}")


def main() -> None:
    args = sys.argv[1:]

    if "--list" in args:
        sessions = list_sessions()
        if not sessions:
            print("No session logs found.")
            return
        for i, s in enumerate(sessions, 1):
            marker = f" {BOLD}← latest{RESET}" if i == 1 else ""
            print(f"  {GRAY}{i:2}.{RESET} {s}{marker}")
        return

    sessions = list_sessions()
    if not sessions:
        print("No session logs found on Gateway.", file=sys.stderr)
        sys.exit(1)

    idx = 0
    if args:
        try:
            idx = int(args[0]) - 1
        except ValueError:
            print(f"Usage: {sys.argv[0]} [--list] [N]", file=sys.stderr)
            sys.exit(1)

    if idx < 0 or idx >= len(sessions):
        print(f"Session {idx+1} not found. Use --list to see available sessions.",
              file=sys.stderr)
        sys.exit(1)

    tail_session(sessions[idx])


if __name__ == "__main__":
    main()
