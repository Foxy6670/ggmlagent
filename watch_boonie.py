#!/usr/bin/env python3
"""
watch_boonie.py — live Boonie session monitor.

Connects to MQ-Pro and tails the latest session log, formatting output with
colors and filtering RAW/SSE noise. Defaults to the Tor onion hidden service
(works from anywhere); pass --lan to go direct over the home LAN when you're
on the same network.

Note: the MQ-Pro's direct LAN is relatively weak and its internet egress
routes via RNDIS through the TUF — so --lan (direct to mangopi.lan) is a
useful fallback when the onion path degrades (e.g. the TUF is busy).

Usage:
    python3 watch_boonie.py           # follow latest session (over Tor onion)
    python3 watch_boonie.py --lan     # follow latest session (direct LAN, no Tor)
    python3 watch_boonie.py --jump    # via TUF jump host → MQ-Pro RNDIS (10.55.0.1)
    python3 watch_boonie.py --list    # list available sessions
    python3 watch_boonie.py <N>       # follow Nth most recent session (1=latest)
    (--lan / --jump combine with --list / <N>.)
"""

import re
import subprocess
import sys

from config import BOONIE_ONION

# Default route: Tor onion hidden service — reachable from anywhere.
ONION     = BOONIE_ONION
SSH_TOR   = ["ssh", "-o", "ProxyCommand=nc -x 127.0.0.1:9050 %h %p", ONION]
# Home-LAN route: direct, no Tor — used only when --lan is passed.
LAN_HOST  = "boonie@192.168.18.63"   # update when avahi resolves mangopi.local again
SSH_LAN   = ["ssh", LAN_HOST]
# Jump route: TUF as ProxyJump → MQ-Pro via RNDIS link (10.55.0.1).
# Useful when avahi/DNS isn't resolving but the USB-ethernet link is up.
JUMP_HOST = "boonie@10.55.0.1"
SSH_JUMP  = ["ssh", "-J", "foxo-tuff17.local", JUMP_HOST]
# Active route; main() flips this when --lan or --jump is given.
SSH       = SSH_TOR
LOGDIR   = "ggmlagent/boonie/logs"

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
GRAY   = "\033[90m"
RED    = "\033[31m"
BLUE   = "\033[34m"


class Unreachable(Exception):
    """The SSH/Tor connection itself failed — distinct from a successful
    command that simply returned no output. Lets callers tell 'host is down /
    onion not yet published' apart from 'connected, but no logs exist'."""


def ssh(cmd: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise Unreachable(f"connection timed out after {timeout}s")
    # ssh reserves exit code 255 for its own failures (connection refused/timeout,
    # DNS, and ProxyCommand/Tor errors) — as opposed to the remote command's own
    # non-zero exit (e.g. ls finding no matching files, which returns 1/2). So 255
    # means we never reached the host; anything else means the command ran.
    if r.returncode == 255:
        err = [l for l in r.stderr.strip().splitlines() if l.strip()]
        raise Unreachable(err[-1] if err else "ssh connection failed")
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

    m = re.match(r"^(\d{2}:\d{2}:\d{2}\.\d{3}) \[(\w+)\s*\] ?(.*)$", line)
    if not m:
        return DIM + line + RESET

    ts, tag, content = m.group(1), m.group(2).strip(), m.group(3)
    ts_str = GRAY + ts + RESET

    if tag == "AGENT":
        stripped = content.strip()
        if not stripped or stripped in ("```", '"""', "'''"):
            return None
        # Highlight command lines inside blocks
        if stripped.startswith(("/", "$", "#")) and stripped != "<|eoc|>":
            return f"{ts_str} {GREEN}{BOLD}[agent]{RESET} {BOLD}{content}{RESET}"
        return f"{ts_str} {GREEN}[agent]{RESET} {content}"

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
        # Highlight session boundaries
        if "SESSION START" in content:
            return f"\n{BOLD}{BLUE}{'═'*60}{RESET}\n{ts_str} {BOLD}{BLUE}[sys  ] {content}{RESET}"
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
    global SSH
    args = sys.argv[1:]

    use_lan  = "--lan"  in args
    use_jump = "--jump" in args
    if use_lan and use_jump:
        print("Error: --lan and --jump are mutually exclusive.", file=sys.stderr)
        sys.exit(1)
    if use_lan:
        args = [a for a in args if a != "--lan"]
        SSH = SSH_LAN
        route = "LAN"
    elif use_jump:
        args = [a for a in args if a != "--jump"]
        SSH = SSH_JUMP
        route = "TUF jump → RNDIS"
    else:
        route = "Tor onion"

    try:
        sessions = list_sessions()
    except Unreachable as e:
        print(f"{RED}[unreachable]{RESET} MQ-Pro not reachable over {route}: {e}",
              file=sys.stderr)
        if use_lan:
            print(f"{DIM}Is the MQ-Pro up and on 192.168.18.63? "
                  f"Try --jump to reach it via TUF's RNDIS link instead.{RESET}", file=sys.stderr)
        elif use_jump:
            print(f"{DIM}Is foxo-tuff17.local reachable and the RNDIS link (10.55.0.1) up? "
                  f"Try --lan or drop both flags for Tor onion.{RESET}", file=sys.stderr)
        else:
            print(f"{DIM}After a reboot the hidden-service descriptor can take a few "
                  f"minutes to republish — retry shortly, or use --jump if you're home.{RESET}",
                  file=sys.stderr)
        sys.exit(2)

    if "--list" in args:
        if not sessions:
            print("No session logs found.")
            return
        for i, s in enumerate(sessions, 1):
            marker = f" {BOLD}← latest{RESET}" if i == 1 else ""
            print(f"  {GRAY}{i:2}.{RESET} {s}{marker}")
        return

    if not sessions:
        print("No session logs found on MQ-Pro.", file=sys.stderr)
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
