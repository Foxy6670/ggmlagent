#!/usr/bin/env python3
"""Entry point for the KoboldCPP agent harness."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="KoboldCPP agent harness")
    parser.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help="Directory the agent reads/writes files in (default: current directory)",
    )
    parser.add_argument(
        "--teleop",
        action="store_true",
        help="Teleoperation mode — you type commands, the harness executes them and logs training data",
    )
    parser.add_argument(
        "--frwx",
        action="store_true",
        help="Full read/write/execute access — enables $ and # shell commands and unrestricted file paths",
    )
    parser.add_argument(
        "--telegram", "-tg",
        action="store_true",
        help="Enable Telegram integration (requires TELEGRAM_BOT_TOKEN and requests[socks])",
    )
    parser.add_argument(
        "--monero", "-xmr",
        action="store_true",
        help="Enable Monero wallet (starts monero-wallet-rpc; requires monero package)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Dry-run mode — Moltbook writes and Telegram (in/out) are intercepted "
             "and synthesized; reads, files, and web stay real. Auto-on with --teleop.",
    )
    args = parser.parse_args()
    # Teleop implies simulate so dry-run scenarios don't commit real social actions.
    simulate = args.simulate or args.teleop

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists():
        workspace.mkdir(parents=True)
        print(f"Created workspace: {workspace}")
    elif not workspace.is_dir():
        print(f"Error: {workspace} is not a directory", file=sys.stderr)
        sys.exit(1)

    os.chdir(workspace)
    print(f"Workspace: {workspace}")

    tg_proc = None
    if args.telegram and not simulate:
        poll_script = Path(__file__).parent / "telegram_poll.py"
        tg_proc = subprocess.Popen(
            [sys.executable, str(poll_script)],
            # Don't pipe stdout — an unread pipe buffer fills up and blocks the poller
            stdout=None,
            stderr=None,
        )
        print(f"[main] telegram_poll started (pid={tg_proc.pid})")
    elif simulate:
        print("[main] Simulation mode — Telegram + Moltbook writes are synthetic.")
        if args.telegram:
            print("[main]   (--telegram ignored under --simulate; no poller started)")
    else:
        print("[main] Telegram disabled (pass --telegram / -tg to enable)")

    xmr_proc = None
    if args.monero:
        xmr_script = Path(__file__).parent / "monero_start.sh"
        try:
            xmr_proc = subprocess.Popen(
                ["bash", str(xmr_script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[main] monero-wallet-rpc started (pid={xmr_proc.pid})")
        except FileNotFoundError:
            print("[main] monero not installed — skipping wallet (apt install monero)")
    else:
        print("[main] Monero disabled (pass --monero / -xmr to enable)")

    # Import after chdir so all relative paths resolve correctly
    from agent import Agent
    try:
        agent = Agent(
            frwx=args.frwx,
            telegram=args.telegram,
            monero=args.monero,
            simulate=simulate,
        )
        if args.teleop:
            agent.run_teleop()
        else:
            agent.run()
    finally:
        if tg_proc:
            tg_proc.terminate()
            tg_proc.wait()
            print("[main] telegram_poll stopped")
        if xmr_proc:
            xmr_proc.terminate()
            xmr_proc.wait()
            print("[main] monero-wallet-rpc stopped")


if __name__ == "__main__":
    main()
