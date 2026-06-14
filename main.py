#!/usr/bin/env python3
"""Entry point for the KoboldCPP agent harness."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Set USE_TOR before any module-level config imports read it.
if "--tor" in sys.argv:
    os.environ["USE_TOR"] = "1"


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
        "--chroot",
        default="",
        metavar="JAIL_PATH",
        help="Chroot jail for shell dispatch — $ and # commands run inside the jail, "
             "keeping .secrets and harness files unreachable. Requires a sudoers entry: "
             "  <user> ALL=(root) NOPASSWD: /usr/sbin/chroot <JAIL_PATH> *",
    )
    parser.add_argument(
        "--tor",
        action="store_true",
        help="Route Telegram API calls through Tor SOCKS5 proxy (default: clearnet)",
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
        xmr_log_path = workspace / "wallet_rpc.log"
        try:
            xmr_log = open(xmr_log_path, "w")
            xmr_proc = subprocess.Popen(
                ["bash", str(xmr_script)],
                stdout=xmr_log,
                stderr=subprocess.STDOUT,
            )
            # Verify the daemon actually stays alive — previously we printed
            # "started" even when the script crashed within milliseconds,
            # leaving the agent with broken wallet state and no diagnostic.
            time.sleep(2)
            if xmr_proc.poll() is not None:
                exit_code = xmr_proc.returncode
                tail = ""
                try:
                    with open(xmr_log_path) as f:
                        tail = f.read()[-800:]
                except OSError:
                    pass
                print(
                    f"[main] monero-wallet-rpc exited immediately "
                    f"(exit={exit_code}, log: {xmr_log_path})",
                    file=sys.stderr,
                )
                if tail.strip():
                    print(f"[main] last lines:\n{tail}", file=sys.stderr)
                xmr_proc = None
            else:
                print(
                    f"[main] monero-wallet-rpc started "
                    f"(pid={xmr_proc.pid}, log: {xmr_log_path})"
                )
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
            chroot=args.chroot,
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
