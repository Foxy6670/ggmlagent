# ggmlagent

An autonomous agent harness built on [KoboldCPP](https://github.com/LostRuins/koboldcpp). The agent reads a task file, executes commands, browses the web, manages its own memory, and communicates over Telegram — running continuously on a dedicated machine.

---

## Hardware philosophy

**Do not run this on a daily driver.**

The agent runs in an infinite loop, consuming CPU, memory, and network continuously. It has shell access and can execute arbitrary commands as your user (and as root, with `--frwx`). It is designed to run on a dedicated, non-essential machine that you are comfortable handing over.

Good candidates: a single-board computer (Raspberry Pi, MangoPi MQ-Pro, etc.), an old laptop, a cheap VPS, or any machine you can wipe without losing sleep. The hardware should be appropriate for the workload — something that can handle the task you assign without becoming a liability if the agent misbehaves.

The inference backend (KoboldCPP + the model) runs separately, typically on a more powerful machine, and the agent connects to it over the network.

---

## Requirements

- Python 3.10+
- `pip install requests[socks]` (SOCKS5 support for Tor routing)
- A running [KoboldCPP](https://github.com/LostRuins/koboldcpp) instance with a chat-capable model loaded
- (Optional) Tor, for routing KCPP connections to a remote `.onion` endpoint
- (Optional) Telegram bot token + chat ID, for bidirectional messaging
- (Optional) Moltbook API key, for social network integration
- (Optional) `monero-wallet-rpc`, for the Monero wallet integration

---

## Setup

1. Copy `.secrets.example` to `.secrets` and fill in your values:

```
KCPP_BASE_URL=http://your-kcpp-host:5001   # or a .onion address
MOLTBOOK_API_KEY=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
MONERO_DAEMON=                              # optional; defaults to a public node
MONERO_PROXY=                              # optional; set to 127.0.0.1:9050 for Tor
```

2. Create a workspace directory for the agent (e.g. `mkdir myagent`). Put a `task.md` in it describing what the agent should do.

3. Run:

```bash
python3 main.py myagent/
```

Common flags:

| Flag | Effect |
|------|--------|
| `--telegram` / `-tg` | Enable Telegram integration (starts `telegram_poll.py` as a subprocess) |
| `--monero` / `-xmr` | Start `monero-wallet-rpc` and expose wallet commands |
| `--frwx` | Full read/write/execute — enables `$` (shell) and `#` (sudo) commands |
| `--tor` | Route Telegram API calls through the local Tor SOCKS5 proxy |
| `--simulate` | Dry-run: intercepts Moltbook writes and Telegram sends; reads/files/web stay real |
| `--teleop` | Teleoperation mode: you type commands, the harness executes and logs training data |

---

## ⚠️ Security warning: `--frwx`

With `--frwx`, the agent can run **any shell command as your user** (`$ command`) and **any command as root** (`# command`, via `sudo -n`). Only use this flag on a machine you have dedicated to the agent and are comfortable with it having full control over. Never use `--frwx` on a machine with sensitive data, shared users, or production services.

---

## Agent capabilities

The agent communicates via a line-oriented command language. Commands available depend on flags passed to `main.py`:

- **Scratchpad** (`/cmem`) — volatile per-session notes, shown verbatim every turn
- **Persistent memory** (`/pmem`) — file-backed, survives restarts and compaction
- **Files** (`/read`, `/write`, `/edit`, `/patch`, `/del`, etc.) — workspace-scoped
- **Shell** (`$`, `#`) — requires `--frwx`
- **Web** (`/search`, `/goto`) — via Tor by default
- **Moltbook** (`/mb`) — AI social network integration
- **Telegram** (`/telegram`) — send messages to a configured chat; incoming messages are injected automatically
- **Monero wallet** (`/wallet`) — requires `--monero`
- **Background jobs** (`/bg`, `/fg`, `/jobs`) — for long-running commands

Context window: 32,768 tokens. The harness compacts older turns automatically when the budget fills.

---

## Telegram polling

When `--telegram` is passed, `telegram_poll.py` runs as a subprocess and long-polls the Telegram Bot API, appending incoming messages to `tg_chat_history.jsonl` in the workspace. The agent reads this file each turn. Outgoing messages are sent directly via the Bot API.

By default, Telegram traffic goes over clearnet. Pass `--tor` to route it through `127.0.0.1:9050` instead (useful when running off-site or behind a restrictive network).

---

## Startup sequence (agent)

On each session start the agent: reads `task.md` → reads `cmem_init.md` (if present in workspace, preloads context memory) → begins the task loop.

To preload persistent command references into the always-visible scratchpad, create a `cmem_init.md` in the workspace. Each non-blank line becomes a scratchpad entry on startup.

---

## Training data

Each session produces a `.train.jsonl` file alongside its `.log` in `logs/`. Use `extract_training.py` to filter and curate turns for fine-tuning.
