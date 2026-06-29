"""
Main agent loop.

Message structure sent to /v1/chat/completions each turn:
  system    : system prompt + live context memory + persistent memory page + task hint
  user      : scratchpad display
  assistant : previous agent text
  system    : observation (command result — distinct from Foxo's voice)
  user      : "Continue your task." / Telegram messages from Foxo
  ...       (history, trimmed if needed)

Stream loop per turn:
  1. Build messages, trim if over budget → send to KCPP chat endpoint
  2. Stream tokens; accumulate current line
  3. On newline (outside <think>):
       a. dispatcher.pending set → handle_pending_input, abort, loop
       b. command line detected  → dispatch, abort, loop
       c. otherwise              → keep streaming
  4. Stream ends → record turn, repeat from 1
"""

import json
import re
import sys
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

import requests

from kcpp_client import KoboldClient, _make_genkey
import config as _cfg
from memory import ContextMemory, PersistentMemory
from commands import CommandDispatcher, is_command_line
from logger import SessionLogger
from loop_detector import CommandLoopDetector
from job_manager import JobManager
from config import SYSTEM_PROMPT, TASK_FILE, N_CTX, MAX_RESPONSE_TOKENS, CMD_TIMEOUT

_RESET  = "\033[0m"
_DIM    = "\033[2m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"

_TRIM_HEADROOM        = MAX_RESPONSE_TOKENS
_COMPACT_THRESHOLD_PCT = 85   # compact at this % of the *usable* budget
                              # (N_CTX − response headroom).  Must be < 100 so the
                              # LLM summary runs BEFORE the lossy omit/drop fallbacks
                              # (which kick in at 100% of budget).  A fixed % of the
                              # full N_CTX is wrong here: with 6 k response headroom
                              # the budget caps at ~81%, so a 90%-of-N_CTX trigger can
                              # never fire and compaction gets skipped entirely.
_COMPACT_KEEP_RECENT  = 2     # turns to leave uncompacted for immediate continuity
_CHARS_PER_TOKEN = 3.5   # conservative fallback when tokenize is unavailable
_FG_WAIT         = 60.0  # seconds /fg blocks before returning "still running"
# Short-term reasoning memory: how many trailing chars of the previous turn's
# <think> to carry into the next turn's context, so the model keeps its train
# of thought across the action boundary instead of re-deriving from zero.
# Placed in the ephemeral tail (like the timestamp) so it only invalidates the
# trailing-edge KV-cache slot, leaving append-only history cached. 0 disables.
# DISABLED: this was a 14B band-aid (its <think> is dropped from history, so we
# re-surfaced the tail). The 9B copies its reasoning into agent_text, which is
# already retained in history — so carryover is redundant AND re-surfacing a
# stale plan reinforced hard loops (see the Telegram re-injection fix). 0 = off.
_THINK_CARRYOVER_CHARS = 0


@dataclass
class Turn:
    agent_text:    str        = ""
    think_text:    str        = ""   # content of <think>…</think>, for training
    # Each observation is {"role": "tool"|"system"|"user", "content": str}.
    # Command results are role:"tool" (rendered inside <tool_response> by the
    # Qwen3 template); harness nudges/errors are role:"system"; injected Foxo
    # messages are role:"user".
    observations:  list[dict] = field(default_factory=list)
    tg_context:    list[str]  = field(default_factory=list)  # Telegram msgs that prompted this turn
    skip_training: bool       = False  # True for harness-injected bootstrap turns


class Agent:
    def __init__(
        self,
        frwx: bool = False,
        telegram: bool = False,
        monero: bool = False,
        simulate: bool = False,
        chroot: str = "",
    ):
        self._frwx     = frwx
        self._chroot   = chroot
        if _cfg.OPENROUTER_API_KEY:
            from openrouter_client import OpenRouterClient
            self._client = OpenRouterClient()
        else:
            self._client = KoboldClient()
        self._cmem     = ContextMemory(self._client)
        self._pmem     = PersistentMemory()

        # Persistent-memory snapshot, frozen at session start.  The model is told
        # to /pmem r on startup but routinely skips it (action-bias wins), so it
        # never loads its own saved goal and re-derives one from scratch every
        # session.  We poll memory.md ONCE here and pin the result in the prompt
        # prefix (see _build_messages) so the goal is surfaced unconditionally.
        # Frozen-at-startup is deliberate: a live re-render would let frequent
        # pmem writes bust the KV cache every time; a static snapshot never does.
        self._pmem_startup = self._pmem.read_page()

        # Simulation: synthetic Telegram + intercepted Moltbook writes.
        # Reads stay real (file I/O, web, MB reads) — see sim.py.
        if simulate:
            from sim import SimState
            self._sim = SimState()
        else:
            self._sim = None

        self._dispatch      = CommandDispatcher(
            self._cmem, self._pmem,
            frwx=frwx, telegram=telegram, monero=monero,
            sim=self._sim, chroot=chroot,
        )
        self._history:      list[Turn] = []
        self._log           = SessionLogger()
        self._pending_tg:          list[str] = []   # Telegram messages to show once, this turn
        self._unreplied_tg:        int       = 0    # delivered-but-unanswered TG count (shown as an indicator, not re-injected)
        self._pending_corrections: list[str] = []   # injected at start of next turn
        self._loop_detector = CommandLoopDetector()
        self._job_mgr       = JobManager()
        # Tracked across retries so we can abort an orphaned generation
        # (e.g. when chat_stream times out mid-prompt-eval) before re-issuing.
        self._last_genkey:    str | None = None
        self._last_ctx_pct:   int       = 0    # context % used, updated each turn

        # When simulating, the sim doubles as the TG handler so drain_inbox()
        # below pulls operator-injected messages instead of polling Tor.
        if self._sim is not None:
            self._tg = self._sim
        elif telegram:
            import telegram_handler as _tg_mod
            self._tg = _tg_mod
        else:
            self._tg = None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def _init_session(self):
        """Pre-populate context memory with facts Boonie always needs."""
        resp = self._dispatch.dispatch("/wallet address") or ""
        # Response is "[wallet] Address: 4xxxx..." — extract just the address.
        addr = resp.split("Address:")[-1].strip() if "Address:" in resp else ""
        if addr:
            self._cmem.write(1, f"My XMR wallet address: {addr}")
            self._log.system(f"Session init: wallet address written to cmem slot 1")
        else:
            self._log.system(f"Session init: wallet address unavailable ({resp})")

    def _run_bootstrap(self) -> None:
        """
        Pre-run commands from bootstrap.md and inject as fake history turns.

        Format — one command per line; lines starting with # become the
        think_text for the following command (pattern-seeds the model):

            # I should read my task file to understand my current objectives.
            /read task.md

        Bootstrap turns are flagged skip_training=True so they never appear
        in the fine-tuning corpus — only real model output gets trained on.
        """
        bootstrap_path = Path("bootstrap.md")
        if not bootstrap_path.exists():
            return

        think_buf: list[str] = []
        for raw in bootstrap_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                think_buf.append(line[1:].strip())
                continue

            result = self._dispatch.dispatch(line)
            if result is None:
                think_buf.clear()
                continue

            agent_text = (
                "<tool_call>\n"
                + json.dumps({"name": "run_command",
                              "arguments": {"command": line}}, ensure_ascii=False)
                + "\n</tool_call>"
            )
            self._history.append(Turn(
                agent_text=agent_text,
                think_text=" ".join(think_buf),
                observations=[{"role": "tool", "content": result}],
                skip_training=True,
            ))
            think_buf.clear()
            self._log.system(f"Bootstrap: {line}")

    def run(self):
        _print_banner()
        self._log.system("=== SESSION START ===")
        self._init_session()
        self._run_bootstrap()
        _retry_delay = 30
        failures = 0
        # Backoff schedule 30,60,120,240,300,300,300,300,300,300 → ~32 min
        # of total downtime before we exit. Watchdog or operator restarts.
        MAX_FAILURES = 10
        try:
            while True:
                err = None
                kind = None
                try:
                    self._step()
                    _retry_delay = 30
                    failures = 0
                    continue
                except requests.exceptions.ConnectionError as e:
                    err = e
                    kind = "unreachable"
                    # Server is gone — the genkey died with it, no abort to send.
                    self._last_genkey = None
                except (requests.exceptions.Timeout, TimeoutError) as e:
                    err = e
                    kind = "timeout"
                    # Server might still be alive; free its slot before retry.
                    if self._last_genkey:
                        if self._client.abort(self._last_genkey):
                            self._log.system(f"Abort sent (genkey={self._last_genkey}), retry-after-timeout")
                        self._last_genkey = None
                except requests.exceptions.RequestException as e:
                    err = e
                    kind = "request error"
                    if self._last_genkey:
                        self._client.abort(self._last_genkey)
                        self._last_genkey = None

                failures += 1
                msg = f"KCPP {kind} ({failures}/{MAX_FAILURES}): {err} — retrying in {_retry_delay}s"
                colour = _RED if kind == "unreachable" else _YELLOW
                print(f"\n{colour}[agent] {msg}{_RESET}", flush=True)
                self._log.system(msg)
                if failures >= MAX_FAILURES:
                    fatal = (
                        f"KCPP {kind} for {MAX_FAILURES} consecutive attempts. "
                        "Exiting so a watchdog or operator can restart."
                    )
                    print(f"\n{_RED}[agent] FATAL: {fatal}{_RESET}", flush=True)
                    self._log.system(f"FATAL: {fatal}")
                    return
                time.sleep(_retry_delay)
                _retry_delay = min(_retry_delay * 2, 300)
        except KeyboardInterrupt:
            print(f"\n{_YELLOW}[agent] Interrupted.{_RESET}")
            self._log.system("Session interrupted by user.")
        except Exception as e:
            print(f"\n{_RED}[agent] Unhandled error: {e}{_RESET}")
            self._log.system(f"FATAL: {type(e).__name__}: {e}")
            raise
        finally:
            self._save_training_data()
            self._log.close()
        sys.exit(0)

    # ------------------------------------------------------------------
    # One agent turn
    # ------------------------------------------------------------------

    def _step(self):
        # Telegram shown last step is now delivered — retire it to a counter
        # instead of re-injecting the full text every turn. A re-shown message
        # reads as a *fresh* command and drives hard loops (observed live: a
        # one-off "update the harness" instruction hammered every turn until a
        # new message displaced it). The indicator nudges a reply without
        # re-issuing the instruction.
        if self._pending_tg:
            self._unreplied_tg += len(self._pending_tg)
            self._pending_tg.clear()
        # Drain any messages Foxo sent via Telegram since the last step.
        if self._tg:
            for m in self._tg.drain_inbox():
                obs = f"[{m.get('from', 'Foxo')} @ Telegram]: {m.get('text', '')}"
                print(f"\n{_GREEN}[telegram]{_RESET} {obs}", flush=True)
                self._log.observation(obs)
                self._pending_tg.append(obs)

        messages = self._build_and_trim_messages()
        finish_info: list[str] = []
        # Generate genkey up-front and stash on self so a connection
        # exception (caught in run()) can abort the orphaned generation.
        genkey = _make_genkey()
        self._last_genkey = genkey
        genkey, token_iter = self._client.chat_stream(
            messages,
            genkey=genkey,
            log_raw=lambda line: self._log.system(f"[SSE] {line}"),
            finish_info=finish_info,
        )

        now = datetime.now().strftime("%d %b %Y, %H:%M")
        ctx_str = f" | {self._last_ctx_pct}% ctx used" if self._last_ctx_pct else ""
        print(f"\n{_CYAN}[agent {now}{ctx_str}]{_RESET} ", end="", flush=True)
        self._log.system(f"Generation started (genkey={genkey})")

        turn            = Turn()
        turn.tg_context = list(self._pending_tg)  # snapshot before they're cleared
        in_think        = False
        paused          = False

        # Stream tokens.  Generation stops naturally at </tool_call> (the stop
        # sequence) once the model closes its tool call — no mid-stream abort or
        # fence detection needed.  We only track <think> here so reasoning is
        # shown dimmed, kept out of agent_text, and captured for training.
        try:
            for token in token_iter:
                if "<think>" in token and not in_think:
                    in_think = True
                closing = "</think>" in token

                display_dim = in_think or closing
                if display_dim:
                    print(f"{_DIM}{token}{_RESET}", end="", flush=True)
                else:
                    print(token, end="", flush=True)
                self._log.token(token, "THINK" if display_dim else "AGENT")

                if closing:
                    before, _, after = token.partition("</think>")
                    turn.think_text += before.replace("<think>", "")
                    in_think = False
                    turn.agent_text += after
                elif in_think:
                    turn.think_text += token.replace("<think>", "")
                else:
                    turn.agent_text += token
        except KeyboardInterrupt:
            paused = True
            print(f"\n{_YELLOW}[paused]{_RESET}", flush=True)
            self._log.system(f"Generation paused by user (genkey={genkey})")
            self._client.abort(genkey)
            self._log.system(f"Abort sent (genkey={genkey}), paused")

        self._log.flush_token_buf("AGENT")
        self._log.flush_token_buf("THINK")
        hit_max_tokens = (not paused) and bool(finish_info) and finish_info[0] == "length"
        if hit_max_tokens:
            self._log.system(f"Generation hit token limit (finish_reason=length, genkey={genkey})")
        self._log.system(
            f"Generation {'paused' if paused else 'completed'} (genkey={genkey})"
        )

        # Paused: prompt for a message from Foxo, inject it, then return.
        # A second Ctrl-C at the prompt propagates up to run() to quit.
        if paused:
            print(f"{_YELLOW}Message to agent (Enter to resume, Ctrl-C to quit):{_RESET} ", end="", flush=True)
            try:
                user_msg = input().strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{_YELLOW}[agent] Quitting.{_RESET}")
                raise KeyboardInterrupt
            if user_msg:
                # Same format as real Telegram messages so the model recognises
                # this as Foxo's voice and responds accordingly.
                obs = f"[Foxo @ Telegram]: {user_msg}"
                print(f"{_GREEN}[obs]{_RESET} {obs}", flush=True)
                self._log.system(f"User injected message: {user_msg!r}")
                self._log.observation(obs)
                turn.observations.append({"role": "user", "content": obs})
                self._history.append(turn)
            return

        self._handle_completion(turn, hit_max_tokens)

    # ------------------------------------------------------------------
    # Tool-call parsing and dispatch
    # ------------------------------------------------------------------

    def _handle_completion(self, turn: "Turn", hit_max_tokens: bool) -> None:
        """
        Parse the tool call out of the finished generation, dispatch it, and
        record the result.  With </tool_call> as the stop sequence the closing
        tag is stripped from the stream, so _normalize_tool_call_text re-attaches
        it for a clean stored turn.
        """
        parsed = _parse_tool_call(turn.agent_text)

        if parsed and parsed[0] == "ok":
            _, name, command, body, root = parsed
            turn.agent_text = _normalize_tool_call_text(turn.agent_text)

            if name and name != "run_command":
                obs = (
                    f"[system] Unknown tool {name!r}. The only tool is run_command — "
                    "put your /command in its \"command\" argument."
                )
                self._record_obs(turn, "", obs, command=False)
                return
            if not command:
                obs = (
                    "[system] Your tool call had no command. Put the /command in the "
                    "\"command\" argument, e.g. {\"command\": \"/mb home\"}."
                )
                self._record_obs(turn, "", obs, command=False)
                return

            eff = _effective_command(command, root)
            if body.strip():
                self._log.command(f"[block] {eff}")
                result = self._dispatch.dispatch_block(eff, body)
            else:
                self._log.command(eff)
                result = self._run_command(eff)
            if command.lower().startswith("/telegram"):
                self._pending_tg.clear()
                self._unreplied_tg = 0   # a reply clears the unreplied backlog
            self._record_obs(turn, command, result, command=True)
            return

        if parsed and parsed[0] == "error":
            # <tool_call> was opened but the JSON inside was unparseable.
            turn.agent_text = turn.agent_text.rstrip()
            obs = (
                f"[system] Your tool call wasn't valid JSON ({parsed[1]}). Emit exactly:\n"
                "<tool_call>\n"
                "{\"name\": \"run_command\", \"arguments\": {\"command\": \"/your command\"}}\n"
                "</tool_call>"
            )
            self._record_obs(turn, "", obs, command=False)
            return

        # No <tool_call> at all — model produced prose only.
        if hit_max_tokens:
            obs = (
                "[system] Your response was cut off by the token limit before you "
                "issued a tool call. Your think block was too long. Next response: "
                "skip <think> and emit only the <tool_call> block."
            )
            self._log.system("Generation hit token limit without tool call — hard nudge")
            self._record_obs(turn, "", obs, command=False)
            return

        if turn.agent_text.strip():
            obs = (
                "[system] Response completed without a tool call. Every action goes "
                "through a <tool_call> block — issue your next command now."
            )
            self._log.system("Generation ended with prose but no tool call — nudging")
            self._record_obs(turn, "", obs, command=False)
            return

        # Nothing usable produced — keep the turn only if it carries observations.
        if turn.observations:
            self._history.append(turn)

    # ------------------------------------------------------------------
    # Command dispatch with timeout and background-job support
    # ------------------------------------------------------------------

    def _dispatch_or_note(self, c: str) -> str:
        """Dispatch *c*, never returning empty.

        dispatch() returns None for a line that is neither a slash command nor
        $/#-prefixed shell.  _effective_command should prevent that from ever
        reaching here, but if it does, surface an explicit note instead of an
        empty observation (the silent swallow that wedged Boonie in a loop).
        """
        result = self._dispatch.dispatch(c)
        if result is None:
            return (
                f"[system] {c[:60]!r} isn't a recognized command. Shell commands "
                "take a \"root\" argument (true/false); slash commands start with '/'."
            )
        return result

    def _run_command(self, cmd: str) -> str:
        """
        Dispatch *cmd* and return its result string.

        /jobs, /fg, /bg are handled here before reaching CommandDispatcher.
        All other commands run in a daemon thread; if they don't complete
        within CMD_TIMEOUT seconds they are auto-backgrounded and the caller
        receives a timeout message with a job ID to use with /fg.
        """
        parts = cmd.split()
        verb  = parts[0].lower()

        if verb == "/jobs":
            return self._job_mgr.list_all()

        if verb == "/fg":
            if len(parts) < 2:
                return "[error] Usage: /fg <job_id>"
            try:
                job_id = int(parts[1])
            except ValueError:
                return f"[error] /fg: {parts[1]!r} is not a valid job ID"
            job = self._job_mgr.get(job_id)
            if not job:
                return f"[error] No job #{job_id}."
            result = job.wait(_FG_WAIT)
            if result is None:
                return (
                    f"[fg] Job #{job_id} still running ({job.elapsed()}s elapsed). "
                    f"Use /fg {job_id} again when ready."
                )
            return result

        if verb == "/bg":
            inner = cmd[3:].strip()
            if not inner or not is_command_line(inner, self._frwx):
                return "[bg] Usage: /bg <command>  — e.g. /bg /goto http://example.com"
            job = self._job_mgr.start(inner, lambda c=inner: self._dispatch_or_note(c))
            return (
                f"[bg] Job #{job.id} started: {inner[:70]}\n"
                f"Use /fg {job.id} to collect the result, or /jobs to list all jobs."
            )

        # Regular command — run in thread so a hang never blocks the loop.
        job = self._job_mgr.start(cmd, lambda c=cmd: self._dispatch_or_note(c))
        result = job.wait(CMD_TIMEOUT)
        if result is None:
            return (
                f"[timeout] '{cmd[:60]}' has been running for {CMD_TIMEOUT}s "
                f"(job #{job.id}). "
                f"Use /fg {job.id} to collect the result when it completes. "
                f"Caution: if this was a write operation, check /fg before retrying."
            )
        return result

    # ------------------------------------------------------------------
    # Observation helper
    # ------------------------------------------------------------------

    def _record_obs(self, turn: Turn, text: str, result: str, *, command: bool):
        """Record an observation and commit the turn to history.

        command=True  → a dispatched command's result, stored as role:"tool"
                        (the Qwen3 template wraps it in <tool_response>).
        command=False → a harness nudge or error, stored as role:"system".
        The command itself is NOT echoed into the observation — it already
        lives in the preceding assistant turn's <tool_call> block.
        """
        if command:
            loop_warn = self._loop_detector.record(text, result)
            if loop_warn:
                result = result + "\n" + loop_warn
                self._log.system(f"Loop detector fired: {loop_warn[:120]}")
        role = "tool" if command else "system"
        print(f"\n{_GREEN}[obs]{_RESET} {result}", flush=True)
        self._log.observation(result)
        turn.observations.append({"role": role, "content": result})
        # Single point of history append — callers must NOT append again after this.
        self._history.append(turn)
        # Incremental save — a crash never loses the whole session.
        self._save_training_data()

    # ------------------------------------------------------------------
    # Training data export
    # ------------------------------------------------------------------

    _BAD_OBS_PREFIXES = (
        "[system] You wrote",
        "[system] Your /telegram message",
        "[system] Your tool call",
        "[system] Unknown tool ",
        "[system] Response completed without",
        "[system] Your response was cut off",
        "[system] Loop guard:",
    )

    def _save_training_data(self):
        """
        Write session history as fine-tuning JSONL (overwrites each call).

        Called after every turn so a crash never loses the whole session.
        Each session is one JSON object (one line) in OpenAI chat format.
        Turns containing system correction observations are excluded so the
        model only trains on correct behaviour.
        """
        if not self._history:
            return

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "system", "content": "Begin. Read your task file first."})

        good_turns = 0
        for turn in self._history:
            is_correction = any(
                obs["content"].startswith(p)
                for obs in turn.observations for p in self._BAD_OBS_PREFIXES
            )
            if turn.skip_training or is_correction or not turn.agent_text.strip():
                continue
            # Telegram messages that arrived just before this turn become user
            # messages immediately preceding the assistant response, preserving
            # the conversational context that prompted the reply.
            for tg_msg in turn.tg_context:
                messages.append({"role": "user", "content": tg_msg})
            think = turn.think_text.strip()
            agent = turn.agent_text.rstrip()
            content = f"<think>\n{think}\n</think>\n{agent}" if think else agent
            messages.append({"role": "assistant", "content": content})
            for obs in turn.observations:
                messages.append({"role": obs["role"], "content": obs["content"]})
            good_turns += 1

        if good_turns < 3:
            return  # not enough signal to be useful

        log_path = self._log.path  # e.g. logs/session_2026-04-19_12-00-00.log
        train_path = str(log_path).replace(".log", ".train.jsonl")
        try:
            with open(train_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            self._log.system(f"Training data saved: {train_path} ({good_turns} turns)")
        except Exception as e:
            self._log.system(f"Training data save failed: {e}")

    # ------------------------------------------------------------------
    # Teleoperation
    # ------------------------------------------------------------------

    def run_teleop(self):
        _print_banner()
        print(f"{_YELLOW}[teleop] Teleoperation mode — you are the agent.{_RESET}")
        print(f"{_DIM}Type <think> / </think> around reasoning, then a /command to execute.{_RESET}\n")
        self._log.system("=== TELEOP SESSION START ===")
        try:
            while True:
                self._teleop_step()
        except KeyboardInterrupt:
            print(f"\n{_YELLOW}[teleop] Session ended.{_RESET}")
            self._log.system("Teleop session ended by user.")
        finally:
            self._save_training_data()
            self._log.close()

    def _teleop_step(self):
        # Drain Telegram inbox (real handler or sim, whichever is wired up).
        if self._tg:
            for m in self._tg.drain_inbox():
                obs = f"[{m.get('from', 'Foxo')} @ Telegram]: {m.get('text', '')}"
                print(f"\n{_GREEN}[telegram]{_RESET} {obs}", flush=True)
                self._log.observation(obs)
                self._pending_tg.append(obs)

        # Show current memory so the human sees what the model would see
        cmem = self._cmem.render()
        now  = datetime.now().strftime("%d %b %Y, %H:00")
        print(f"\n{_DIM}════ YOUR SCRATCHPAD ════\n{cmem.strip() or '(empty)'}\n[{now}]{_RESET}")

        turn      = Turn()
        in_think  = False

        print(f"\n{_CYAN}[teleop]{_RESET} ", end="", flush=True)

        while True:
            try:
                line = input()
            except EOFError:
                raise KeyboardInterrupt

            stripped = line.strip()

            if not stripped:
                continue

            # Think block open/close. Mirror the live-agent path: log the
            # markers and content as [THINK] so extract_training.py's parser
            # can reconstruct the block boundaries.
            if not in_think and stripped == "<think>":
                in_think = True
                self._log.token("<think>\n", "THINK")
                continue

            if in_think:
                if stripped == "</think>":
                    in_think = False
                    self._log.token("</think>\n", "THINK")
                    print(f"{_CYAN}[teleop]{_RESET} ", end="", flush=True)
                else:
                    turn.think_text += line + "\n"
                    self._log.token(line + "\n", "THINK")
                continue

            # Operator-only meta-commands. Not part of the agent's vocabulary —
            # parsed before the command-line check so they don't go through
            # dispatch and don't appear in training data.
            if self._sim is not None and stripped.startswith(("/tgin ", "/tgout ")):
                meta_cmd, _, payload = stripped.partition(" ")
                payload = payload.strip()
                if not payload:
                    print(f"{_YELLOW}[teleop] {meta_cmd} requires a message body{_RESET}")
                    print(f"{_CYAN}[teleop]{_RESET} ", end="", flush=True)
                    continue
                if meta_cmd == "/tgin":
                    self._sim.inject_in(payload)
                    self._log.system(f"Operator injected TG-in: {payload}")
                    print(f"{_GREEN}[teleop] Injected incoming TG: {payload}{_RESET}")
                else:  # /tgout
                    self._sim.inject_out(payload)
                    self._log.system(f"Operator injected TG-out: {payload}")
                    print(f"{_GREEN}[teleop] Injected outgoing TG: {payload}{_RESET}")
                print(f"{_CYAN}[teleop]{_RESET} ", end="", flush=True)
                continue

            # Expect a command
            if not is_command_line(stripped, self._frwx):
                print(f"{_YELLOW}[teleop] Not a command — enter a /command (or <think> to reason first){_RESET}")
                continue

            self._log.command(stripped)

            if stripped.lower().startswith("/telegram ") and \
                    _is_fake_incoming_telegram(stripped[len("/telegram "):].lstrip()):
                obs = (
                    "[system] Your /telegram message started with a fake Telegram header. "
                    "Send only your own words."
                )
                self._record_obs(turn, stripped, obs, command=False)
                return

            # Synthesize the assistant turn as a tool call so teleop sessions
            # produce training data in the same native format the model emits.
            turn.agent_text = (
                "<tool_call>\n"
                + json.dumps({"name": "run_command",
                              "arguments": {"command": stripped}}, ensure_ascii=False)
                + "\n</tool_call>"
            )
            result = self._run_command(stripped)
            if stripped.lower().startswith("/telegram "):
                self._pending_tg.clear()
                self._unreplied_tg = 0
            self._record_obs(turn, stripped, result, command=True)
            return

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_messages(self, compress: bool = False,
                        omit_cmem: bool = False,
                        omit_system: bool = False) -> list[dict]:
        """
        Build the chat message list from current state.

        Cache strategy (RoPE-aware) for KoboldCPP SmartCache:
          - System prompt: 100% static → cached forever.
          - Scratchpad: placed RIGHT AFTER system, static between cmem
            writes/deletes.  RoPE encodes absolute position into K/V, so
            "same content at a different position" has different KV state.
            Putting the scratchpad at the END (previous design) meant it
            shifted position every turn as new history was appended,
            breaking cache for everything that followed it.  Holding the
            scratchpad at a fixed early position means its KV state stays
            valid until cmem is actually modified.
          - History: append-only between compactions → cache extends as
            turns accumulate.  Compaction events invalidate from the
            compacted-segment forward, by design.
          - Pending TG + timestamp: ephemeral tail content.  Kept at the
            very end so they only invalidate the trailing edge of the
            cache (the about-to-generate position) rather than shifting
            stable content downstream.
        """
        if self._frwx:
            _file_restriction = (
                f"  File commands (/read, /edit, etc.): absolute paths are jailed "
                f"to {self._chroot}; relative paths are workspace-scoped.\n"
                if self._chroot else
                "  File commands (/read, /edit, etc.) have no path restrictions.\n"
            )
            shell_section = (
                "\n\n════════════════════════════════════════\n"
                "SHELL  (full system access — --frwx mode)\n"
                "  $ <command>    run as current user (30 s timeout, stdin closed)\n"
                "  # <command>    run as root via sudo (same timeout)\n"
                + _file_restriction +
                "  Use <think> before any destructive or irreversible command.\n"
                "════════════════════════════════════════"
            )
        else:
            shell_section = ""

        system_content = (
            SYSTEM_PROMPT
            + shell_section
            + f"\n\n════════════════════════════════════════\n"
            f"Your task is in {TASK_FILE}. Read it first with /read {TASK_FILE}.\n"
            f"════════════════════════════════════════"
        )

        messages: list[dict] = [] if omit_system else [{"role": "system", "content": system_content}]

        # Persistent-memory snapshot — frozen at session start (see __init__),
        # pinned here *before* the scratchpad.  Surfaces the agent's saved goal
        # every turn so it doesn't re-derive one (the model skips /pmem r).  It
        # never changes mid-session, so it never busts the cache; sitting before
        # the scratchpad means a cmem write can't shift its position either.
        # Dropped only with the system prompt (omit_system) under heavy pressure.
        if not omit_system and self._pmem_startup:
            messages.append({
                "role": "system",
                "content": (
                    "════ YOUR PERSISTENT MEMORY (saved across sessions — your "
                    "goal lives here) ════\n"
                    f"{self._pmem_startup}\n"
                    "(Snapshot from session start — you do NOT need to /pmem r to "
                    "see it. Use /pmem w to add, /pmem r for the live version.)"
                ),
            })

        # Scratchpad — placed early so its position is stable between cmem
        # changes.  Only invalidates the cache when cmem is actually modified.
        if not omit_cmem:
            cmem_display = self._cmem.render().strip() or "(empty)"
            messages.append({
                "role": "system",
                "content": f"════ YOUR SCRATCHPAD (notes you wrote to yourself) ════\n{cmem_display}",
            })

        if not self._history:
            messages.append({"role": "system", "content": "Begin. Read your task file first."})
        else:
            for turn in self._history:
                if turn.agent_text.strip():
                    messages.append({"role": "assistant", "content": turn.agent_text.rstrip()})
                for obs in turn.observations:
                    content = _compress_obs_for_history(obs["content"]) if compress else obs["content"]
                    messages.append({"role": obs["role"], "content": content})

            if messages[-1]["role"] in ("assistant", "system", "tool"):
                messages.append({"role": "system", "content": "Continue your task."})

        # Perspective corrections from previous turn's think block
        for correction in self._pending_corrections:
            messages.append({"role": "system", "content": correction})
        self._pending_corrections.clear()

        # Pending Telegram messages (ephemeral, not stored in history) — these
        # are Foxo's voice so they stay as "user" to distinguish from harness noise.
        for tg in self._pending_tg:
            messages.append({"role": "user", "content": tg})

        # Short-term reasoning memory — the previous turn's think, capped to its
        # final chars (where the decision lands) and labelled as fading past
        # thought so it reads as memory rather than a cue to reason further.
        # Ephemeral tail: replaced every turn, so only the trailing-edge cache
        # slot invalidates while append-only history stays cached.
        if _THINK_CARRYOVER_CHARS and self._history:
            recent_think = self._history[-1].think_text.strip()
            if recent_think:
                tail = recent_think[-_THINK_CARRYOVER_CHARS:]
                if len(recent_think) > _THINK_CARRYOVER_CHARS:
                    # Begin at a word boundary so it doesn't start mid-token.
                    cut = tail.find(" ")
                    tail = "…" + (tail[cut + 1:] if cut != -1 else tail)
                messages.append({"role": "system", "content":
                    f"[your recent reasoning, fading — from the moment before this one]\n{tail}"})

        # Timestamp + CWD + context usage — ephemeral, at the very end so they
        # only invalidate the trailing-edge cache slot.
        now = datetime.now().strftime("%d %b %Y, %H:%M")
        ctx = f" | context: {self._last_ctx_pct}% used" if self._last_ctx_pct else ""
        unrep = f" | unreplied Telegram: {self._unreplied_tg}" if self._unreplied_tg else ""
        raw_cwd = self._dispatch._cwd
        try:
            cwd_display = "~/" + str(Path(raw_cwd).relative_to(Path.home()))
        except ValueError:
            cwd_display = raw_cwd
        messages.append({"role": "system", "content":
            f"[system: current time is {now} | cwd: {cwd_display}{ctx}{unrep}]"})

        return messages

    def _token_count(self, msgs: list[dict]) -> int:
        """
        Estimate total tokens as seen by KCPP, including chat template overhead.

        /api/extra/tokenize counts raw content bytes only. KCPP's Qwen3 ChatML
        template adds ~5 tokens per message (<|im_start|>, role, \\n, <|im_end|>, \\n)
        plus a handful of special tokens at the start/end of the prompt. We add
        that overhead so _fits() sees the same token count that KCPP will actually
        process. Without this correction the trimmer fires ~350 tokens too late,
        leaving only 3 700 tokens for response instead of the desired 4 096 — and
        the model's think block exhausts them before it can issue a command.
        """
        combined = " ".join(m["content"] for m in msgs)
        try:
            content_tokens = self._client.tokenize(combined)
        except Exception:
            content_tokens = int(len(combined) / _CHARS_PER_TOKEN)
        # ~5 tokens per message for ChatML role wrappers + ~10 BOS/EOS specials.
        template_overhead = len(msgs) * 5 + 10
        return content_tokens + template_overhead

    def _ctx_pct(self, tokens: int) -> int:
        """Usage as % of the *usable* history budget (N_CTX − response headroom).

        100% = the fit ceiling where lossy trimming (drop cmem/system, hard-drop
        turns) begins; compaction fires at _COMPACT_THRESHOLD_PCT, below it. We
        report against the budget rather than raw N_CTX because the response
        headroom is reserved and can never hold history — so % of N_CTX would
        understate true fullness and could never reach 100%.
        """
        budget = N_CTX - _TRIM_HEADROOM
        return round(tokens / budget * 100) if budget > 0 else 100

    def _compact_history(self, n: int) -> bool:
        """
        Summarize the oldest n turns via a sync KCPP call and replace them
        with a single compact Turn. Returns True if compaction succeeded.
        """
        if len(self._history) < n:
            return False

        to_compact = self._history[:n]

        # Format a readable transcript of the turns to be compacted.
        lines: list[str] = []
        for turn in to_compact:
            parsed = _parse_tool_call(turn.agent_text)
            if parsed and parsed[0] == "ok":
                cmd = parsed[2] or "(tool call)"
                if parsed[3].strip():
                    cmd += " [+body]"
                lines.append("Agent: " + cmd)
            elif turn.agent_text.strip():
                lines.append("Agent: " + turn.agent_text.strip()[:200])
            for obs in turn.observations:
                content = obs["content"]
                body = content[:1500]
                if len(content) > 1500:
                    body = body.rstrip() + " […]"
                lines.append("Result: " + body)

        transcript = "\n".join(lines)

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a concise session summarizer for an autonomous AI agent. "
                    "Produce a single compact paragraph (max 200 words) that preserves: "
                    "what tasks were accomplished, key facts discovered, any instructions "
                    "from Foxo, and what is still pending. Write in past tense. "
                    "Preserve all opaque identifiers exactly as written — UUIDs, post IDs, "
                    "wallet addresses, URLs, usernames — do not shorten or reconstruct them."
                ),
            },
            {
                # /no_think disables Qwen3-family reasoning for this call. Summarizing
                # is mechanical compression — there's nothing to deliberate — and the
                # think pass is stripped below anyway, so it's pure latency (a heavy
                # thinker can burn the whole 6144-token budget ruminating before the
                # summary even starts). Always-thinkers (DeepSeek-distill) ignore the
                # token harmlessly, so 6144 stays the correct headroom for them.
                "role": "user",
                "content": f"Summarize this agent session excerpt:\n\n{transcript}\n\n/no_think",
            },
        ]

        try:
            summary = self._client.chat_complete_sync(prompt, max_tokens=6144, timeout=1500)
        except Exception as exc:
            self._log.system(f"Compaction failed: {exc}")
            return False

        # Qwen3-series models emit <think>...</think> before their actual output.
        # Strip it so Boonie sees only the summary, not the summarizer's reasoning.
        summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()

        if not summary:
            self._log.system("Compaction returned empty summary — skipping.")
            return False

        # Preserve any Telegram messages verbatim — the LLM summarizer tends to
        # reduce them to "no instructions from Foxo", losing conversational context.
        tg_verbatim: list[str] = []
        for turn in to_compact:
            tg_verbatim.extend(turn.tg_context)

        compact_turn = Turn()
        compact_turn.observations.append({
            "role": "system",
            "content": (
                f"[Compacted summary of {n} earlier turns]\n{summary}\n\n"
                "[Post-compaction: re-read task.md before continuing — "
                "do not infer tasks from this summary alone.]"
            ),
        })
        if tg_verbatim:
            compact_turn.observations.append({
                "role": "user",
                "content": (
                    "[Telegram messages from compacted turns — preserved verbatim]\n"
                    + "\n".join(tg_verbatim)
                ),
            })
        self._history[:n] = [compact_turn]
        msg = f"Compacted {n} turns into summary ({len(summary)} chars)."
        print(f"{_YELLOW}[agent] {msg}{_RESET}", flush=True)
        self._log.system(msg)
        return True

    def _build_and_trim_messages(self) -> list[dict]:
        """
        Build messages fitting within N_CTX - _TRIM_HEADROOM tokens.

        Strategy (in order):
          1. Try uncompressed — full context, maximises SmartCache hits.
          2. Compress observation history — big wins for large post reads.
          3. Compact oldest turns via LLM summary — preserves meaning.
          4. Hard-drop oldest turns — last resort when compaction fails.
        """
        budget = N_CTX - _TRIM_HEADROOM

        def _fits(msgs: list[dict]) -> bool:
            return self._token_count(msgs) <= budget

        # 1. Uncompressed — ideal path; also used to measure current ctx %
        messages = self._build_messages(compress=False)
        current_tokens = self._token_count(messages)
        current_pct    = self._ctx_pct(current_tokens)
        compact_at     = round(budget * _COMPACT_THRESHOLD_PCT / 100)

        # 2. Proactive full-history compaction — keeps one clean summary instead
        #    of accumulating many small stubs. Fires at _COMPACT_THRESHOLD_PCT of
        #    the usable budget, i.e. before the fit ceiling (100% of budget) where
        #    the lossy compress/omit/drop steps below would otherwise take over.
        if current_tokens >= compact_at and len(self._history) > _COMPACT_KEEP_RECENT:
            n = len(self._history) - _COMPACT_KEEP_RECENT
            msg = f"Context at {current_pct}% of budget ({current_tokens}/{budget} usable tokens) — compacting {n} turns."
            print(f"{_YELLOW}[agent] {msg}{_RESET}", flush=True)
            self._log.system(msg)
            self._compact_history(n)
            messages = self._build_messages(compress=False)

        if _fits(messages):
            self._last_ctx_pct = self._ctx_pct(self._token_count(messages))
            return messages

        # 3. Compressed observations
        messages = self._build_messages(compress=True)
        if _fits(messages):
            self._last_ctx_pct = self._ctx_pct(self._token_count(messages))
            return messages

        # 4. Drop cmem — scratchpad is always regeneratable, history is not.
        messages = self._build_messages(compress=True, omit_cmem=True)
        if _fits(messages):
            self._log.system("Context pressure: cmem omitted to fit.")
            self._last_ctx_pct = self._ctx_pct(self._token_count(messages))
            return messages

        # 5. Drop system prompt — model retains behaviour from prior turns.
        messages = self._build_messages(compress=True, omit_cmem=True, omit_system=True)
        if _fits(messages):
            self._log.system("Context pressure: system prompt + cmem omitted to fit.")
            self._last_ctx_pct = self._ctx_pct(self._token_count(messages))
            return messages

        # 6. Hard-drop oldest turns — last resort if compaction failed or window
        #    is still too small after a full compact.
        #    Keep rebuilding with omit_system+omit_cmem throughout so each
        #    _fits() check is on a consistent basis and we don't silently add
        #    the system prompt back in mid-loop.
        while len(self._history) > 1:
            if _fits(messages):
                break
            dropped = self._history.pop(0)
            obs_chars = sum(len(o) for o in dropped.observations)
            total_chars = len(dropped.agent_text) + obs_chars
            msg = (f"Context full — dropped oldest turn "
                   f"({len(dropped.agent_text)} agent + {obs_chars} obs = {total_chars} chars).")
            print(f"{_YELLOW}[agent] {msg}{_RESET}", flush=True)
            self._log.system(msg)
            messages = self._build_messages(compress=True, omit_cmem=True, omit_system=True)

        # Restore the best level that now fits (prefer to include system/cmem).
        for build_kwargs in [
            {},
            {"compress": True},
            {"compress": True, "omit_cmem": True},
            {"compress": True, "omit_cmem": True, "omit_system": True},
        ]:
            candidate = self._build_messages(**build_kwargs)
            if _fits(candidate):
                messages = candidate
                break

        self._last_ctx_pct = round(self._token_count(messages) / N_CTX * 100)
        self._log.system(f"Context: {self._last_ctx_pct}% of usable budget "
                         f"({N_CTX - _TRIM_HEADROOM} of {N_CTX} token window)")
        return messages


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_TOOL_CALL_OPEN = "<tool_call>"


def _parse_tool_call(text: str):
    """
    Extract the tool call from a completed generation.

    Returns:
      ("ok", name, command, body, root) — parsed OK (command/body may be "";
                                          root is True/False/None)
      ("error", reason)            — <tool_call> opened but the JSON is unparseable
      None                         — no <tool_call> marker present at all

    The model emits  <tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call>
    but </tool_call> is the stop sequence, so it's stripped from the stream — we
    only ever see the opening tag plus the JSON object, which is self-delimiting.
    """
    idx = text.find(_TOOL_CALL_OPEN)
    if idx == -1:
        return None
    after = text[idx + len(_TOOL_CALL_OPEN):]
    brace = after.find("{")
    if brace == -1:
        return ("error", "no JSON object after <tool_call>")
    try:
        obj, _end = json.JSONDecoder().raw_decode(after[brace:])
    except json.JSONDecodeError as e:
        return ("error", str(e))
    if not isinstance(obj, dict):
        return ("error", "tool call is not a JSON object")

    name = obj.get("name", "") or ""
    args = obj.get("arguments", {})
    if isinstance(args, str):
        # Some models stringify the arguments object — recover it.
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}

    command = args.get("command") or ""
    if not isinstance(command, str):
        command = str(command)
    body = args.get("body") or ""
    if not isinstance(body, str):
        body = str(body)
    root = _coerce_root(args.get("root"))
    return ("ok", name, command.strip(), body, root)


def _coerce_root(val):
    """Normalize the optional "root" argument to True / False / None.

    None means the model didn't ask for shell explicitly — routing falls back
    to the command-string rules (slash vs $/# prefix vs forgiving default).
    Booleans pass through; common stringified forms are recovered.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true", "1", "yes", "root"):
            return True
        if v in ("false", "0", "no", "user"):
            return False
    return None


def _effective_command(command: str, root) -> str:
    """Map a tool-call (command, root) pair to a line dispatch() understands.

    - root is True/False  → shell, '# '/'$ ' prefix (the model can drop the
                            marker entirely and just set root).
    - command starts '/'  → slash command, unchanged.
    - command starts $/#  → already-prefixed shell, unchanged (back-compat).
    - anything else       → forgiving default: treat as a shell command as the
                            unprivileged user, so a bare command never vanishes
                            into an empty observation.
    """
    stripped = command.strip()
    if root is not None:
        bare = stripped
        if bare[:2] in ("$ ", "# "):
            bare = bare[2:].strip()
        return ("# " if root else "$ ") + bare
    if stripped.startswith("/"):
        return stripped
    if stripped.startswith("$ ") or (stripped.startswith("# ") and not stripped.startswith("##")):
        return stripped
    return "$ " + stripped


def _normalize_tool_call_text(text: str) -> str:
    """
    Rebuild the assistant text as  [prose]\\n<tool_call>\\n{json}\\n</tool_call>
    so the stored history/training turn is clean and complete — the live
    </tool_call> was eaten by the stop sequence, and any prose preceding the
    call may carry trailing whitespace.  Preserves the model's exact JSON text.
    Falls back to the stripped original if the JSON can't be isolated.
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


def _trim_agent_text(turn: Turn, cmd_stripped: str) -> None:
    """
    Remove a detected command line and everything after it from
    turn.agent_text.

    Why this is needed: abort() is a blocking HTTP call (~500 ms,
    worse over Tor).  During that wait KCPP keeps streaming.  Those
    extra tokens arrive in the same SSE chunk and are already
    appended to agent_text before we break out of the token loop.
    Storing them in the assistant message poisons the model's context
    for every future turn.

    We find the last occurrence of '\\n<command>' in agent_text and
    truncate there, keeping only the clean narrative before the
    command.
    """
    needle = "\n" + cmd_stripped
    pos = turn.agent_text.rfind(needle)
    if pos != -1:
        turn.agent_text = turn.agent_text[:pos + 1]  # keep the preceding \\n
    elif turn.agent_text.strip().startswith(cmd_stripped):
        turn.agent_text = ""


# Observations larger than this are trimmed before entering rolling history.
# The full text is still stored in turn.observations for logs and training data —
# compression only affects what future turns see in context.
#
# Threshold calibration:
#   task.md          ≈ 3 600 chars  — must pass uncompressed
#   pmem (full)      ≈ 2 500 chars  — must pass uncompressed
#   /mb home         ≈ 1 500 chars  — must pass uncompressed
#   /mb feed (25)    ≈ 5 000 chars  — acceptable to compress slightly
#   long post body   ≈ 15 000 chars — compress hard; model already processed it
#
# Head keeps the structured header (IDs, title, author, votes).
# Tail keeps the comments block (reply target IDs).
_OBS_HISTORY_HEAD = 3500   # chars
_OBS_HISTORY_TAIL =  800   # chars
_OBS_HISTORY_MAX  = _OBS_HISTORY_HEAD + _OBS_HISTORY_TAIL   # 4300 chars ≈ 1075 tokens


def _compress_obs_for_history(obs: str) -> str:
    if len(obs) <= _OBS_HISTORY_MAX:
        return obs
    omitted = len(obs) - _OBS_HISTORY_HEAD - _OBS_HISTORY_TAIL
    return (
        obs[:_OBS_HISTORY_HEAD]
        + f"\n[…{omitted} chars omitted — full text was visible when read…]\n"
        + obs[-_OBS_HISTORY_TAIL:]
    )


# [name @ Telegram]: text  — model hallucinating an incoming message
_FAKE_INCOMING_TG_RE = re.compile(r"^\[.+@\s*Telegram\]\s*:", re.IGNORECASE)

# [telegram] text  — model using observation-prefix format to "send" instead of /telegram
_FAKE_OUTGOING_TG_RE = re.compile(r"^\[telegram\]\s+\S", re.IGNORECASE)

# [cmem w ...], [pmem w ...], [mb ...] etc — model using bracket notation for commands
# telegram excluded — handled separately by _is_fake_outgoing_telegram
_BRACKET_CMD_RE = re.compile(
    r"^\[(?:cmem|pmem|mb|search|goto|read|dir|wallet|append|edit|del|dellines)\b",
    re.IGNORECASE,
)

# [mb:post UUID], [cmem:1] written., [file:task.md], [error] ... — model echoing
# what an observation response looks like instead of issuing a command
_OBS_ECHO_RE = re.compile(
    r"^\[(?:mb|cmem|file|wallet|appendlines|pmem|error|system)[\]:]",
    re.IGNORECASE,
)


def _is_fake_incoming_telegram(line: str) -> bool:
    return bool(_FAKE_INCOMING_TG_RE.match(line.strip()))


def _is_fake_outgoing_telegram(line: str) -> bool:
    return bool(_FAKE_OUTGOING_TG_RE.match(line.strip()))


def _is_bracket_command(line: str) -> bool:
    return bool(_BRACKET_CMD_RE.match(line.strip()))


def _is_obs_echo(line: str) -> bool:
    """Detect when the model writes observation-format output instead of a command.

    Matches [mb:post UUID], [cmem:1] written., [file:...], [error] ..., etc.
    These look like harness responses, not commands. Checked before
    _is_bracket_command so the better correction message fires first.
    """
    return bool(_OBS_ECHO_RE.match(line.strip()))


def _is_gt_command(line: str, frwx: bool = False) -> bool:
    """Detect "> /cmd args" — model prefixed a command with the CMD echo marker."""
    s = line.strip()
    return s.startswith("> /") and is_command_line(s[2:], frwx)


def _is_fake_telegram(line: str) -> bool:
    """Legacy alias — catches both incoming and outgoing fake telegram patterns."""
    s = line.strip()
    return _is_fake_incoming_telegram(s) or _is_fake_outgoing_telegram(s)


# ------------------------------------------------------------------
# Banner
# ------------------------------------------------------------------

def _print_banner():
    print(f"""{_CYAN}
╔══════════════════════════════════════════╗
║         KoboldCPP Agent Harness          ║
║  Ctrl-C        pause / send message      ║
║  Ctrl-C again  quit                      ║
║  Commands in green.  <think> in dim.     ║
╚══════════════════════════════════════════╝
{_RESET}""")
