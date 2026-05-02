"""
Main agent loop.

Message structure sent to /v1/chat/completions each turn:
  system  : system prompt + live context memory + persistent memory page + task hint
  user    : "Begin." (first turn only) OR last observation
  assistant: previous agent text
  user    : observation
  ...     (history, trimmed if needed)

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

import requests

from kcpp_client import KoboldClient
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

_TRIM_HEADROOM   = MAX_RESPONSE_TOKENS
_COMPACT_TRIGGER = 8     # compact when history exceeds this many turns
_COMPACT_BATCH   = 6     # turns to fold into each compaction summary
_CHARS_PER_TOKEN = 3.5   # conservative fallback when tokenize is unavailable
_FG_WAIT         = 60.0  # seconds /fg blocks before returning "still running"


@dataclass
class Turn:
    agent_text:   str       = ""
    think_text:   str       = ""   # content of <think>…</think>, for training
    observations: list[str] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        frwx: bool = False,
        telegram: bool = False,
        monero: bool = False,
        simulate: bool = False,
    ):
        self._frwx     = frwx
        self._client   = KoboldClient()
        self._cmem     = ContextMemory(self._client)
        self._pmem     = PersistentMemory()

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
            sim=self._sim,
        )
        self._history:      list[Turn] = []
        self._log           = SessionLogger()
        self._pending_tg:   list[str] = []   # Telegram messages waiting to be shown
        self._loop_detector = CommandLoopDetector()
        self._job_mgr       = JobManager()

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

    def run(self):
        _print_banner()
        self._log.system("=== SESSION START ===")
        _retry_delay = 30
        try:
            while True:
                try:
                    self._step()
                    _retry_delay = 30   # reset backoff on successful turn
                except requests.exceptions.RequestException as e:
                    msg = f"Connection error: {e} — retrying in {_retry_delay}s"
                    print(f"\n{_YELLOW}[agent] {msg}{_RESET}", flush=True)
                    self._log.system(msg)
                    time.sleep(_retry_delay)
                    _retry_delay = min(_retry_delay * 2, 300)  # cap at 5 min
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
        # Drain any messages Foxo sent via Telegram since the last step.
        if self._tg:
            for m in self._tg.drain_inbox():
                obs = f"[{m.get('from', 'Foxo')} @ Telegram]: {m.get('text', '')}"
                print(f"\n{_GREEN}[telegram]{_RESET} {obs}", flush=True)
                self._log.observation(obs)
                self._pending_tg.append(obs)

        messages = self._build_and_trim_messages()
        finish_info: list[str] = []
        genkey, token_iter = self._client.chat_stream(
            messages,
            log_raw=lambda line: self._log.system(f"[SSE] {line}"),
            finish_info=finish_info,
        )

        now = datetime.now().strftime("%d %b %Y, %H:%M")
        print(f"\n{_CYAN}[agent {now}]{_RESET} ", end="", flush=True)
        self._log.system(f"Generation started (genkey={genkey})")

        turn          = Turn()
        in_think      = False
        cur_line      = ""
        aborted       = False
        paused        = False
        blank_lines   = 0
        prose_lines   = 0   # non-blank content lines without a command
        hit_max_tokens = False

        try:
            for token in token_iter:
                cur_line += token

                was_thinking = in_think
                if not in_think and "<think>" in cur_line:
                    in_think = True
                if in_think and "</think>" in cur_line:
                    in_think = False
                    # Discard everything accumulated in cur_line during the
                    # think block (including the synthetic </think> tag itself).
                    # If we don't clear here, that content bleeds into the very
                    # next newline-split and the first segment passed to
                    # handle_pending_input / is_command_line is "<think>" or a
                    # reasoning line instead of the real command/content.
                    cur_line = ""

                if in_think or was_thinking or "<think>" in token or "</think>" in token:
                    print(f"{_DIM}{token}{_RESET}", end="", flush=True)
                else:
                    print(token, end="", flush=True)

                kind = "THINK" if (in_think or was_thinking) else "AGENT"
                self._log.token(token, kind)

                # Only accumulate non-think content into agent_text.
                # Think-block tokens (including synthetic <think>/<think> tags
                # emitted by the client) must never enter the stored assistant
                # message — they would pollute the model's history and cause it
                # to re-enter reasoning mode on every subsequent turn.
                is_think_token = in_think or was_thinking or "<think>" in token or "</think>" in token
                if not is_think_token:
                    turn.agent_text += token
                elif was_thinking and "</think>" not in token:
                    turn.think_text += token

                # Always reset cur_line on newlines — even inside a think block.
                # Without this, the entire reasoning content accumulates in
                # cur_line and then bleeds through when the first content token
                # with a \n arrives, causing think-block lines to be mistakenly
                # passed to handle_pending_input or is_command_line.
                if "\n" in token and in_think:
                    _, _, cur_line = cur_line.rpartition("\n")

                if "\n" in token and not in_think:
                    segments = cur_line.split("\n")
                    for completed in segments[:-1]:
                        stripped = completed.strip()
                        if not stripped:
                            blank_lines += 1
                            if blank_lines >= 8:
                                self._client.abort(genkey)
                                aborted = True
                                obs = (
                                    "[system] Generation stalled — only blank lines produced "
                                    "after think block. Issue your next command directly."
                                )
                                self._log.system(f"Abort sent (genkey={genkey}), blank-line stall")
                                self._record_obs(turn, "", obs, command=False)
                                break
                            continue
                        blank_lines = 0

                        if self._dispatch.pending is not None:
                            # appendlines, edit phases, and patch are batch:
                            # process each line silently so the model writes the
                            # full block in one generation.  Only abort when
                            # pending clears.  Pass the raw line (no leading
                            # strip) so patch markers and indentation survive.
                            is_batch = self._dispatch.pending.mode in (
                                "appendlines", "edit_old", "edit_new", "patch"
                            )
                            raw_line = completed.rstrip()
                            self._log.pending_input(raw_line)
                            result = self._dispatch.handle_pending_input(raw_line)
                            session_ended = self._dispatch.pending is None
                            if session_ended or not is_batch:
                                _trim_agent_text(turn, stripped)
                                self._client.abort(genkey)
                                aborted = True
                                self._log.system(
                                    f"Abort sent (genkey={genkey}), "
                                    f"pending {'complete' if session_ended else 'input'}"
                                )
                                self._record_obs(turn, stripped, result, command=False)
                                break
                            # else: batch line written, keep streaming

                        elif _is_fake_incoming_telegram(stripped):
                            _trim_agent_text(turn, stripped)
                            self._client.abort(genkey)
                            aborted = True
                            self._log.system(f"Abort sent (genkey={genkey}), hallucinated incoming Telegram")
                            obs = (
                                "[system] You wrote a fake incoming Telegram message. "
                                "Messages from Foxo arrive automatically — never fabricate them. "
                                "Continue your task."
                            )
                            self._record_obs(turn, stripped, obs, command=False)
                            break

                        elif _is_obs_echo(stripped):
                            _trim_agent_text(turn, stripped)
                            self._client.abort(genkey)
                            aborted = True
                            self._log.system(f"Abort sent (genkey={genkey}), observation-echo hallucination")
                            obs = (
                                "[system] You wrote what a command response looks like, not a command. "
                                "Issue the command directly with a leading slash, e.g. /mb read <id>."
                            )
                            self._record_obs(turn, stripped, obs, command=False)
                            break

                        elif _is_bracket_command(stripped):
                            _trim_agent_text(turn, stripped)
                            self._client.abort(genkey)
                            aborted = True
                            self._log.system(f"Abort sent (genkey={genkey}), bracket-command syntax")
                            inner = stripped.lstrip("[").rstrip("]").strip()
                            obs = (
                                f"[system] Commands use a leading slash, not brackets. "
                                f"Write '/{inner}' instead of '{stripped[:60]}'. "
                                "Reissue the command now."
                            )
                            self._record_obs(turn, stripped, obs, command=False)
                            break

                        elif _is_fake_outgoing_telegram(stripped):
                            _trim_agent_text(turn, stripped)
                            self._client.abort(genkey)
                            aborted = True
                            self._log.system(f"Abort sent (genkey={genkey}), fake outgoing Telegram format")
                            inner = stripped[len("[telegram]"):].strip()
                            obs = (
                                f"[system] Use /telegram to send a message, not [telegram]. "
                                f"Write '/telegram {inner}' to send."
                            )
                            self._record_obs(turn, stripped, obs, command=False)
                            break

                        elif _is_gt_command(stripped, self._frwx):
                            # Model prefixed command with "> " (CMD echo format) — strip and execute
                            actual = stripped[2:]
                            _trim_agent_text(turn, actual)
                            self._client.abort(genkey)
                            aborted = True
                            self._log.system(f"Abort sent (genkey={genkey}), stripped '> ' prefix, executing {actual!r}")
                            self._log.command(actual)
                            result = self._run_command(actual)
                            if actual.lower().startswith("/telegram "):
                                self._pending_tg.clear()
                            prose_lines = 0
                            self._record_obs(turn, actual, result, command=True)
                            break

                        elif is_command_line(completed, self._frwx):
                            _trim_agent_text(turn, stripped)
                            self._client.abort(genkey)
                            aborted = True
                            # Catch hallucinated Telegram headers smuggled inside /telegram body
                            if stripped.lower().startswith("/telegram ") and \
                                    _is_fake_incoming_telegram(stripped[len("/telegram "):].lstrip()):
                                self._log.system(f"Abort sent (genkey={genkey}), /telegram body contains fake header")
                                obs = (
                                    "[system] Your /telegram message started with a fake Telegram header. "
                                    "Send only your own words — never include [name @ Telegram]: in the message. "
                                    "Continue your task."
                                )
                                self._record_obs(turn, stripped, obs, command=False)
                                break
                            self._log.command(stripped)
                            self._log.system(f"Abort sent (genkey={genkey}), executing command")
                            result = self._run_command(stripped)
                            if stripped.lower().startswith("/telegram "):
                                self._pending_tg.clear()
                            prose_lines = 0
                            self._record_obs(turn, stripped, result, command=True)
                            break

                        elif stripped == "<|eoc|>":
                            _trim_agent_text(turn, stripped)
                            self._client.abort(genkey)
                            aborted = True
                            self._log.system(f"Abort sent (genkey={genkey}), hallucinated <|eoc|>")
                            obs = (
                                "[system] The <|eoc|> marker is appended by the harness after each "
                                "dispatched command — do not write it yourself. "
                                "Issue your next command now."
                            )
                            self._record_obs(turn, stripped, obs, command=False)
                            break

                        else:
                            prose_lines += 1
                            if prose_lines >= 10:
                                self._client.abort(genkey)
                                aborted = True
                                obs = (
                                    "[system] You wrote a long response without issuing a command. "
                                    "You are an autonomous agent — issue your next command now."
                                )
                                self._log.system(f"Abort sent (genkey={genkey}), prose monologue ({prose_lines} lines)")
                                self._record_obs(turn, "", obs, command=False)
                                break

                    if aborted:
                        break
                    cur_line = segments[-1]

        except KeyboardInterrupt:
            paused = True
            print(f"\n{_YELLOW}[paused]{_RESET}", flush=True)
            self._log.system(f"Generation paused by user (genkey={genkey})")
            self._client.abort(genkey)
            self._log.system(f"Abort sent (genkey={genkey}), paused")

        self._log.flush_token_buf("AGENT")
        self._log.flush_token_buf("THINK")
        hit_max_tokens = bool(finish_info) and finish_info[0] == "length"
        if hit_max_tokens:
            self._log.system(f"Generation hit token limit (finish_reason=length, genkey={genkey})")
        self._log.system(f"Generation {'paused' if paused else 'aborted' if aborted else 'completed'} (genkey={genkey})")

        # If paused: prompt for a user message, inject it, then return.
        # A second Ctrl-C at the prompt propagates up to run() to quit.
        if paused:
            print(f"{_YELLOW}Message to agent (Enter to resume, Ctrl-C to quit):{_RESET} ", end="", flush=True)
            try:
                user_msg = input().strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{_YELLOW}[agent] Quitting.{_RESET}")
                raise KeyboardInterrupt
            if user_msg:
                # Use the same format as real Telegram messages so the model
                # recognises this as Foxo's voice and responds accordingly.
                obs = f"[Foxo @ Telegram]: {user_msg}"
                print(f"{_GREEN}[obs]{_RESET} {obs}", flush=True)
                self._log.system(f"User injected message: {user_msg!r}")
                self._log.observation(obs)
                turn.observations.append(obs)
                self._history.append(turn)
            return

        # Bug fix: if the stream ended without a trailing newline, the last
        # line never went through the newline-detection block.  Check it now.
        if not aborted and not in_think:
            stripped = cur_line.strip()
            if stripped:
                if self._dispatch.pending is not None:
                    is_batch = self._dispatch.pending.mode in (
                        "appendlines", "edit_old", "edit_new", "patch"
                    )
                    raw_line = cur_line.rstrip()
                    self._log.pending_input(raw_line)
                    result = self._dispatch.handle_pending_input(raw_line)
                    session_ended = self._dispatch.pending is None
                    if session_ended or not is_batch:
                        self._record_obs(turn, stripped, result, command=False)
                        aborted = True
                    # else: batch mode, generation ended mid-entry without
                    # 'done' — we'll inject a reminder below.
                elif _is_fake_incoming_telegram(stripped):
                    _trim_agent_text(turn, stripped)
                    obs = (
                        "[system] You wrote a fake incoming Telegram message. "
                        "Messages from Foxo arrive automatically — never fabricate them. "
                        "Continue your task."
                    )
                    self._record_obs(turn, stripped, obs, command=False)
                    aborted = True
                elif _is_obs_echo(stripped):
                    _trim_agent_text(turn, stripped)
                    obs = (
                        "[system] You wrote what a command response looks like, not a command. "
                        "Issue the command directly with a leading slash, e.g. /mb read <id>."
                    )
                    self._record_obs(turn, stripped, obs, command=False)
                    aborted = True
                elif _is_bracket_command(stripped):
                    _trim_agent_text(turn, stripped)
                    inner = stripped.lstrip("[").rstrip("]").strip()
                    obs = (
                        f"[system] Commands use a leading slash, not brackets. "
                        f"Write '/{inner}' instead of '{stripped[:60]}'. "
                        "Reissue the command now."
                    )
                    self._record_obs(turn, stripped, obs, command=False)
                    aborted = True
                elif _is_fake_outgoing_telegram(stripped):
                    _trim_agent_text(turn, stripped)
                    inner = stripped[len("[telegram]"):].strip()
                    obs = (
                        f"[system] Use /telegram to send a message, not [telegram]. "
                        f"Write '/telegram {inner}' to send."
                    )
                    self._record_obs(turn, stripped, obs, command=False)
                    aborted = True
                elif stripped == "<|eoc|>":
                    _trim_agent_text(turn, stripped)
                    obs = (
                        "[system] The <|eoc|> marker is appended by the harness after each "
                        "dispatched command — do not write it yourself. "
                        "Issue your next command now."
                    )
                    self._record_obs(turn, stripped, obs, command=False)
                    aborted = True
                elif _is_gt_command(stripped, self._frwx):
                    actual = stripped[2:]
                    _trim_agent_text(turn, actual)
                    self._log.command(actual)
                    result = self._run_command(actual)
                    if actual.lower().startswith("/telegram "):
                        self._pending_tg.clear()
                    self._record_obs(turn, actual, result, command=True)
                    aborted = True
                elif is_command_line(cur_line, self._frwx):
                    _trim_agent_text(turn, stripped)
                    if stripped.lower().startswith("/telegram ") and \
                            _is_fake_incoming_telegram(stripped[len("/telegram "):].lstrip()):
                        obs = (
                            "[system] Your /telegram message started with a fake Telegram header. "
                            "Send only your own words — never include [name @ Telegram]: in the message. "
                            "Continue your task."
                        )
                        self._record_obs(turn, stripped, obs, command=False)
                    else:
                        self._log.command(stripped)
                        result = self._run_command(stripped)
                        if stripped.lower().startswith("/telegram "):
                            self._pending_tg.clear()
                        self._record_obs(turn, stripped, result, command=True)
                    aborted = True

        # If a batch session is still open after generation ends (token limit
        # hit mid-entry), remind the model where it left off and what to type.
        if not aborted and self._dispatch.pending is not None:
            mode = self._dispatch.pending.mode
            fp   = self._dispatch.pending.file_path
            if mode == "appendlines":
                obs = (
                    f"[appendlines:{fp}] Generation ended before 'done' (token limit). "
                    "Continue writing remaining lines, then type 'done' alone."
                )
            elif mode == "edit_old":
                obs = (
                    f"[edit:{fp}] Generation ended before '---' (token limit). "
                    "Continue writing the old text, then type '---' alone to separate."
                )
            elif mode == "edit_new":
                obs = (
                    f"[edit:{fp}] Generation ended before 'done' (token limit). "
                    "Continue writing replacement text, then type 'done' alone."
                )
            elif mode == "patch":
                obs = (
                    "[patch] Generation ended before '*** End Patch' (token limit). "
                    "Continue the patch and finish with '*** End Patch'."
                )
            else:
                obs = None
            if obs is not None:
                print(f"\n{_GREEN}[obs]{_RESET} {obs}", flush=True)
                self._log.observation(obs)
                turn.observations.append(obs)

        # _record_obs already appended the turn when a command was found.
        # Only append here for clean (no-command) completions.
        if not aborted:
            # Strip stray partial-command lines from the tail of agent_text.
            # These appear when max_tokens cuts the model off mid-command
            # (e.g. it starts typing "/cmem r 1" but only the first BPE
            # sub-token "/cm" arrives before the budget runs out).  The
            # fragment isn't a valid command so it was never dispatched or
            # trimmed, but leaving it in the assistant message causes the
            # model to echo it on every subsequent turn.
            lines = turn.agent_text.splitlines()
            dropped = []
            while lines and lines[-1].lstrip().startswith("/") and not is_command_line(lines[-1], self._frwx):
                partial = lines[-1].strip()
                self._log.system(f"Dropped stray partial command from agent_text: {partial!r}")
                dropped.append(partial)
                lines.pop()
            turn.agent_text = "\n".join(lines)

            # Feed back an error for each dropped fragment so the model knows
            # the command was never executed and must be reissued in full.
            for partial in dropped:
                obs = (
                    f"[error] Command cut off mid-token: {partial!r}. "
                    "Generation hit the token limit before the command was complete. "
                    "Reissue the full command on its own line."
                )
                print(f"\n{_GREEN}[obs]{_RESET} {obs}", flush=True)
                self._log.observation(obs)
                turn.observations.append(obs)

            # If the model produced content but no command, inject a nudge.
            if not turn.observations:
                if hit_max_tokens:
                    # Think block exhausted the token budget before a command
                    # could be issued. Explicitly tell the model to skip think
                    # and write the command bare.
                    obs = (
                        "[system] Your response was cut off by the token limit "
                        "before you issued a command. Your think block was too long. "
                        "On your next response write ONLY the command — no <think> block, "
                        "no prose, just the command line."
                    )
                    self._log.system("Generation hit token limit without command — hard nudge")
                elif turn.agent_text.strip():
                    obs = (
                        "[system] Response completed without issuing a command. "
                        "Issue your next command now."
                    )
                    self._log.system("Generation ended with prose but no command — nudging")
                else:
                    obs = None

                if obs:
                    print(f"\n{_GREEN}[obs]{_RESET} {obs}", flush=True)
                    self._log.observation(obs)
                    turn.observations.append(obs)

            if turn.agent_text.strip() or turn.observations:
                self._history.append(turn)

    # ------------------------------------------------------------------
    # Command dispatch with timeout and background-job support
    # ------------------------------------------------------------------

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
            job = self._job_mgr.start(inner, lambda c=inner: self._dispatch.dispatch(c) or "")
            return (
                f"[bg] Job #{job.id} started: {inner[:70]}\n"
                f"Use /fg {job.id} to collect the result, or /jobs to list all jobs."
            )

        # Regular command — run in thread so a hang never blocks the loop.
        job = self._job_mgr.start(cmd, lambda c=cmd: self._dispatch.dispatch(c) or "")
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
        """Record an observation and commit the turn to history."""
        if command:
            loop_warn = self._loop_detector.record(text, result)
            if loop_warn:
                result = result + "\n" + loop_warn
                self._log.system(f"Loop detector fired: {loop_warn[:120]}")
            # EOC marker: stored in agent_text so training data shows every
            # dispatched command terminated with <|eoc|>, teaching the model
            # that commands are one-liners that end the turn.
            turn.agent_text = turn.agent_text.rstrip("\n") + "\n<|eoc|>"
        prefix = "> " if command else ""
        print(f"\n{_GREEN}[obs]{_RESET} {result}", flush=True)
        self._log.observation(result)
        turn.observations.append(f"{prefix}{text}\n{result}")
        # Single point of history append — _step() must NOT append again after this.
        self._history.append(turn)
        # Incremental save — a crash never loses the whole session.
        self._save_training_data()

    # ------------------------------------------------------------------
    # Training data export
    # ------------------------------------------------------------------

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
        messages.append({"role": "user", "content": "Begin. Read your task file first."})

        good_turns = 0
        for turn in self._history:
            is_correction = any(
                obs.startswith(p) for obs in turn.observations for p in self._BAD_OBS_PREFIXES
            )
            if is_correction or not turn.agent_text.strip():
                continue
            think = turn.think_text.strip()
            agent = turn.agent_text.rstrip()
            content = f"<think>\n{think}\n</think>\n{agent}" if think else agent
            messages.append({"role": "assistant", "content": content})
            for obs in turn.observations:
                messages.append({"role": "user", "content": obs})
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

            # Think block open/close
            if not in_think and stripped == "<think>":
                in_think = True
                continue

            if in_think:
                if stripped == "</think>":
                    in_think = False
                    print(f"{_CYAN}[teleop]{_RESET} ", end="", flush=True)
                else:
                    turn.think_text += line + "\n"
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

            result = self._run_command(stripped)
            if stripped.lower().startswith("/telegram "):
                self._pending_tg.clear()
            self._record_obs(turn, stripped, result, command=True)
            return

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_messages(self, compress: bool = False) -> list[dict]:
        """
        Build the chat message list from current state.

        Cache strategy for KoboldCPP SmartCache:
          - System prompt: 100% static → always cached.
          - History observation messages: stored clean (no memory/timestamp
            mixed in) → identical between turns → always cached.
          - Memory + timestamp: injected as a DEDICATED final user message
            that is never stored in history.  When memory changes, only this
            one message (~300–800 tokens) needs reprocessing.  When memory is
            stable, the slot matches through the entire history and only new
            content beyond the saved slot is processed.
        """
        shell_section = (
            "\n\n════════════════════════════════════════\n"
            "SHELL  (full system access — --frwx mode)\n"
            "  $ <command>    run as current user (30 s timeout, stdin closed)\n"
            "  # <command>    run as root via sudo (same timeout)\n"
            "  File commands (/read, /edit, etc.) have no path restrictions.\n"
            "  Use <think> before any destructive or irreversible command.\n"
            "════════════════════════════════════════"
        ) if self._frwx else ""

        system_content = (
            SYSTEM_PROMPT
            + shell_section
            + f"\n\n════════════════════════════════════════\n"
            f"Your task is in {TASK_FILE}. Read it first with /read {TASK_FILE}.\n"
            f"════════════════════════════════════════"
        )

        messages: list[dict] = [{"role": "system", "content": system_content}]

        if not self._history:
            messages.append({"role": "user", "content": "Begin. Read your task file first."})
        else:
            for turn in self._history:
                if turn.agent_text.strip():
                    messages.append({"role": "assistant", "content": turn.agent_text.rstrip()})
                for obs in turn.observations:
                    content = _compress_obs_for_history(obs) if compress else obs
                    messages.append({"role": "user", "content": content})

            if messages[-1]["role"] == "assistant":
                messages.append({"role": "user", "content": "Continue your task."})

        # Pending Telegram messages (ephemeral, not stored in history)
        for tg in self._pending_tg:
            messages.append({"role": "user", "content": tg})

        # Dedicated memory + timestamp message — always the final entry, never
        # mixed into observation history.  Keeping it separate means history
        # messages are stable across turns, maximising SmartCache prefix hits.
        cmem         = self._cmem.render()
        now          = datetime.now().strftime("%d %b %Y, %H:00")  # hourly — reduces cache invalidations
        cmem_display = cmem.strip() or "(empty)"

        mem_parts = [f"════ YOUR SCRATCHPAD (notes you wrote to yourself) ════\n{cmem_display}", f"[{now}]"]

        messages.append({"role": "user", "content": "\n\n".join(mem_parts)})

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
            cmds = [l.strip() for l in turn.agent_text.splitlines() if l.strip()]
            if cmds:
                lines.append("Agent: " + " | ".join(cmds))
            for obs in turn.observations:
                first = obs.split("\n")[0][:200]
                lines.append("Result: " + first)

        transcript = "\n".join(lines)

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a concise session summarizer for an autonomous AI agent. "
                    "Produce a single compact paragraph (max 200 words) that preserves: "
                    "what tasks were accomplished, key facts discovered, any instructions "
                    "from Foxo, and what is still pending. Write in past tense."
                ),
            },
            {
                "role": "user",
                "content": f"Summarize this agent session excerpt:\n\n{transcript}",
            },
        ]

        try:
            summary = self._client.chat_complete_sync(prompt, max_tokens=300, timeout=90)
        except Exception as exc:
            self._log.system(f"Compaction failed: {exc}")
            return False

        if not summary:
            self._log.system("Compaction returned empty summary — skipping.")
            return False

        compact_turn = Turn()
        compact_turn.observations.append(
            f"[Compacted summary of {n} earlier turns]\n{summary}"
        )
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

        # 1. Uncompressed — ideal path
        messages = self._build_messages(compress=False)
        if _fits(messages):
            return messages

        # 2. Compressed observations
        messages = self._build_messages(compress=True)
        if _fits(messages):
            return messages

        # 3 & 4. Context still too large: alternate compaction and hard-drop
        while len(self._history) > 1:
            if _fits(messages):
                break

            # Try LLM compaction first if we have enough turns
            if len(self._history) >= _COMPACT_TRIGGER:
                if self._compact_history(_COMPACT_BATCH):
                    messages = self._build_messages(compress=True)
                    continue

            # Fall back to hard-drop of oldest turn
            dropped = self._history.pop(0)
            msg = f"Context full — dropped oldest turn ({len(dropped.agent_text)} chars)."
            print(f"{_YELLOW}[agent] {msg}{_RESET}", flush=True)
            self._log.system(msg)
            messages = self._build_messages(compress=True)

        return messages


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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
