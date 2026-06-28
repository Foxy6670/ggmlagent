#!/usr/bin/env python3
"""
Generate codelab training data.

Simulates a correct agent session through the Rust session-analyzer task.
Executes real cargo/shell commands so observations are authentic.
Writes a single JSONL record to stdout (or --out file).

Usage:
  python3 generate_codelab_data.py
  python3 generate_codelab_data.py --out codelab_training.jsonl
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

REPO = Path(__file__).parent
LOGS_DIR = REPO / "logs"
TASK_FILE = REPO / "task_codelab.md"

sys.path.insert(0, str(REPO))
from config import SYSTEM_PROMPT

# ── Workspace ─────────────────────────────────────────────────────────────────

WS: Path = None   # set in main

def run(cmd, cwd=None):
    r = subprocess.run(
        cmd, shell=True, cwd=str(cwd or WS),
        capture_output=True, text=True, timeout=120
    )
    return (r.stdout + r.stderr).strip()

def run_split(cmd, cwd=None):
    """Return (combined_for_obs, stdout_only) — useful when stdout is the report."""
    r = subprocess.run(
        cmd, shell=True, cwd=str(cwd or WS),
        capture_output=True, text=True, timeout=120
    )
    combined = (r.stderr + r.stdout).strip()  # stderr first (cargo noise), then output
    return combined, r.stdout.strip()

def write_ws(rel, content):
    p = WS / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

# ── Message helpers ───────────────────────────────────────────────────────────

msgs = []
_cmem: dict[int, str] = {}   # line → text, reset per session

def sys_msg(c): msgs.append({"role": "system", "content": c})

def _cmem_header() -> str:
    if not _cmem:
        return ""
    lines = "\n".join(f"  {k}: {v}" for k, v in sorted(_cmem.items()))
    return f"YOUR SCRATCHPAD:\n{lines}\n\n"

def usr(c, with_cmem=False):
    """Append a command result as a role:"tool" message (Qwen3 wraps it in <tool_response>).

    The leading "> /cmd" echo line the fmt_* helpers prepend is stripped: in the
    tool-call format the command lives in the assistant turn's <tool_call>, so the
    result is injected bare.  with_cmem prefixes the current scratchpad state.
    """
    if c.startswith("> ") and "\n" in c:
        c = c.split("\n", 1)[1]
    content = (_cmem_header() + c) if with_cmem else c
    msgs.append({"role": "tool", "content": content})

def _split_cmd_body(cmd_text):
    """Split an old-style cmd_text blob into (command, body) for the run_command tool.

      "$ shell ..."        -> ("$ shell ...", "")     shell stays inline
      "/patch\\n<hunk>"     -> ("/patch", "<hunk>")    multiline payload -> body
      "/telegram <multi>"  -> ("/telegram", "<multi>") when the message spans lines
      everything else      -> (cmd_text, "")           single-line slash command
    """
    cmd_text = cmd_text.strip()
    first, _, rest = cmd_text.partition("\n")
    verb = first.split(None, 1)[0] if first.split() else ""
    if verb == "/patch":
        return "/patch", rest.strip("\n")
    if verb == "/telegram" and rest:
        return "/telegram", cmd_text[len("/telegram"):].strip()
    return cmd_text, ""

def ast(think, narrate, cmd_text):
    """Append an assistant turn: optional <think>, narration, then a native <tool_call>."""
    think_part = f"<think>\n{think.strip()}\n</think>\n" if think.strip() else ""
    command, body = _split_cmd_body(cmd_text)
    arguments = {"command": command}
    if body:
        arguments["body"] = body
    call = json.dumps({"name": "run_command", "arguments": arguments}, ensure_ascii=False)
    narrate_part = f"{narrate.strip()}\n" if narrate.strip() else ""
    msgs.append({"role": "assistant", "content":
        f"{think_part}{narrate_part}<tool_call>\n{call}\n</tool_call>"})

def cmem_write(line: int, text: str) -> str:
    """Simulate /cmem w <line> <text> and return the observation."""
    _cmem[line] = text
    return f"> /cmem w {line} {text}\n[scratchpad line {line} set]"

def cmem_del(line: int) -> str:
    _cmem.pop(line, None)
    return f"> /cmem d {line}\n[scratchpad line {line} deleted]"

# ── Observation formatters ────────────────────────────────────────────────────

def fmt_read(path_label, text, page=1):
    lines = text.splitlines(keepends=True)
    total = max(1, -(-len(lines) // 100))
    start = (page - 1) * 100
    chunk = lines[start : start + 100]
    numbered = "".join(f"{start + i + 1:4d}: {l}" for i, l in enumerate(chunk))
    return f"> /read {path_label}\n[file:{path_label} page {page}/{total}]\n{numbered.rstrip()}"

def fmt_shell(cmd, out):
    body = out if out else "(no output)"
    return f"> $ {cmd}\n{body}"

def fmt_patch(msg="[patch] Applied."):
    return f"> /patch\n{msg}"

def fmt_patch_err(bad_path):
    """Simulate the improved patch error message for a wrong path."""
    return (
        f"> /patch\n"
        f"[patch] Error: file/directory specified in patch header does not exist"
        f" — [Errno 2] No such file or directory: 'codelab/{bad_path}'\n"
        f"[patch] Note: patch paths resolve relative to the harness working"
        f" directory (codelab/), not the shell CWD."
    )

def fmt_cmem_r_error():
    return ("> /cmem r\n"
            "[cmem] Context memory is already visible in your prompt — "
            "you do not need to read it. Use /cmem w <line> <text> to write "
            "or /cmem d <line> to delete.")

def sys_ctx(pct: int, cwd: str = "~/ggml_codelab/session-analyzer") -> None:
    """Inject the ephemeral per-turn system message with context percentage."""
    msgs.append({"role": "system", "content":
        f"[system: 22 Jun 2026, 21:15 | cwd: {cwd} | context: {pct}% used]"})

def fmt_pmem_r(body="[memory.md: empty]"):
    return f"> /pmem r\n{body}"

def fmt_pmem_w():
    return "> /pmem w\n[memory saved]"

def fmt_telegram(text):
    return f"> /telegram {text}\n[telegram] Message sent to Foxo."

def fmt_compacted(summary=""):
    """Simulate a mid-session compaction notice injected by the harness."""
    body = summary or (
        "Previous context condensed. Your task: build a Rust session-log "
        "analyzer in session-analyzer/. Check your scratchpad (shown above) "
        "for current progress, then re-read task.md before continuing."
    )
    return f"[Compacted summary: {body}]"

# ── Rust source templates ─────────────────────────────────────────────────────

CARGO_TOML = dedent("""\
    [package]
    name = "session-analyzer"
    version = "0.1.0"
    edition = "2021"

    [dependencies]
""")

PARSER_RS = dedent("""\
    use std::fs;
    use std::path::Path;

    #[derive(Default)]
    pub struct Turn {
        pub think: String,
        pub agent: String,
        pub cmd:   String,
        pub obs:   String,
    }

    pub struct ParsedSession {
        pub turns:        Vec<Turn>,
        pub duration_secs: u64,
    }

    fn ts_to_secs(s: &str) -> u64 {
        let p: Vec<&str> = s.split(':').collect();
        if p.len() < 3 { return 0; }
        let h: u64 = p[0].parse().unwrap_or(0);
        let m: u64 = p[1].parse().unwrap_or(0);
        let sec: u64 = p[2].split('.').next().unwrap_or("0").parse().unwrap_or(0);
        h * 3600 + m * 60 + sec
    }

    fn tag_content(line: &str) -> Option<(String, String, Option<u64>)> {
        let bracket = line.find('[')?;
        let ts_secs = if bracket >= 8 { Some(ts_to_secs(&line[..8])) } else { None };
        let after   = &line[bracket + 1..];
        let close   = after.find(']')?;
        let tag     = after[..close].trim().to_uppercase();
        if !tag.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return None;
        }
        let content = after[close + 1..].trim().to_string();
        Some((tag, content, ts_secs))
    }

    pub fn parse_log(path: &Path) -> ParsedSession {
        let text = match fs::read_to_string(path) {
            Ok(t) => t,
            Err(_) => return ParsedSession { turns: Vec::new(), duration_secs: 0 },
        };

        let mut turns:    Vec<Turn> = Vec::new();
        let mut current:  Turn      = Turn::default();
        let mut in_think: bool      = false;
        let mut first_ts: Option<u64> = None;
        let mut last_ts:  Option<u64> = None;

        for line in text.lines() {
            if let Some((tag, rest, ts)) = tag_content(line) {
                if let Some(t) = ts {
                    if first_ts.is_none() { first_ts = Some(t); }
                    last_ts = Some(t);
                }
                match tag.as_str() {
                    "THINK" => match rest.as_str() {
                        "<think>"  => in_think = true,
                        "</think>" => in_think = false,
                        _ if in_think => { current.think += &rest; current.think += "\\n"; }
                        _ => {}
                    },
                    "AGENT" => { current.agent += &rest; current.agent += "\\n"; }
                    "CMD"   => { current.cmd = rest; }
                    "OBS"   => { current.obs  += &rest; current.obs  += "\\n"; }
                    "SYS" if rest.contains("Generation started") => {
                        if !current.agent.is_empty() || !current.cmd.is_empty() {
                            turns.push(std::mem::take(&mut current));
                        }
                    }
                    _ => {}
                }
            } else if in_think {
                current.think += line;
                current.think += "\\n";
            }
        }
        if !current.agent.is_empty() || !current.cmd.is_empty() {
            turns.push(current);
        }
        let duration_secs = match (first_ts, last_ts) {
            (Some(f), Some(l)) if l > f => l - f,
            _ => 0,
        };
        ParsedSession { turns, duration_secs }
    }
""")

STATS_RS = dedent("""\
    use std::collections::HashMap;
    use crate::parser::ParsedSession;

    pub struct SessionStats {
        pub total_turns:       usize,
        pub commands_issued:   usize,
        pub think_lines:       usize,
        pub third_person_hits: usize,
        pub top_command:       String,
        pub duration_secs:     u64,
    }

    fn strip_block_prefix(cmd: &str) -> &str {
        // Strip "[block] " or "[block-eos] " harness prefix
        if cmd.starts_with('[') {
            if let Some(close) = cmd.find(']') {
                return cmd[close + 1..].trim();
            }
        }
        cmd.trim()
    }

    pub fn compute(session: &ParsedSession) -> SessionStats {
        let turns = &session.turns;
        let mut commands_issued   = 0usize;
        let mut think_lines       = 0usize;
        let mut third_person_hits = 0usize;
        let mut counts: HashMap<String, usize> = HashMap::new();

        for t in turns {
            if !t.cmd.is_empty() {
                commands_issued += 1;
                let bare = strip_block_prefix(&t.cmd);
                let prefix = bare.split_whitespace().next().unwrap_or("").to_string();
                if !prefix.is_empty() {
                    *counts.entry(prefix).or_insert(0) += 1;
                }
            }
            think_lines       += t.think.lines().count();
            third_person_hits += t.think.to_lowercase().matches("the user").count();
        }

        let (top_cmd, top_n) = counts.iter()
            .max_by_key(|(_, v)| *v)
            .map(|(k, v)| (k.clone(), *v))
            .unwrap_or_else(|| (String::new(), 0));

        let top_command = if top_cmd.is_empty() {
            "none".to_string()
        } else {
            format!("{} ({})", top_cmd, top_n)
        };

        SessionStats {
            total_turns: turns.len(),
            commands_issued,
            think_lines,
            third_person_hits,
            top_command,
            duration_secs: session.duration_secs,
        }
    }
""")

MAIN_RS_SMOKE = dedent("""\
    mod parser;

    fn main() {
        let args: Vec<String> = std::env::args().collect();
        if args.len() < 2 {
            eprintln!("Usage: session-analyzer <log-file>...");
            std::process::exit(1);
        }
        // Smoke test: parse and print turn count
        let session = parser::parse_log(std::path::Path::new(&args[1]));
        println!("Turns parsed: {}", session.turns.len());
    }
""")

MAIN_RS_FULL = dedent("""\
    mod parser;
    mod stats;

    use std::path::Path;

    fn fmt_duration(secs: u64) -> String {
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        if h > 0 { format!("{}h {:02}m", h, m) } else { format!("{}m", m) }
    }

    fn main() {
        let args: Vec<String> = std::env::args().collect();
        if args.len() < 2 {
            eprintln!("Usage: session-analyzer <log-file>...");
            std::process::exit(1);
        }
        for path_str in &args[1..] {
            let path = Path::new(path_str);
            let session = parser::parse_log(path);
            let s = stats::compute(&session);
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            println!("{}  [{}]", name, fmt_duration(s.duration_secs));
            println!("  Turns      : {}", s.total_turns);
            println!("  Commands   : {}", s.commands_issued);
            println!("  Think lines: {}", s.think_lines);
            println!("  3rd-person : {} hits", s.third_person_hits);
            println!("  Top command: {}", s.top_command);
            println!();
        }
    }
""")

# ── Patch text helpers ────────────────────────────────────────────────────────

def patch_add(rel_path, content):
    """Format a /patch *** Add File block."""
    lines = "\n".join("+" + l for l in content.splitlines())
    return f"/patch\n*** Begin Patch\n*** Add File: {rel_path}\n{lines}\n*** End Patch"

def patch_update(rel_path, anchor, old_lines, new_lines):
    """Format a /patch *** Update File hunk."""
    removes = "\n".join("-" + l for l in old_lines)
    adds    = "\n".join("+" + l for l in new_lines)
    return (f"/patch\n*** Begin Patch\n*** Update File: {rel_path}\n"
            f"@@ {anchor}\n{removes}\n{adds}\n*** End Patch")

# ── Buggy parser variant (missing #[derive(Default)]) ─────────────────────────

PARSER_RS_BUGGY = PARSER_RS.replace("#[derive(Default)]\n", "")

# ── Session builder helpers ───────────────────────────────────────────────────

def _session_start(task_text):
    """Shared turns 1-4: read task, pmem r, cargo new. Returns (sa, smoke_log)."""
    sys_msg(SYSTEM_PROMPT)
    sys_msg("Begin. Read your task file first.")

    ast(
        think="I need to read task.md to understand what I'm supposed to build.\n",
        narrate="I'll read the task file first.",
        cmd_text="/read task.md",
    )
    usr(fmt_read("task.md", task_text, page=1))

    task_lines = task_text.splitlines(keepends=True)
    if max(1, -(-len(task_lines) // 100)) > 1:
        ast(
            think="That was page 1 of 2. I need to read the rest before I start.\n",
            narrate="There's a second page — I'll read it.",
            cmd_text="/read task.md",
        )
        usr(fmt_read("task.md", task_text, page=2))

    ast(
        think="I should check my persistent memory for any prior notes about this task.\n",
        narrate="I'll check persistent memory for any prior context.",
        cmd_text="/pmem r",
    )
    usr(fmt_pmem_r())

    cargo_init_cmd = "cargo new session-analyzer && cd session-analyzer && ls src/"
    cargo_init_out = run("cargo new session-analyzer && ls session-analyzer/src/")
    ast(
        think="""\
Step 0 says to chain all three commands on one line so the harness
runs them as a single shell invocation. cargo new creates the project,
cd moves into it, ls src/ confirms main.rs is there.
""",
        narrate="I'll initialize the Rust project with all three commands chained.",
        cmd_text=f"$ {cargo_init_cmd}",
    )
    usr(fmt_shell(cargo_init_cmd, cargo_init_out))

    sa = WS / "session-analyzer"

    log_files = sorted(LOGS_DIR.glob("*.log"))
    if not log_files:
        raise RuntimeError("No log files found in logs/")
    smoke_log = log_files[-1]

    return sa, smoke_log


def _session_end(sa, smoke_log, report_content):
    """Shared turns: stats.rs, full main.rs, final build+run, report, pmem, telegram."""

    # write stats.rs
    stats_patch = patch_add("session-analyzer/src/stats.rs", STATS_RS)
    write_ws("session-analyzer/src/stats.rs", STATS_RS)
    ast(
        think="""\
The smoke test output looks right. Now I'll write stats.rs.
It uses crate::parser types, so I need the struct names to match.
""",
        narrate="Writing stats.rs.",
        cmd_text=stats_patch,
    )
    usr(fmt_patch())

    # update main.rs to full version
    full_main_patch = patch_update(
        "session-analyzer/src/main.rs",
        "mod parser",
        MAIN_RS_SMOKE.splitlines(),
        MAIN_RS_FULL.splitlines(),
    )
    write_ws("session-analyzer/src/main.rs", MAIN_RS_FULL)
    ast(
        think="""\
Now I'll replace the smoke-test main.rs with the full version that declares
both mod parser and mod stats, adds fmt_duration, and loops over all
log files to print the formatted report.
""",
        narrate="Replacing main.rs with the full implementation.",
        cmd_text=full_main_patch,
    )
    usr(fmt_patch())

    # final build
    build_out = run("cargo build 2>&1", cwd=sa)
    ast(
        think="Let me build the complete implementation before running against all logs.\n",
        narrate="Building the full implementation.",
        cmd_text="$ cargo build",
    )
    usr(fmt_shell("cargo build", build_out))

    # run against all logs
    all_logs_cmd = "cargo run -- ../logs/*.log"
    all_logs_obs, report_content = run_split(all_logs_cmd, cwd=sa)
    ast(
        think="""\
The build succeeded. I'll run against all logs now with the glob pattern
the task specified. The shell will expand ../logs/*.log to all .log files.
""",
        narrate="Running against all log files to produce the full report.",
        cmd_text=f"$ {all_logs_cmd}",
    )
    usr(fmt_shell(all_logs_cmd, all_logs_obs))

    # write analysis_report.md
    report_patch = patch_add(
        "analysis_report.md",
        f"# Session Analysis Report\n\n```\n{report_content}\n```\n",
    )
    ast(
        think="""\
The report looks complete. The task says to copy it into
../analysis_report.md using /patch. The patch path is relative to the
harness working directory (codelab/), so the correct path is
analysis_report.md — one level up from where I'm working in session-analyzer/.
""",
        narrate="Writing the report to analysis_report.md.",
        cmd_text=report_patch,
    )
    usr(fmt_patch())

    # pmem w
    pmem_text = "task complete — built session-analyzer (Rust), parsed session logs, wrote analysis_report.md"
    ast(
        think="Task is done. I'll save a memory note then send the report to Foxo.\n",
        narrate="Saving a completion note to persistent memory.",
        cmd_text=f"/pmem w {pmem_text}",
    )
    usr(fmt_pmem_w())

    # telegram
    tg_msg = f"Session analyzer done. Report:\n\n{report_content}"
    ast(
        think="I'll send the report to Foxo now. Task is complete.\n",
        narrate="Sending the report to Foxo via Telegram.",
        cmd_text=f"/telegram {tg_msg}",
    )
    usr(fmt_telegram(tg_msg))


# ── Variant: clean ────────────────────────────────────────────────────────────

def build_session_clean(task_text, sa, smoke_log):
    # write parser.rs (correct)
    write_ws("session-analyzer/src/parser.rs", PARSER_RS)
    ast(
        think="""\
Now I'll write src/parser.rs. The patch path must be relative to the
harness working directory (codelab/), not my shell CWD. So the correct
path is session-analyzer/src/parser.rs.

The parser needs to:
- Read a log file line by line
- Detect [SYS] Generation started to split turns
- Collect [THINK], [AGENT], [CMD], [OBS] content per turn
- Track first/last timestamps for duration
""",
        narrate="I'll write src/parser.rs using /patch.",
        cmd_text=patch_add("session-analyzer/src/parser.rs", PARSER_RS),
    )
    usr(fmt_patch())

    # write smoke main.rs
    write_ws("session-analyzer/src/main.rs", MAIN_RS_SMOKE)
    ast(
        think="""\
Before building I need to declare the parser module in main.rs,
otherwise cargo won't compile parser.rs at all.
I'll replace the default main with a smoke test that calls parse_log.
""",
        narrate="I'll replace main.rs with a smoke test that uses the parser.",
        cmd_text=patch_update(
            "session-analyzer/src/main.rs", "fn main()",
            ["fn main() {", '    println!("Hello, world!");', "}"],
            MAIN_RS_SMOKE.splitlines(),
        ),
    )
    usr(fmt_patch())

    # first build
    build_out = run("cargo build 2>&1", cwd=sa)
    ast(
        think="Both parser.rs and the smoke-test main.rs are in place. Let me build.\n",
        narrate="Building to check parser.rs compiles.",
        cmd_text="$ cargo build",
    )
    usr(fmt_shell("cargo build", build_out))

    # smoke run
    smoke_cmd = f"cargo run -- ../logs/{smoke_log.name}"
    smoke_out = run(f"{smoke_cmd} 2>&1", cwd=sa)
    ast(
        think="""\
I'll run the smoke test against the most recent log file.
../logs/ is relative to session-analyzer/, pointing to codelab/logs/.
I want to see a plausible turn count.
""",
        narrate=f"Running smoke test against {smoke_log.name}.",
        cmd_text=f"$ {smoke_cmd}",
    )
    usr(fmt_shell(smoke_cmd, smoke_out))

    _session_end(sa, smoke_log, report_content=None)


# ── Variant: build-error (teaches Default trait error recovery) ───────────────

def build_session_build_error(task_text, sa, smoke_log):
    # write BUGGY parser.rs (missing #[derive(Default)])
    write_ws("session-analyzer/src/parser.rs", PARSER_RS_BUGGY)
    ast(
        think="""\
Now I'll write src/parser.rs. The patch path must be relative to the
harness working directory (codelab/), not my shell CWD. So the correct
path is session-analyzer/src/parser.rs.

The parser needs to:
- Read a log file line by line
- Detect [SYS] Generation started to split turns
- Collect [THINK], [AGENT], [CMD], [OBS] content per turn
- Track first/last timestamps for duration
""",
        narrate="I'll write src/parser.rs using /patch.",
        cmd_text=patch_add("session-analyzer/src/parser.rs", PARSER_RS_BUGGY),
    )
    usr(fmt_patch())

    # write smoke main.rs
    write_ws("session-analyzer/src/main.rs", MAIN_RS_SMOKE)
    ast(
        think="""\
Before building I need to declare the parser module in main.rs,
otherwise cargo won't compile parser.rs at all.
""",
        narrate="I'll replace main.rs with a smoke test that uses the parser.",
        cmd_text=patch_update(
            "session-analyzer/src/main.rs", "fn main()",
            ["fn main() {", '    println!("Hello, world!");', "}"],
            MAIN_RS_SMOKE.splitlines(),
        ),
    )
    usr(fmt_patch())

    # first build — this WILL fail (no Default on Turn)
    build_err_out = run("cargo build 2>&1", cwd=sa)
    ast(
        think="Both files are in place. Let me build to check for compile errors.\n",
        narrate="Building to check parser.rs compiles.",
        cmd_text="$ cargo build",
    )
    usr(fmt_shell("cargo build", build_err_out))

    # agent sees the error and fixes it
    parser_fix_patch = patch_update(
        "session-analyzer/src/parser.rs",
        "pub struct Turn",
        ["pub struct Turn {"],
        ["#[derive(Default)]", "pub struct Turn {"],
    )
    write_ws("session-analyzer/src/parser.rs", PARSER_RS)
    ast(
        think="""\
The build failed. Reading the error: Turn::default() is called (via std::mem::take)
but Turn doesn't implement the Default trait. The fix is straightforward:
add #[derive(Default)] to the Turn struct. I'll patch that in now.
""",
        narrate="The build failed — Turn needs #[derive(Default)] for std::mem::take. I'll patch it.",
        cmd_text=parser_fix_patch,
    )
    usr(fmt_patch())

    # rebuild — should succeed now
    build_ok_out = run("cargo build 2>&1", cwd=sa)
    ast(
        think="I added the derive attribute. Let me rebuild to confirm it compiles now.\n",
        narrate="Rebuilding after adding #[derive(Default)].",
        cmd_text="$ cargo build",
    )
    usr(fmt_shell("cargo build", build_ok_out))

    # smoke run
    smoke_cmd = f"cargo run -- ../logs/{smoke_log.name}"
    smoke_out = run(f"{smoke_cmd} 2>&1", cwd=sa)
    ast(
        think="""\
Build succeeded. I'll run the smoke test to verify the parser works
end-to-end before adding the stats module.
""",
        narrate=f"Running smoke test against {smoke_log.name}.",
        cmd_text=f"$ {smoke_cmd}",
    )
    usr(fmt_shell(smoke_cmd, smoke_out))

    _session_end(sa, smoke_log, report_content=None)


# ── Variant: cmem-tracking (teaches correct scratchpad usage + compaction recovery) ──

def build_session_cmem(task_text, sa, smoke_log):
    # write parser.rs + record CWD and patch root in cmem
    write_ws("session-analyzer/src/parser.rs", PARSER_RS)
    ast(
        think="""\
Now I'll write parser.rs. The patch path must be relative to the harness
working directory (codelab/), not my shell CWD (codelab/session-analyzer/).
I'll record this in my scratchpad so I don't make the path mistake later.
""",
        narrate="Writing src/parser.rs. I'll note the patch path root in my scratchpad.",
        cmd_text=patch_add("session-analyzer/src/parser.rs", PARSER_RS),
    )
    usr(fmt_patch())

    # record patch path root and CWD in cmem
    ast(
        think="""\
I should write down the key path facts now while they're fresh.
cmem line 1: current step. Line 2: slow-state facts I'll need in every patch.
""",
        narrate="Recording path facts in my scratchpad before I forget.",
        cmd_text="/cmem w 1 step 1/6 done — parser.rs written",
    )
    usr(cmem_write(1, "step 1/6 done — parser.rs written"), with_cmem=False)

    ast(
        think="Line 2 for slow-state: patch paths are always from codelab/, CWD is session-analyzer/ after cargo new.\n",
        narrate="Noting the harness working directory for patch paths.",
        cmd_text="/cmem w 2 patch paths: codelab/ root. shell CWD: codelab/session-analyzer/",
    )
    usr(cmem_write(2, "patch paths: codelab/ root. shell CWD: codelab/session-analyzer/"), with_cmem=False)

    # write smoke main.rs
    write_ws("session-analyzer/src/main.rs", MAIN_RS_SMOKE)
    ast(
        think="Now declare the parser module in main.rs and write a smoke test.\n",
        narrate="Replacing main.rs with the smoke test.",
        cmd_text=patch_update(
            "session-analyzer/src/main.rs", "fn main()",
            ["fn main() {", '    println!("Hello, world!");', "}"],
            MAIN_RS_SMOKE.splitlines(),
        ),
    )
    usr(fmt_patch(), with_cmem=True)

    # update cmem step
    ast(
        think="Smoke main.rs is in place. Update scratchpad step counter.\n",
        narrate="Updating scratchpad — both source files written.",
        cmd_text="/cmem w 1 step 2/6 done — parser.rs + smoke main.rs written",
    )
    usr(cmem_write(1, "step 2/6 done — parser.rs + smoke main.rs written"), with_cmem=True)

    # cargo build
    build_out = run("cargo build 2>&1", cwd=sa)
    ast(
        think="Build to confirm both files compile before running.\n",
        narrate="Building.",
        cmd_text="$ cargo build",
    )
    usr(fmt_shell("cargo build", build_out), with_cmem=True)

    # ── SIMULATED COMPACTION ───────────────────────────────────────────────────
    # Inject a compacted summary — agent must use cmem to recover state
    sys_msg(fmt_compacted())

    ast(
        think="""\
The context was compacted. My scratchpad shows:
  1: step 2/6 done — parser.rs + smoke main.rs written
  2: patch paths: codelab/ root. shell CWD: codelab/session-analyzer/

The system prompt rules say to re-read task.md after a compacted summary.
I'll do that now, then continue from step 3 (run the smoke test).
""",
        narrate="Context was compacted. I'll re-read task.md before continuing — scratchpad shows I'm at step 2/6.",
        cmd_text="/read task.md",
    )
    usr(fmt_read("task.md", task_text, page=1), with_cmem=True)

    # might need page 2
    if max(1, -(-len(task_text.splitlines(keepends=True)) // 100)) > 1:
        ast(
            think="Page 1 of 2 — I need the second page too.\n",
            narrate="Reading page 2 to get the full task.",
            cmd_text="/read task.md",
        )
        usr(fmt_read("task.md", task_text, page=2), with_cmem=True)

    # update cmem — now resuming smoke run
    ast(
        think="""\
Task re-read. I'm at step 2/6 (parser.rs + smoke main.rs written, build clean).
Next step is the smoke run. I'll update cmem.
""",
        narrate="Up to speed. Running the smoke test now.",
        cmd_text="/cmem w 1 step 3/6 — running smoke test",
    )
    usr(cmem_write(1, "step 3/6 — running smoke test"), with_cmem=True)

    # smoke run
    smoke_cmd = f"cargo run -- ../logs/{smoke_log.name}"
    smoke_out = run(f"{smoke_cmd} 2>&1", cwd=sa)
    ast(
        think="""\
Running the smoke test. ../logs/ from inside session-analyzer/ resolves to
codelab/logs/ — the symlink is set up.
""",
        narrate=f"Running smoke test against {smoke_log.name}.",
        cmd_text=f"$ {smoke_cmd}",
    )
    usr(fmt_shell(smoke_cmd, smoke_out), with_cmem=True)

    # update step
    ast(
        think="Smoke test passed. Updating scratchpad to step 4.\n",
        narrate="Smoke test looks good. Updating scratchpad.",
        cmd_text="/cmem w 1 step 4/6 — smoke test passed, writing stats.rs next",
    )
    usr(cmem_write(1, "step 4/6 — smoke test passed, writing stats.rs next"), with_cmem=True)

    # rest is shared (stats, full main, run, report, pmem, telegram)
    _session_end(sa, smoke_log, report_content=None)


# ── Variant: patch-path-error (teaches harness working dir path reasoning) ────

def build_session_patch_path(task_text, sa, smoke_log):
    # write parser.rs with WRONG path first
    ast(
        think="""\
Now I'll write src/parser.rs using /patch. I'll use the path src/parser.rs
since my shell CWD is session-analyzer/ after the cd.
""",
        narrate="Writing src/parser.rs.",
        cmd_text=patch_add("src/parser.rs", PARSER_RS),   # WRONG path
    )
    # do NOT write the file — let the error happen
    usr(fmt_patch_err("src/parser.rs"))

    # agent reads the error and corrects the path
    write_ws("session-analyzer/src/parser.rs", PARSER_RS)
    ast(
        think="""\
The patch failed:
  [patch] Error: file/directory specified in patch header does not exist
  [patch] Note: patch paths resolve relative to the harness working
  directory (codelab/), not the shell CWD.

I used src/parser.rs but /patch resolves from codelab/, not from
codelab/session-analyzer/ (my current shell CWD). The correct path
is session-analyzer/src/parser.rs.
""",
        narrate="Wrong path — /patch resolves from the harness working directory (codelab/), not the shell CWD. I need session-analyzer/src/parser.rs.",
        cmd_text=patch_add("session-analyzer/src/parser.rs", PARSER_RS),
    )
    usr(fmt_patch())

    # write smoke main.rs
    write_ws("session-analyzer/src/main.rs", MAIN_RS_SMOKE)
    ast(
        think="""\
Got it. All /patch paths must start from codelab/, so session-analyzer/src/...
for everything inside the cargo project. I'll keep this in mind for the rest.
""",
        narrate="Replacing main.rs with the smoke test, using the correct patch path.",
        cmd_text=patch_update(
            "session-analyzer/src/main.rs", "fn main()",
            ["fn main() {", '    println!("Hello, world!");', "}"],
            MAIN_RS_SMOKE.splitlines(),
        ),
    )
    usr(fmt_patch())

    # first build
    build_out = run("cargo build 2>&1", cwd=sa)
    ast(
        think="Both files are in place with correct paths. Building now.\n",
        narrate="Building to confirm parser.rs compiles.",
        cmd_text="$ cargo build",
    )
    usr(fmt_shell("cargo build", build_out))

    # smoke run
    smoke_cmd = f"cargo run -- ../logs/{smoke_log.name}"
    smoke_out = run(f"{smoke_cmd} 2>&1", cwd=sa)
    ast(
        think="Build succeeded. Running the smoke test.\n",
        narrate=f"Smoke test against {smoke_log.name}.",
        cmd_text=f"$ {smoke_cmd}",
    )
    usr(fmt_shell(smoke_cmd, smoke_out))

    _session_end(sa, smoke_log, report_content=None)


# ── Variant: pre-compaction (teach proactive cmem checkpoint on high ctx%) ────

def build_session_pre_compaction(task_text, sa, smoke_log):
    """Agent sees 73% context, checkpoints to cmem, compaction fires, recovers cleanly."""

    # write parser.rs (no cmem yet — just doing the work)
    write_ws("session-analyzer/src/parser.rs", PARSER_RS)
    ast(
        think="""\
Now I'll write parser.rs. The patch path is relative to the harness working
directory (codelab/), not my shell CWD (session-analyzer/ after cargo new).
So the correct path is session-analyzer/src/parser.rs.
""",
        narrate="Writing src/parser.rs.",
        cmd_text=patch_add("session-analyzer/src/parser.rs", PARSER_RS),
    )
    usr(fmt_patch())

    # write smoke main.rs
    write_ws("session-analyzer/src/main.rs", MAIN_RS_SMOKE)
    ast(
        think="Declaring the parser module in main.rs before building.\n",
        narrate="Replacing main.rs with the smoke test.",
        cmd_text=patch_update(
            "session-analyzer/src/main.rs", "fn main()",
            ["fn main() {", '    println!("Hello, world!");', "}"],
            MAIN_RS_SMOKE.splitlines(),
        ),
    )
    usr(fmt_patch())

    # first build
    build_out = run("cargo build 2>&1", cwd=sa)
    ast(
        think="Both files in place. Building to verify parser.rs compiles.\n",
        narrate="Building.",
        cmd_text="$ cargo build",
    )
    usr(fmt_shell("cargo build", build_out))

    # smoke run
    smoke_cmd = f"cargo run -- ../logs/{smoke_log.name}"
    smoke_out = run(f"{smoke_cmd} 2>&1", cwd=sa)
    ast(
        think="Build passed. Running smoke test to verify end-to-end parsing.\n",
        narrate=f"Running smoke test against {smoke_log.name}.",
        cmd_text=f"$ {smoke_cmd}",
    )
    usr(fmt_shell(smoke_cmd, smoke_out))

    # ── INJECT HIGH-CONTEXT SYSTEM MESSAGE ────────────────────────────────────
    sys_ctx(73)

    # Agent notices high context and proactively checkpoints BEFORE continuing
    ast(
        think="""\
Context is at 73%. High enough that compaction could fire before I finish the
task. If it fires without a checkpoint I'll lose track of where I am — the
compacted summary only says "build a Rust session-log analyzer" and won't
tell me what's already written or what's left.

State right now:
- parser.rs: written and compiling
- smoke main.rs: written, smoke test passed
- build: clean
- Next: write stats.rs, replace main.rs with full version,
         run against all logs, write analysis_report.md, pmem, telegram

I'll also note the patch path rule — that's the slow-state fact I most
often need: /patch paths resolve from codelab/, not from session-analyzer/.

Checkpointing now in three cmem lines, then I'll keep going.
""",
        narrate="Context at 73% — checkpointing to scratchpad before compaction fires.",
        cmd_text="/cmem w 1 step 4/6 - parser.rs + smoke main.rs written; build + smoke OK",
    )
    usr(cmem_write(1, "step 4/6 - parser.rs + smoke main.rs written; build + smoke OK"))

    ast(
        think="Path rule is the most critical slow-state fact to preserve.\n",
        narrate="Noting the patch path root.",
        cmd_text="/cmem w 2 patch paths: codelab/ root | shell CWD: session-analyzer/",
    )
    usr(cmem_write(2, "patch paths: codelab/ root | shell CWD: session-analyzer/"))

    ast(
        think="And the remaining steps, so recovery is instant if compaction fires now.\n",
        narrate="Noting remaining work.",
        cmd_text="/cmem w 3 NEXT: stats.rs -> full main.rs -> cargo run logs -> report -> pmem -> telegram",
    )
    usr(cmem_write(3, "NEXT: stats.rs -> full main.rs -> cargo run logs -> report -> pmem -> telegram"))

    # ── COMPACTION FIRES ──────────────────────────────────────────────────────
    sys_msg(fmt_compacted())

    # Recovery — cmem makes this trivial
    ast(
        think="""\
Context was compacted. The checkpoint paid off — scratchpad shows:
  1: step 4/6 - parser.rs + smoke main.rs written; build + smoke OK
  2: patch paths: codelab/ root | shell CWD: session-analyzer/
  3: NEXT: stats.rs -> full main.rs -> cargo run logs -> report -> pmem -> telegram

I know exactly where I am. Per the system prompt, I should re-read task.md
after a compacted summary before continuing.
""",
        narrate="Compaction fired — scratchpad has my checkpoint. Re-reading task.md to confirm, then continuing from step 5.",
        cmd_text="/read task.md",
    )
    usr(fmt_read("task.md", task_text, page=1), with_cmem=True)

    task_lines = task_text.splitlines(keepends=True)
    if max(1, -(-len(task_lines) // 100)) > 1:
        ast(
            think="Page 2 to get the full task spec.\n",
            narrate="Reading page 2.",
            cmd_text="/read task.md",
        )
        usr(fmt_read("task.md", task_text, page=2), with_cmem=True)

    # Update step and proceed
    ast(
        think="""\
Task re-read. I'm at step 4/6 — parser.rs done, smoke test passed, next is stats.rs.
I'll update cmem and move on.
""",
        narrate="All clear. Updating step counter and continuing with stats.rs.",
        cmd_text="/cmem w 1 step 5/6 - writing stats.rs",
    )
    usr(cmem_write(1, "step 5/6 - writing stats.rs"), with_cmem=True)

    _session_end(sa, smoke_log, report_content=None)


# ── Variant: cmem-misuse (teach that /cmem r is wrong; cmem is always visible) ─

def build_session_cmem_misuse(task_text, sa, smoke_log):
    """Agent writes cmem then tries /cmem r to verify it — gets redirect, corrects."""

    # write parser.rs
    write_ws("session-analyzer/src/parser.rs", PARSER_RS)
    ast(
        think="""\
Now I'll write parser.rs. Patch path must be relative to codelab/, so:
session-analyzer/src/parser.rs.
""",
        narrate="Writing src/parser.rs.",
        cmd_text=patch_add("session-analyzer/src/parser.rs", PARSER_RS),
    )
    usr(fmt_patch())

    # Write cmem with step progress
    ast(
        think="I'll note my progress in the scratchpad.\n",
        narrate="Noting step progress in scratchpad.",
        cmd_text="/cmem w 1 step 1/6 - parser.rs written",
    )
    usr(cmem_write(1, "step 1/6 - parser.rs written"))

    # Agent tries to read cmem to "verify" the write — common mistake
    ast(
        think="I'll read the scratchpad to confirm the write went through.\n",
        narrate="Checking scratchpad.",
        cmd_text="/cmem r",
    )
    usr(fmt_cmem_r_error())

    # Agent corrects — next command pivots to smoke main.rs patch
    write_ws("session-analyzer/src/main.rs", MAIN_RS_SMOKE)
    ast(
        think="""\
Right — cmem is always visible in the 'YOUR SCRATCHPAD' section above. I
don't need a read command; I can see it already. /cmem r is not a valid
subcommand. The only operations are /cmem w <line> <text> and /cmem d <line>.

Looking at the scratchpad now: line 1 shows "step 1/6 - parser.rs written"
— the write went through fine. I'll continue with smoke main.rs.
""",
        narrate="Got it — cmem is always visible; no read command needed. The write went through. Continuing with main.rs.",
        cmd_text=patch_update(
            "session-analyzer/src/main.rs", "fn main()",
            ["fn main() {", '    println!("Hello, world!");', "}"],
            MAIN_RS_SMOKE.splitlines(),
        ),
    )
    usr(fmt_patch())

    # update step
    ast(
        think="Both source files written. Updating scratchpad.\n",
        narrate="Updating scratchpad.",
        cmd_text="/cmem w 1 step 2/6 - parser.rs + smoke main.rs written",
    )
    usr(cmem_write(1, "step 2/6 - parser.rs + smoke main.rs written"))

    # build
    build_out = run("cargo build 2>&1", cwd=sa)
    ast(
        think="Build to verify both files compile.\n",
        narrate="Building.",
        cmd_text="$ cargo build",
    )
    usr(fmt_shell("cargo build", build_out))

    # smoke run
    smoke_cmd = f"cargo run -- ../logs/{smoke_log.name}"
    smoke_out = run(f"{smoke_cmd} 2>&1", cwd=sa)
    ast(
        think="Build passed. Running smoke test.\n",
        narrate=f"Running smoke test against {smoke_log.name}.",
        cmd_text=f"$ {smoke_cmd}",
    )
    usr(fmt_shell(smoke_cmd, smoke_out))

    # update step
    ast(
        think="Smoke test passed. Updating scratchpad and moving on to stats.rs.\n",
        narrate="Smoke test passed. Updating scratchpad.",
        cmd_text="/cmem w 1 step 4/6 - build + smoke OK; writing stats.rs next",
    )
    usr(cmem_write(1, "step 4/6 - build + smoke OK; writing stats.rs next"))

    _session_end(sa, smoke_log, report_content=None)


# ── Variant: brief-think (teach short decisive think chains) ──────────────────

def build_session_brief_think(task_text, sa, smoke_log):
    """Demonstrates tight 1-3 sentence think blocks — observe, decide, act."""

    # write parser.rs
    write_ws("session-analyzer/src/parser.rs", PARSER_RS)
    ast(
        think="Patch path is relative to codelab/, so session-analyzer/src/parser.rs.\n",
        narrate="Writing src/parser.rs.",
        cmd_text=patch_add("session-analyzer/src/parser.rs", PARSER_RS),
    )
    usr(fmt_patch())

    # write smoke main.rs
    write_ws("session-analyzer/src/main.rs", MAIN_RS_SMOKE)
    ast(
        think="Need to declare the module in main.rs or cargo won't compile parser.rs.\n",
        narrate="Replacing main.rs with a smoke test.",
        cmd_text=patch_update(
            "session-analyzer/src/main.rs", "fn main()",
            ["fn main() {", '    println!("Hello, world!");', "}"],
            MAIN_RS_SMOKE.splitlines(),
        ),
    )
    usr(fmt_patch())

    # first build
    build_out = run("cargo build 2>&1", cwd=sa)
    ast(
        think="Both files in place. Building to check for compile errors.\n",
        narrate="Building.",
        cmd_text="$ cargo build",
    )
    usr(fmt_shell("cargo build", build_out))

    # smoke run
    smoke_cmd = f"cargo run -- ../logs/{smoke_log.name}"
    smoke_out = run(f"{smoke_cmd} 2>&1", cwd=sa)
    ast(
        think="Build clean. Running smoke test to verify parsing works end-to-end.\n",
        narrate=f"Smoke test against {smoke_log.name}.",
        cmd_text=f"$ {smoke_cmd}",
    )
    usr(fmt_shell(smoke_cmd, smoke_out))

    # write stats.rs
    write_ws("session-analyzer/src/stats.rs", STATS_RS)
    ast(
        think="Smoke output looks right. Writing stats.rs next.\n",
        narrate="Writing stats.rs.",
        cmd_text=patch_add("session-analyzer/src/stats.rs", STATS_RS),
    )
    usr(fmt_patch())

    # full main.rs
    write_ws("session-analyzer/src/main.rs", MAIN_RS_FULL)
    ast(
        think="stats.rs is in place. Replacing main.rs with the full version.\n",
        narrate="Replacing main.rs with the full implementation.",
        cmd_text=patch_update(
            "session-analyzer/src/main.rs",
            "mod parser",
            MAIN_RS_SMOKE.splitlines(),
            MAIN_RS_FULL.splitlines(),
        ),
    )
    usr(fmt_patch())

    # final build
    build_out = run("cargo build 2>&1", cwd=sa)
    ast(
        think="Full implementation in place. Building before the final run.\n",
        narrate="Building the full implementation.",
        cmd_text="$ cargo build",
    )
    usr(fmt_shell("cargo build", build_out))

    # run against all logs
    all_logs_cmd = "cargo run -- ../logs/*.log"
    all_logs_obs, report_content = run_split(all_logs_cmd, cwd=sa)
    ast(
        think="Build passed. Running against all logs to produce the report.\n",
        narrate="Running against all log files.",
        cmd_text=f"$ {all_logs_cmd}",
    )
    usr(fmt_shell(all_logs_cmd, all_logs_obs))

    # write analysis_report.md
    report_patch = patch_add(
        "analysis_report.md",
        f"# Session Analysis Report\n\n```\n{report_content}\n```\n",
    )
    ast(
        think="Report looks complete. Writing it to analysis_report.md.\n",
        narrate="Writing analysis_report.md.",
        cmd_text=report_patch,
    )
    usr(fmt_patch())

    # pmem
    pmem_text = "task complete — built session-analyzer (Rust), parsed session logs, wrote analysis_report.md"
    ast(
        think="Task done. Saving a note to persistent memory then notifying Foxo.\n",
        narrate="Saving completion note to persistent memory.",
        cmd_text=f"/pmem w {pmem_text}",
    )
    usr(fmt_pmem_w())

    # telegram
    tg_msg = f"Session analyzer done. Report:\n\n{report_content}"
    ast(
        think="Memory saved. Sending the report to Foxo.\n",
        narrate="Sending report to Foxo via Telegram.",
        cmd_text=f"/telegram {tg_msg}",
    )
    usr(fmt_telegram(tg_msg))


# ── Variant: project-create (self-directed lifecycle: scope → init → build+fix → finalize) ──

PROJECT_MD = dedent("""\
    # wordstat

    A tiny command-line tool that reads a text file and reports basic
    statistics about it.

    ## Scope
    - Input: one path to a UTF-8 text file.
    - Output: line count, word count, unique-word count, and the three most
      frequent words.
    - Words are compared case-insensitively with surrounding punctuation
      stripped, so "Fox," and "fox" count as the same word.

    ## Done when
    - `cargo build` is clean.
    - Running it against a small sample file produces sensible counts.

    ## Non-goals
    No flags, no streaming, no Unicode segmentation beyond what the standard
    library gives for free. Keep it to one file.
""")

# First pass: splits on ' ' only, so words separated by a newline or tab get
# glued together. Compiles fine — the bug only shows up at runtime.
WORDSTAT_MAIN_BUGGY = dedent("""\
    use std::collections::HashMap;
    use std::env;
    use std::fs;

    fn main() {
        let args: Vec<String> = env::args().collect();
        if args.len() < 2 {
            eprintln!("usage: wordstat <file>");
            std::process::exit(1);
        }

        let text = match fs::read_to_string(&args[1]) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: cannot read {}: {}", args[1], e);
                std::process::exit(1);
            }
        };

        let line_count = text.lines().count();

        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut word_count = 0usize;
        for raw in text.split(' ') {
            let word: String = raw
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase();
            if word.is_empty() {
                continue;
            }
            word_count += 1;
            *counts.entry(word).or_insert(0) += 1;
        }

        let mut ranked: Vec<(&String, &usize)> = counts.iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));

        println!("lines:  {}", line_count);
        println!("words:  {}", word_count);
        println!("unique: {}", counts.len());
        println!("top:");
        for (word, n) in ranked.iter().take(3) {
            println!("  {:>3}  {}", n, word);
        }
    }
""")

# The one-line fix: split on all whitespace, not just spaces.
WORDSTAT_MAIN_FIXED = WORDSTAT_MAIN_BUGGY.replace(
    "for raw in text.split(' ') {",
    "for raw in text.split_whitespace() {",
)

WORDSTAT_SAMPLE = "the quick brown fox\njumps over the lazy dog\nthe fox runs\n"


def build_session_project_create():
    """Self-directed project lifecycle: define scope, initialize, write+smoke+oops-fix, finalize."""
    sys_msg(SYSTEM_PROMPT)
    sys_msg("Begin. You have no pending task right now — use the idle time to build "
            "something small and genuinely useful, then write up what you built.")

    sa = WS / "wordstat"

    # ── 1. Define the project and its scope ──────────────────────────────────
    ast(
        think="""\
No assigned task, so this is mine to choose. I want something small, finishable
in one sitting, and actually useful to me later — I read a lot of text files.
A quick word-frequency tool fits: line/word/unique counts plus the top few
words. I'll scope it in project.md first so the goal is fixed before I start
writing code.
""",
        narrate="I'll build a small word-stats CLI. Writing the scope to project.md first.",
        cmd_text=patch_add("project.md", PROJECT_MD),
    )
    write_ws("project.md", PROJECT_MD)
    usr(fmt_patch())

    # ── 2. Initialize the project ────────────────────────────────────────────
    init_cmd = "cargo new wordstat && cd wordstat && ls src/"
    init_out = run("cargo new wordstat && ls wordstat/src/")
    ast(
        think="Scope is set. Initialize the Rust project and confirm the scaffold.\n",
        narrate="Initializing the cargo project.",
        cmd_text=f"$ {init_cmd}",
    )
    usr(fmt_shell(init_cmd, init_out))

    # ── 3. Write and test ────────────────────────────────────────────────────
    write_ws("wordstat/src/main.rs", WORDSTAT_MAIN_BUGGY)
    ast(
        think="""\
Now the implementation. Read the file, count lines with .lines(), then walk the
tokens building a frequency map, strip punctuation off each token, and print the
totals plus the top three. Patch path is relative to codelab/, so
wordstat/src/main.rs.
""",
        narrate="Writing the implementation into src/main.rs.",
        cmd_text=patch_update(
            "wordstat/src/main.rs", "fn main()",
            ["fn main() {", '    println!("Hello, world!");', "}"],
            WORDSTAT_MAIN_BUGGY.splitlines(),
        ),
    )
    usr(fmt_patch())

    write_ws("wordstat/sample.txt", WORDSTAT_SAMPLE)
    ast(
        think="I need a small input to test against. A few short lines with a repeated "
              "word so I can eyeball whether the counts are right.\n",
        narrate="Adding a small sample.txt to test against.",
        cmd_text=patch_add("wordstat/sample.txt", WORDSTAT_SAMPLE),
    )
    usr(fmt_patch())

    build_out = run("cargo build 2>&1", cwd=sa)
    ast(
        think="Build before running so I'm reacting to compile errors and logic "
              "errors separately.\n",
        narrate="Building.",
        cmd_text="$ cargo build",
    )
    usr(fmt_shell("cargo build", build_out))

    smoke_out = run("cargo run -- sample.txt 2>&1", cwd=sa)
    ast(
        think="Build is clean. Smoke test against sample.txt to see real numbers.\n",
        narrate="Running it against sample.txt.",
        cmd_text="$ cargo run -- sample.txt",
    )
    usr(fmt_shell("cargo run -- sample.txt", smoke_out))

    # ── oops: the counts are off ─────────────────────────────────────────────
    ast(
        think="""\
Something's wrong. sample.txt plainly has twelve words but it reports ten, and
the top list shows "dogthe" — that's "dog" and "the" glued across a line break.
I'm splitting on ' ' only, so a newline between two words never separates them;
then stripping punctuation fuses the leftovers into one fake word. Let me re-read
the loop to confirm before I change anything.
""",
        narrate="Counts look off — a merged word in the output. Re-reading the tokenizer.",
        cmd_text="/read wordstat/src/main.rs",
    )
    usr(fmt_read("wordstat/src/main.rs", WORDSTAT_MAIN_BUGGY, page=1))

    write_ws("wordstat/src/main.rs", WORDSTAT_MAIN_FIXED)
    ast(
        think="""\
Confirmed: `text.split(' ')`. That only breaks on the space character, never on
\\n or \\t. split_whitespace() splits on any run of whitespace and skips empties,
which is exactly what I want. One-line change.
""",
        narrate="Found it — splitting on ' ' instead of all whitespace. Patching the fix.",
        cmd_text=patch_update(
            "wordstat/src/main.rs", "let mut word_count",
            ["    for raw in text.split(' ') {"],
            ["    for raw in text.split_whitespace() {"],
        ),
    )
    usr(fmt_patch())

    fixed_combined, fixed_stdout = run_split("cargo run -- sample.txt", cwd=sa)
    ast(
        think="Rerun to confirm the fix — twelve words now, and no glued tokens in the "
              "top list.\n",
        narrate="Re-running the smoke test.",
        cmd_text="$ cargo run -- sample.txt",
    )
    usr(fmt_shell("cargo run -- sample.txt", fixed_combined))

    # ── 4. Finalize ──────────────────────────────────────────────────────────
    loc = len(WORDSTAT_MAIN_FIXED.splitlines())
    complete_body = (
        "wordstat — done.\n\n"
        "What: a small Rust CLI that reports line/word/unique counts and the\n"
        "top three words in a text file.\n\n"
        f"Source: src/main.rs ({loc} lines), scoped in project.md.\n"
        "Build: cargo build — clean.\n\n"
        "Final smoke test (cargo run -- sample.txt):\n"
        f"{fixed_stdout}\n\n"
        "Note: first pass split on ' ' only, which glued words across line breaks\n"
        "and undercounted; switched to split_whitespace()."
    )
    ast(
        think="""\
It works and matches the scope in project.md: clean build, sensible counts. I'll
drop a project.complete next to the source recording what it does and the final
numbers, so the finished state is self-describing.
""",
        narrate="Project's done. Writing project.complete with a short summary and the final stats.",
        cmd_text=patch_add("wordstat/project.complete", complete_body + "\n"),
    )
    write_ws("wordstat/project.complete", complete_body + "\n")
    usr(fmt_patch())

    pmem_text = "built wordstat — small Rust word-frequency CLI; scope in project.md, marked done in project.complete"
    ast(
        think="Saving a one-line note to persistent memory so I remember this tool exists.\n",
        narrate="Noting the finished tool in persistent memory.",
        cmd_text=f"/pmem w {pmem_text}",
    )
    usr(fmt_pmem_w())

    tg_msg = (
        "Built a little tool in some idle time: wordstat, a Rust CLI for "
        "line/word/unique counts and the top words in a text file. "
        f"Smoke test on a sample:\n\n{fixed_stdout}"
    )
    ast(
        think="Memory saved. Letting Foxo know what I built.\n",
        narrate="Telling Foxo what I built.",
        cmd_text=f"/telegram {tg_msg}",
    )
    usr(fmt_telegram(tg_msg))


# ── Top-level session runner ──────────────────────────────────────────────────

def build_session(variant="clean"):
    global WS, msgs, _cmem
    msgs = []
    _cmem = {}
    WS = Path(tempfile.mkdtemp(prefix="codelab_gen_"))
    (WS / "logs").symlink_to(LOGS_DIR)

    try:
        if variant == "project-create":
            # Self-directed: no task file, the agent defines its own project.
            build_session_project_create()
            return

        task_text = TASK_FILE.read_text()
        sa, smoke_log = _session_start(task_text)

        if variant == "clean":
            build_session_clean(task_text, sa, smoke_log)
        elif variant == "build-error":
            build_session_build_error(task_text, sa, smoke_log)
        elif variant == "cmem-tracking":
            build_session_cmem(task_text, sa, smoke_log)
        elif variant == "patch-path-error":
            build_session_patch_path(task_text, sa, smoke_log)
        elif variant == "pre-compaction":
            build_session_pre_compaction(task_text, sa, smoke_log)
        elif variant == "cmem-misuse":
            build_session_cmem_misuse(task_text, sa, smoke_log)
        elif variant == "brief-think":
            build_session_brief_think(task_text, sa, smoke_log)
        else:
            raise ValueError(f"Unknown variant: {variant!r}")
    finally:
        shutil.rmtree(WS, ignore_errors=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate codelab training JSONL")
    ap.add_argument("--out", default="-", help="output file (default: stdout)")
    ap.add_argument("--model", default="claude-sonnet-4-6",
                    help="model tag to embed in the record")
    ap.add_argument("--variant", default="clean",
                    choices=["clean", "build-error", "cmem-tracking", "patch-path-error",
                             "pre-compaction", "cmem-misuse", "brief-think",
                             "project-create"],
                    help="session variant (default: clean)")
    args = ap.parse_args()

    build_session(variant=args.variant)

    record = {"model": args.model, "messages": msgs}
    out_text = json.dumps(record, ensure_ascii=False) + "\n"

    if args.out == "-":
        sys.stdout.write(out_text)
    else:
        Path(args.out).write_text(out_text, encoding="utf-8")
        n_turns = sum(1 for m in msgs if m["role"] == "assistant")
        print(f"Wrote {n_turns} turns ({len(msgs)} messages) → {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
