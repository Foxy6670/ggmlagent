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
    """Append a user/observation message, optionally prefixed with current cmem state."""
    content = (_cmem_header() + c) if with_cmem else c
    msgs.append({"role": "user", "content": content})

def ast(think, narrate, cmd_text):
    think_part = f"<think>\n{think.strip()}\n</think>\n" if think else ""
    msgs.append({"role": "assistant", "content":
        f"{think_part}{narrate.strip()}\n```\n{cmd_text.strip()}\n```\n<|eoc|>"})

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
    usr("Begin. Read your task file first.")

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
    usr(fmt_compacted())

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


# ── Top-level session runner ──────────────────────────────────────────────────

def build_session(variant="clean"):
    global WS, msgs, _cmem
    msgs = []
    _cmem = {}
    WS = Path(tempfile.mkdtemp(prefix="codelab_gen_"))
    (WS / "logs").symlink_to(LOGS_DIR)

    try:
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
                    choices=["clean", "build-error", "cmem-tracking", "patch-path-error"],
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
