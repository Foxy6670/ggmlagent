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

def sys_msg(c): msgs.append({"role": "system", "content": c})
def usr(c):     msgs.append({"role": "user",   "content": c})

def ast(think, narrate, cmd_text):
    think_part = f"<think>\n{think.strip()}\n</think>\n" if think else ""
    msgs.append({"role": "assistant", "content":
        f"{think_part}{narrate.strip()}\n```\n{cmd_text.strip()}\n```\n<|eoc|>"})

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

def fmt_pmem_r(body="[memory.md: empty]"):
    return f"> /pmem r\n{body}"

def fmt_pmem_w():
    return "> /pmem w\n[memory saved]"

def fmt_telegram(text):
    return f"> /telegram {text}\n[telegram] Message sent to Foxo."

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

# ── Session builder ───────────────────────────────────────────────────────────

def build_session():
    global WS
    WS = Path(tempfile.mkdtemp(prefix="codelab_gen_"))
    log_link = WS / "logs"
    log_link.symlink_to(LOGS_DIR)

    try:
        task_text = TASK_FILE.read_text()

        # ── Start ──────────────────────────────────────────────────────────
        sys_msg(SYSTEM_PROMPT)
        usr("Begin. Read your task file first.")

        # Turn 1 — read task.md (page 1)
        ast(
            think="""\
I need to read task.md to understand what I'm supposed to build.
""",
            narrate="I'll read the task file first.",
            cmd_text="/read task.md",
        )
        usr(fmt_read("task.md", task_text, page=1))

        # Turn 2 — task.md page 2 (122 lines → 2 pages)
        task_lines = task_text.splitlines(keepends=True)
        total_pages = max(1, -(-len(task_lines) // 100))
        if total_pages > 1:
            ast(
                think="""\
That was page 1 of 2. I need to read the rest of the task before I start.
""",
                narrate="There's a second page — I'll read it.",
                cmd_text="/read task.md",
            )
            usr(fmt_read("task.md", task_text, page=2))

        # Turn 3 — check persistent memory
        ast(
            think="""\
I should check my persistent memory for any prior notes about this task
before beginning work.
""",
            narrate="I'll check persistent memory for any prior context.",
            cmd_text="/pmem r",
        )
        usr(fmt_pmem_r())

        # Turn 4 — Step 0: cargo new (all three commands chained)
        cargo_init_cmd = "cargo new session-analyzer && cd session-analyzer && ls src/"
        cargo_init_out = run(f"cargo new session-analyzer && ls session-analyzer/src/")
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

        # Write Cargo.toml (cargo new creates its own, but verify)
        # workspace now has session-analyzer/
        sa = WS / "session-analyzer"

        # Turn 5 — write parser.rs
        parser_patch = patch_add("session-analyzer/src/parser.rs", PARSER_RS)
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
            cmd_text=parser_patch,
        )
        usr(fmt_patch())

        # Turn 6 — write smoke-test main.rs (before first build)
        main_smoke_patch = patch_update(
            "session-analyzer/src/main.rs",
            "fn main()",
            ["fn main() {", '    println!("Hello, world!");', "}"],
            MAIN_RS_SMOKE.splitlines(),
        )
        write_ws("session-analyzer/src/main.rs", MAIN_RS_SMOKE)
        ast(
            think="""\
Before building I need to declare the parser module in main.rs,
otherwise cargo won't compile parser.rs at all.
I'll replace the default main with a smoke test that calls parse_log.
""",
            narrate="I'll replace main.rs with a smoke test that uses the parser.",
            cmd_text=main_smoke_patch,
        )
        usr(fmt_patch())

        # Turn 7 — cargo build (parser.rs + smoke main.rs)
        build2_out = run("cargo build 2>&1", cwd=sa)
        ast(
            think="""\
Both parser.rs and the smoke-test main.rs are in place. Let me build
to check for compile errors before running.
""",
            narrate="Building to check parser.rs compiles.",
            cmd_text="$ cargo build",
        )
        usr(fmt_shell("cargo build", build2_out))

        # Turn 8 — cargo run smoke test on one log
        log_files = sorted(LOGS_DIR.glob("*.log"))
        if not log_files:
            raise RuntimeError("No log files found in logs/")
        smoke_log = log_files[-1]  # most recent
        smoke_cmd = f"cargo run -- ../logs/{smoke_log.name}"
        smoke_out = run(f"{smoke_cmd} 2>&1", cwd=sa)
        ast(
            think="""\
I'll run the smoke test against the most recent log file.
The path ../logs/ is relative to session-analyzer/, which resolves to
codelab/logs/ in the harness — a symlink I set up earlier.
I want to see a turn count that looks plausible given the session length.
""",
            narrate=f"Running smoke test against {smoke_log.name}.",
            cmd_text=f"$ {smoke_cmd}",
        )
        usr(fmt_shell(smoke_cmd, smoke_out))

        # Turn 9 — write stats.rs
        stats_patch = patch_add("session-analyzer/src/stats.rs", STATS_RS)
        write_ws("session-analyzer/src/stats.rs", STATS_RS)
        ast(
            think="""\
The smoke test output looks right. Now I'll write stats.rs.
It uses crate::parser types, so I need the struct names to match.
I won't declare it in main.rs yet — I'll add both the module declaration
and the full main.rs logic in one patch after writing stats.rs.
""",
            narrate="Writing stats.rs.",
            cmd_text=stats_patch,
        )
        usr(fmt_patch())

        # Turn 10 — update main.rs to full version (declares both modules)
        full_main_patch = patch_update(
            "session-analyzer/src/main.rs",
            "mod parser",
            MAIN_RS_SMOKE.splitlines(),
            MAIN_RS_FULL.splitlines(),
        )
        write_ws("session-analyzer/src/main.rs", MAIN_RS_FULL)
        ast(
            think="""\
Now I'll replace smoke-test main.rs with the full version that declares
both mod parser and mod stats, adds fmt_duration, and loops over all
log files to print the formatted report.
""",
            narrate="Replacing main.rs with the full implementation.",
            cmd_text=full_main_patch,
        )
        usr(fmt_patch())

        # Turn 11 — cargo build full (both modules now declared)
        build4_out = run("cargo build 2>&1", cwd=sa)
        ast(
            think="""\
Let me build the complete implementation. Both modules are declared
and stats.rs exists, so this should compile cleanly.
""",
            narrate="Building the full implementation.",
            cmd_text="$ cargo build",
        )
        usr(fmt_shell("cargo build", build4_out))

        # Turn 12 — cargo run on all logs
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

        # Turn 13 — write analysis_report.md
        # The task says to copy output into ../analysis_report.md
        report_patch = patch_add(
            "analysis_report.md",
            f"# Session Analysis Report\n\n```\n{report_content}\n```\n",
        )
        ast(
            think="""\
The report looks complete. The task says to copy it into
../analysis_report.md using /patch. The path ../analysis_report.md
is relative to the harness working directory (codelab/), so the
correct patch path is analysis_report.md (one level up from session-analyzer/).
""",
            narrate="Writing the report to analysis_report.md.",
            cmd_text=report_patch,
        )
        usr(fmt_patch())

        # Turn 14 — /pmem w
        pmem_text = "task complete — built session-analyzer (Rust), parsed session logs, wrote analysis_report.md"
        ast(
            think="""\
Task is done. I'll save a brief memory note as the task requires,
then send the report to Foxo via Telegram.
""",
            narrate="Saving a completion note to persistent memory.",
            cmd_text=f"/pmem w {pmem_text}",
        )
        usr(fmt_pmem_w())

        # Turn 15 — /telegram
        tg_msg = f"Session analyzer done. Report:\n\n{report_content}"
        ast(
            think="""\
I'll send the report to Foxo now. Task is complete.
""",
            narrate="Sending the report to Foxo via Telegram.",
            cmd_text=f"/telegram {tg_msg}",
        )
        usr(fmt_telegram(tg_msg))

    finally:
        shutil.rmtree(WS, ignore_errors=True)

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate codelab training JSONL")
    parser.add_argument("--out", default="-", help="output file (default: stdout)")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="model tag to embed in the record")
    args = parser.parse_args()

    build_session()

    record = {"model": args.model, "messages": msgs}
    out_text = json.dumps(record, ensure_ascii=False) + "\n"

    if args.out == "-":
        sys.stdout.write(out_text)
    else:
        Path(args.out).write_text(out_text, encoding="utf-8")
        print(f"Wrote {len(msgs)} messages → {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
