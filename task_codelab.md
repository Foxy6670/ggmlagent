# Task: Build a session log analyzer (Rust)

Write a Rust tool that reads harness session logs and produces a human-readable
report. The logs live in `logs/` — each is a `session_YYYY-MM-DD_HH-MM-SS.log`
file with timestamped lines tagged `[THINK]`, `[AGENT]`, `[CMD]`, `[OBS]`,
`[SYS]`, and `[RAW]`.

## Setup

Initialize the project first:

```
$ cargo new session-analyzer
$ cd session-analyzer
```

All source goes in `src/`. Use `$ cargo build` to check compilation and
`$ cargo run -- <args>` to test. Fix every compiler error before moving on —
the compiler output tells you exactly what is wrong.

## What to build

### `src/parser.rs`
A module that parses a session log file into a `Vec<Turn>`. Each `Turn` holds:
- `think: String` — full content of the `[THINK]` block
- `agent: String` — `[AGENT]` lines (narration + command syntax)
- `cmd: String` — command from the `[CMD]` line (strip the `[block]` prefix)
- `obs: String` — `[OBS]` lines that follow

A new turn starts at each `[SYS] Generation started` marker.

### `src/stats.rs`
A module that takes a `&[Turn]` and computes a `SessionStats` struct:
- `total_turns: usize`
- `commands_issued: usize` (non-empty `cmd`)
- `think_lines: usize`
- `third_person_hits: usize` (count of "the user" in think text, case-insensitive)
- `top_command: String` (most-used command prefix, e.g. `/append`)
- `duration_secs: u64` (first to last timestamp in the file)

### `src/main.rs`
CLI entry point. Accept one or more log file paths as arguments:

```
$ cargo run -- ../logs/session_2026-06-21_04-44-52.log
```

Print one block per session:
```
session_2026-06-21_04-44-52.log  [3h 15m]
  Turns      : 51
  Commands   : 51
  Think lines: 621
  3rd-person : 61 hits
  Top command: /append (12)
```

## Rules

- Build in order: `parser.rs` → `stats.rs` → `main.rs`.
- After each file, run `$ cargo build` and fix all errors before continuing.
- Test with a real log from `../logs/` at each stage — don't assume it works.
- Write code to files with `/edit` or `/appendlines`. Do not draft in triple-quote
  blocks without writing — write directly to the file.
- If `cargo build` fails, read the full error output and fix the root cause.
  Do not guess-and-retry without understanding the error.

## Finish line

When `$ cargo run -- ../logs/*.log` runs cleanly and prints a report for every
log, copy the output into `../analysis_report.md`.

Once `analysis_report.md` is written:
1. `/pmem w` — note task complete, what was built, any issues encountered
2. `/telegram` — send Foxo the summary output
3. Stop. Do not invent follow-on tasks.
