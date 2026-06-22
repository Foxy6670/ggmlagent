# Task: Build a session log analyzer (Rust)

Write a Rust tool that reads harness session logs and produces a human-readable
report. The logs live in `../logs/` — each is a `session_YYYY-MM-DD_HH-MM-SS.log`
file with timestamped lines tagged `[THINK]`, `[AGENT]`, `[CMD]`, `[OBS]`,
`[SYS]`, and `[RAW]`.

## Step 0 — Initialize the project (do this first, before writing any code)

Chain all three commands on one line:
```
$ cargo new session-analyzer && cd session-analyzer && ls src/
```

If `cargo new` fails because the directory already exists, it is already set up —
just run `$ cd session-analyzer && ls src/` to confirm and move on.
Do not write any `.rs` files until `ls src/` shows `main.rs`.

## How to write and edit files

Use `/patch` for all source code. It handles both creating new files and editing
existing ones. Never use `/appendlines` for `.rs` or `Cargo.toml` files — it
can only append and will corrupt code.

**Creating a new file:**
```
/patch
*** Begin Patch
*** Add File: src/parser.rs
+use std::fs;
+
+pub struct Turn {
+    pub think: String,
+    pub cmd:   String,
+}
*** End Patch
```

**Editing an existing file:**
```
/patch
*** Begin Patch
*** Update File: src/parser.rs
@@ parse_log
-    let mut turns = Vec::new();
+    let mut turns: Vec<Turn> = Vec::new();
*** End Patch
```

`@@ <name>` is an anchor — use a nearby function or struct name to locate the
hunk. Context lines (no prefix) are optional but help when the file is large.

**After every `/patch`:** run `$ cargo build` from inside `session-analyzer/`.
Read the full error output and fix the root cause before continuing.

## What to build

### `src/parser.rs`
A public module. Parse a session log file into a `Vec<Turn>`. Each `Turn`:
```rust
pub struct Turn {
    pub think: String,   // [THINK] block content
    pub agent: String,   // [AGENT] lines
    pub cmd:   String,   // [CMD] line, prefix and whitespace stripped
    pub obs:   String,   // [OBS] lines
}
```
A new turn starts at each line containing `[SYS]` and `Generation started`.

### `src/stats.rs`
A public module. Takes `&[Turn]` and returns:
```rust
pub struct SessionStats {
    pub total_turns:       usize,
    pub commands_issued:   usize,   // non-empty cmd
    pub think_lines:       usize,
    pub third_person_hits: usize,   // "the user" in think, case-insensitive
    pub top_command:       String,  // most-used command prefix, e.g. "/append"
    pub duration_secs:     u64,     // first to last timestamp (parse HH:MM:SS)
}
```

### `src/main.rs`
Accept log file paths as CLI arguments. For each file, parse → compute stats →
print:
```
session_2026-06-22_03-32-13.log  [8h 09m]
  Turns      : 51
  Commands   : 51
  Think lines: 621
  3rd-person : 61 hits
  Top command: /append (12)
```

## Build order and verification

1. Write `src/parser.rs` with `/patch`. Run `$ cargo build`. Fix all errors.
2. Write a quick smoke test inline in `src/main.rs` that calls `parse_log` on
   one real log from `../logs/` and prints the turn count. Run it. Verify the
   number looks right.
3. Write `src/stats.rs` with `/patch`. Run `$ cargo build`. Fix all errors.
4. Complete `src/main.rs` to call both modules. Run against all logs.

## Rules

- Run `$ cargo build` (or `$ cargo run`) after every file change. Do not skip.
- If `cargo build` fails, `/read` the file you just patched, find the actual
  error, and fix it. Do not re-patch the same thing without reading first.
- Work from `session-analyzer/` for all cargo commands. Use `../logs/` to reach
  the log files.
- Never use `/edit` or `/appendlines` for `.rs` files. Only `/patch`.

## Finish line

When `$ cargo run -- ../logs/*.log` prints a clean report for every log file,
copy the output into `../analysis_report.md` with `/patch *** Add File:`.

Then:
1. `/pmem w` — note task complete and what was built
2. `/telegram` — send Foxo the report output
3. Stop. Do not invent follow-on work.
