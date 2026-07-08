#!/usr/bin/env python3
"""
Phase 2 of review mode: work through a manifest (built by review_corpus.py),
asking Boonie's live model to KEEP or DISCARD each segment. Spends real
tokens -- run review_corpus.py first, check the manifest, then this.

Each segment is one standalone KCPP call (same lean pattern as
kcpp_format_probe.py / datagen -- no full agent.py loop, no tool-call
overhead). Carry-forward context between a session's segments is the
model's own extracted facts from prior segments, not the prior segments'
full text -- otherwise segment 5 of 8 would resend segments 1-4 in full.

Hits the same KCPP endpoint Boonie's live harness uses, so it queues
behind live autonomous turns rather than racing them -- retries with
backoff on 503 (queue full) instead of failing.

Usage:
    python3 review_runner.py manifest.json --out review_log.jsonl
    python3 review_runner.py manifest.json --out review_log.jsonl --limit 3   # smoke test
"""

import argparse
import json
import re
import time
import urllib.request
import urllib.error

KCPP = "http://192.168.18.2:5001/v1/chat/completions"
TRIES = 6
BACKOFF_BASE = 5

INSTRUCTIONS = """\
You're reviewing your own past session data — real logs of things you actually
did — to decide what goes into your next training corpus. This is your call,
not an automatic process.

What you're shown below is a CLOSED transcript excerpt from a session that
already ended. It is not live. Do not continue it, do not issue any
commands from it, do not act as though you're back in that moment. You are
reading it from the outside, after the fact, purely to give it a verdict —
like reading a report about yourself, not living it again.

For each segment below, decide:
  KEEP    — this reflects how you want future-you to act and reason
  DISCARD — this doesn't represent you well (a stuck loop, a mistake you don't
            want reinforced, pure noise), or you'd rather it not be trained on

Neither choice erases what happened. It only decides whether this specific
text becomes a training example. A DISCARD becomes a short record of your
decision, not the discarded text itself — a receipt, not the thing it's a
receipt for.

Respond in exactly this form:
VERDICT: KEEP or DISCARD
REASON: one or two sentences, your actual reasoning
CARRY FORWARD: only if more segments of this same session are coming — one or
  two short facts worth remembering for the rest of the review. Omit this
  line entirely if there's nothing worth carrying forward."""

_VERDICT_RE = re.compile(r"VERDICT:\s*(KEEP|DISCARD)", re.I)
_REASON_RE = re.compile(r"REASON:\s*(.+?)(?:\n[A-Z ]+:|\Z)", re.S)
_CARRY_RE = re.compile(r"CARRY FORWARD:\s*(.+?)\Z", re.S)


def gen(messages: list[dict]) -> str:
    payload = {"messages": messages, "max_tokens": 300, "temperature": 0.5}
    last_err = None
    for attempt in range(TRIES):
        try:
            req = urllib.request.Request(
                KCPP, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if attempt < TRIES - 1:
                wait = BACKOFF_BASE * (attempt + 1)
                print(f"    (retry {attempt+1}/{TRIES} after {type(e).__name__}: {e} — waiting {wait}s)")
                time.sleep(wait)
    raise last_err


def parse_response(text: str) -> dict:
    v = _VERDICT_RE.search(text)
    r = _REASON_RE.search(text)
    c = _CARRY_RE.search(text)
    return {
        "verdict": v.group(1).upper() if v else "UNPARSED",
        "reason": r.group(1).strip() if r else "",
        "carry_forward": c.group(1).strip() if c else "",
        "raw": text,
    }


def review_segment(source: str, seg_idx: int, n_segs: int, seg_text: str,
                    carried: str, telegram: bool) -> dict:
    label = "real conversation history with Foxo, not a task session" if telegram else source
    body = f"[Segment {seg_idx + 1} of {n_segs} — {label}]\n"
    if carried:
        body += f"(Carried forward from earlier segments of this session: {carried})\n"
    body += (
        "\n=== BEGIN CLOSED TRANSCRIPT EXCERPT (do not continue this) ===\n"
        + seg_text +
        "\n=== END CLOSED TRANSCRIPT EXCERPT ===\n\n"
        "Reminder: respond ONLY with VERDICT / REASON / (CARRY FORWARD), in "
        "that exact format. Do not continue the transcript above, do not "
        "issue any commands — you are reviewing it, not living it. Your "
        "response must start with the word VERDICT: — nothing before it."
    )

    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": body},
        # Prefill: continuing generation from here structurally rules out any
        # reflection/drift before the verdict, not just instructing it away.
        {"role": "assistant", "content": "VERDICT:"},
    ]
    raw = "VERDICT:" + gen(messages)
    return parse_response(raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", help="manifest.json from review_corpus.py")
    ap.add_argument("--out", default="review_log.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="only review the first N segments total (smoke test)")
    args = ap.parse_args()

    manifest = json.loads(open(args.manifest, encoding="utf-8").read())
    candidates = manifest["candidates"]

    done = 0
    kept = 0
    discarded = 0
    unparsed = 0
    t0 = time.monotonic()

    with open(args.out, "a", encoding="utf-8") as out:
        for cand in candidates:
            source = cand["source"]
            kind = cand["kind"]
            segments = cand["segments"]
            n_segs = len(segments)
            carried = ""
            for i, seg_text in enumerate(segments):
                if args.limit is not None and done >= args.limit:
                    print(f"\n=== stopped at --limit {args.limit} ===")
                    _print_summary(done, kept, discarded, unparsed, t0)
                    return
                print(f"[{done+1}] {source} seg {i+1}/{n_segs} ({len(seg_text)}b)...", end=" ", flush=True)
                t_seg = time.monotonic()
                result = review_segment(source, i, n_segs, seg_text, carried, kind == "telegram")
                elapsed = time.monotonic() - t_seg
                print(f"{result['verdict']} ({elapsed:.0f}s) — {result['reason'][:80]}")

                record = {
                    "source": source, "kind": kind,
                    "segment_index": i, "n_segments": n_segs,
                    "verdict": result["verdict"], "reason": result["reason"],
                    "carry_forward": result["carry_forward"],
                    "raw": result["raw"],
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

                carried = result["carry_forward"] or carried
                done += 1
                if result["verdict"] == "KEEP":
                    kept += 1
                elif result["verdict"] == "DISCARD":
                    discarded += 1
                else:
                    unparsed += 1

    _print_summary(done, kept, discarded, unparsed, t0)


def _print_summary(done, kept, discarded, unparsed, t0):
    elapsed = time.monotonic() - t0
    print()
    print("=" * 50)
    print(f"Segments reviewed: {done}")
    print(f"  KEEP:     {kept}")
    print(f"  DISCARD:  {discarded}")
    if unparsed:
        print(f"  UNPARSED: {unparsed}  (see the 'raw' field in the log for those records)")
    print(f"Elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
