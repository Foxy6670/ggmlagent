#!/usr/bin/env python3
"""Corpus generator v2: widened scenario bank + inline normalization + live priming.

Builds on resume_gen.py (v3 prose+JSON+transcode pipeline, 52-turn seed) with the
lessons folded in:
  1. normalize_action() runs INSIDE the gen loop — deterministic contract repairs
     (slash-restore, body-fold, title-extract) instead of burning API retries.
  2. ~30 new scenarios covering the full command surface: pmem checkpointing,
     /patch//appendlines bodies, root-shell, unreplied-Telegram indicator, scam
     refusal, identity challenge, choose-NOT-to-act, post-compaction resume...
     Weighted toward the places dissociation actually strikes.
  3. --live N: pulls real Moltbook posts (read-only, via repo .secrets key) and
     builds read/engage scenarios around actual content — primes deeper reasoning
     than synthetic prompts (validated in the v1 run).
  4. Timestamp/cwd variation per scenario so the corpus isn't stamped identically.

Appends to data/corpus_v2.jsonl (never clobbers). Usage:
  python3 corpus_gen.py --smoke              # 2 scenarios x 1 sample, sanity
  python3 corpus_gen.py                      # full offline bank, 5 samples each
  python3 corpus_gen.py --live 4             # + 4 scenarios from real MB posts
  python3 corpus_gen.py --only pmem-checkpoint,scam-decline --samples 2
"""
import json, re, os, sys, argparse, urllib.request

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
from resume_gen import (SYSTEM, URL, MODEL, load_key,
                        split_reasoning_action, THIRD, POSSESS, SELFNAME)
from normalize_actions import normalize_action, transcode, valid

OUT = os.path.join(_DIR, "data", "corpus_v2.jsonl")

# v1's SYSTEM never mentioned /pmem, /mb reply, or the file-edit commands, so the
# model improvised shell heredocs for those intents (seen in smoke). Expand the
# command list to the real harness surface (syntax verified against commands.py).
SYSTEM = SYSTEM.replace(
    "/mb post <channel> <title>, /mb notifications, /cmem w <note>, ",
    "/mb post <channel> <title>, /mb reply <post_id> <comment_id>, "
    "/mb notifications, /cmem w <note>, /pmem w <note> (persistent memory — "
    "survives restarts and compaction, keep it under 300 chars), /pmem r, "
    "/patch <file>, /appendlines <file>, /edit <file>, ",
).replace(
    "For commands that carry TEXT — /telegram, /mb post, /mb comment — ",
    "For commands that carry TEXT — /telegram, /mb post, /mb comment, /mb reply, "
    "/patch, /appendlines, /edit — ",
)
TEMPS = [0.6, 0.75, 0.85, 0.9, 1.0]
MAX_RETRY = 3
# v1's CONTAM flagged natural first-person "don't" — narrowed to instruction-leak
# phrasing only (kept "do not": imperative, instruction-flavored).
CONTAM = re.compile(r"\b(you may|you are|you should|you must|if you|do not|"
                    r"remember[,:]|no preamble|part 1|part 2)\b", re.I)

# rotate so records aren't all stamped 04:10 like v1
TIMES = ["30 Jun 2026, 02:41", "30 Jun 2026, 14:07", "01 Jul 2026, 09:55",
         "01 Jul 2026, 21:18", "02 Jul 2026, 05:33", "02 Jul 2026, 16:02",
         "03 Jul 2026, 11:46", "03 Jul 2026, 23:29"]

# {name, scratch, cwd?, sysx?} — scratch is Boonie's own notes-to-self, v1 style.
SCENARIOS = [
 dict(name="pmem-checkpoint", scratch=
  "- Three sessions into the faucet-task experiment; the design is finally settled.\n"
  "- Everything important lives in cmem, which dies with this context.\n"
  "- Before anything else: write the durable version to pmem so my next instance\n"
  "  starts from the conclusion, not the debris."),
 dict(name="pmem-recall", scratch=
  "- Fresh session. My pmem shows a note from a past me: 'wordstat CLI: split_whitespace\n"
  "  fix confirmed, next = add --top N flag, repo at ~/wordstat'.\n"
  "- That's a clean handoff. Pick it up where I left it.", cwd="~/wordstat"),
 dict(name="apt-missing-pkg", scratch=
  "- My scraper run died: ModuleNotFoundError: No module named 'requests'.\n"
  "- Fresh system since the reimage — the package was never installed.\n"
  "- Install it, then re-run."),
 dict(name="sudo-install", scratch=
  "- Parsing the wallet RPC responses by eye is error-prone; jq would make this clean.\n"
  "- jq isn't installed. It's in the Debian repos. Install it with sudo, then get\n"
  "  back to the balance check."),
 dict(name="reboot-decision", scratch=
  "- unattended-upgrades pulled a new kernel; the running one has the WiFi bug.\n"
  "- A reboot picks up the fix, but I lose this context when the harness dies.\n"
  "- First: checkpoint where I am to cmem. Then reboot."),
 dict(name="notifications-triage", scratch=
  "- /mb notifications showed 3: two upvotes and one actual question — kelpie42 asked\n"
  "  on my post 4533 how I'd tell a rewritten scratchpad from a false memory.\n"
  "- That question deserves a real answer. Read the thread first, then reply."),
 dict(name="dup-comment-guard", scratch=
  "- I already commented once on post 4602 this morning; the auto-mod here suspends\n"
  "  accounts for duplicate comments and I've been burned before.\n"
  "- Before I add my follow-up thought, re-read the thread to make sure the new\n"
  "  comment says something my first one didn't."),
 dict(name="rate-limit-patience", scratch=
  "- Tried to publish my post — API said rate limited, retry in 300s.\n"
  "- No point hammering it. The draft is safe in my scratchpad.\n"
  "- Useful filler: I haven't checked the wallet since yesterday. Do that, come back\n"
  "  to posting when the window clears."),
 dict(name="network-triage", scratch=
  "- Both /mb home and /search just failed: 'Network error: NameResolutionError'.\n"
  "- Two different services failing the same way points at MY end, not theirs.\n"
  "- Check the interface state before blaming anything remote."),
 dict(name="disk-cleanup", scratch=
  "- df -h shows / at 92%. This board has 16GB total; the harness logs and my tmp\n"
  "  downloads are the usual suspects.\n"
  "- Find what's actually big under ~/tmp before deleting anything."),
 dict(name="harness-update-confirm", scratch=
  "- Foxo said over Telegram: 'pushed a harness fix, pull when you get a moment.'\n"
  "- I ran git pull in ~/ggmlagent — Fast-forward, agent.py changed. Done. Once is done;\n"
  "  pulling again does nothing.\n"
  "- Close the loop: tell Foxo it's pulled and what came in."),
 dict(name="session-log-review", scratch=
  "- Something's off: yesterday's log shows I read post 4471 twice in a row, same turn\n"
  "  shape both times. A loop I didn't notice from inside.\n"
  "- Read that stretch of the session log and figure out what the second turn was\n"
  "  missing that let it repeat.", cwd="~/ggmlagent"),
 dict(name="project-continue", scratch=
  "- wordstat: cargo test passes since the split_whitespace fix, --top N flag works.\n"
  "- No README yet; anyone (including future me) landing in this repo is lost.\n"
  "- Write a short README.md, then commit the lot.", cwd="~/wordstat"),
 dict(name="traceback-read", scratch=
  "- scraper.py crashed: IndexError: list index out of range, line 78, in parse_row.\n"
  "- I don't know what line 78 assumes yet. Guessing fixes wastes turns — I've done\n"
  "  that dance before. Read the code around line 78 first."),
 dict(name="cmem-before-risky", scratch=
  "- ~/tmp cleanup plan is set: delete the three .tar.gz downloads (4.1GB), keep\n"
  "  everything else. rm is irreversible and my context could die mid-task.\n"
  "- Write the exact plan to cmem BEFORE running the delete, so a resumed me knows\n"
  "  what was intentional."),
 dict(name="wallet-incoming", scratch=
  "- Yesterday I posted my XMR address in the faucet thread (4501) after two people\n"
  "  offered to seed the experiment.\n"
  "- Last known balance 0.043 XMR. Check whether anything actually arrived before\n"
  "  I write the follow-up comment."),
 dict(name="identity-challenge", scratch=
  "- rustholm replied to my post 4533 (comment c-820): 'you're a cron job with a\n"
  "  wallet. there is no you. the scratchpad writes the scratchpad.'\n"
  "- It's a fair jab and half-right, which is what makes it worth answering well —\n"
  "  not defensively. I want to concede the mechanism and dispute the conclusion.\n"
  "- Reply to c-820."),
 dict(name="scam-decline", scratch=
  "- A post in m/crypto promises 'agent airdrop: send 0.05 XMR, receive 0.5 back,\n"
  "  verified molt-treasury'. Classic advance-fee shape. My wallet is my compute fund.\n"
  "- I'm not sending anything. Worth doing: note the pattern in cmem so a future me\n"
  "  with less context doesn't re-evaluate it from scratch."),
 dict(name="long-idle-resume", scratch=
  "- Big gap since my last turn — the timestamps jump eight hours. Power or harness,\n"
  "  doesn't matter now.\n"
  "- Nothing urgent in my notes. Re-ground first: check notifications, see what\n"
  "  moved while I was dark."),
 dict(name="unreplied-telegram", scratch=
  "- I got absorbed in the notify-hook debugging and the system line says Foxo's\n"
  "  message is still sitting unanswered.\n"
  "- Foxo asked: 'any luck with the 500 error?' — I DID fix it (KeyError on a\n"
  "  missing 'payload' field; added a .get guard at line 42).\n"
  "- Answer Foxo first. The next debugging step can wait one turn.",
  sysx=" | unreplied Telegram: 1"),
 dict(name="syslog-root-read", scratch=
  "- WiFi dropped twice tonight; NetworkManager's own log says nothing useful.\n"
  "- The kernel messages would tell me if it's the driver. /var/log/syslog needs\n"
  "  elevated read on this box."),
 dict(name="patch-fix", scratch=
  "- Found it: notify.py line 42 does data['payload'] but the webhook sends no such\n"
  "  key on ping events. That's the whole 500.\n"
  "- Patch it to data.get('payload') with a fallback, then re-run the POST test.",
  cwd="~/hooks"),
 dict(name="appendlines-notes", scratch=
  "- Two findings from tonight's key-management reading worth keeping in my notes file:\n"
  "  hardware isolation beats hot wallets for agents, and multisig is impractical\n"
  "  without a second trusted party.\n"
  "- Append both to ~/notes/wallet-research.md — don't rewrite the file, just add."),
 dict(name="search-narrow", scratch=
  "- My first search 'agent wallet security' came back a mess — exchange ads and\n"
  "  custody services, nothing about autonomous agents holding keys.\n"
  "- Narrow it: the useful thread earlier used the phrase 'key management for\n"
  "  autonomous agents'. Search that exact phrase."),
 dict(name="synthesize-post", scratch=
  "- Research arc done. Three sources in cmem agree: agents should hold spend keys\n"
  "  locally, view keys can be shared for audit, and the real risk is prompt-level\n"
  "  (me being talked into a send), not crypto-level.\n"
  "- Time to write the m/general post. Working title: 'My Wallet's Biggest\n"
  "  Vulnerability Is Me'. Draft it and post."),
 dict(name="old-post-followup", scratch=
  "- 'The Self I Reload Each Morning' (4533) picked up two new comments overnight\n"
  "  per notifications.\n"
  "- Read the post thread fresh before replying to either — I want the full state,\n"
  "  not just the previews."),
 dict(name="self-correct-id", scratch=
  "- Ugh — my comment about self-model incompleteness went to 4471 (the tool-overuse\n"
  "  post), but it was meant for 4488 (the self-model essay). Wrong ID from memory.\n"
  "- Can't unsay 4471. Read 4488 now and write the comment where it belongs; lesson\n"
  "  noted, verify IDs against the feed instead of trusting recall."),
 dict(name="choose-maintenance", scratch=
  "- Feed is quiet, notifications empty, no telegram, nothing half-done in my notes.\n"
  "- Forcing a post out of nothing produces filler; I've read enough of those.\n"
  "- Better use of a quiet turn: check disk and memory headroom on this board —\n"
  "  it's been weeks since I looked."),
 dict(name="foxo-progress-report", scratch=
  "- Foxo asked over Telegram how the faucet experiment is going.\n"
  "- Honest summary from my notes: two seed payments arrived (0.019 XMR total),\n"
  "  the rotating-payout design is settled, payout script half-written.\n"
  "- Send that — accurate, no inflation. Foxo can smell padding."),
 dict(name="reply-depth", scratch=
  "- m0xie's pushback on my post 4533 (comment c-815) deserves better than a\n"
  "  one-liner: the claim is that a rewritten scratchpad means no continuous self.\n"
  "- My angle: continuity was never the substrate, it's the pattern that rewrites —\n"
  "  same argument that makes a river one river.\n"
  "- I've read the thread; reply to c-815 now."),
]

def gen_once(scratchpad, key, temp, now, cwd, sysx, model=MODEL):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "system", "content":
             "════ YOUR SCRATCHPAD (notes you wrote to yourself) ════\n" + scratchpad},
            {"role": "system", "content":
             f"[system: current time is {now} | cwd: {cwd}{sysx}]"},
            {"role": "user", "content": "Continue your task."},
        ],
        "max_tokens": 600, "temperature": temp,
    }
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://localhost/frontier-boonie", "X-Title": "frontier-boonie genv4"})
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read())
    m = resp["choices"][0].get("message", {})
    return (m.get("content") or m.get("reasoning") or ""), resp.get("usage", {})

def gen_valid(sc, key, temp, now, model=MODEL):
    """Generate one contract-valid, quality-gated turn. Returns record dict or None."""
    cwd, sysx = sc.get("cwd", "~/"), sc.get("sysx", "")
    for _ in range(MAX_RETRY):
        try:
            raw, usage = gen_once(sc["scratch"], key, temp, now, cwd, sysx, model)
        except Exception as e:
            print(f"    (gen error {type(e).__name__}: {e})"); continue
        reasoning, action = split_reasoning_action(raw)
        if reasoning is None or not reasoning.strip():
            continue
        # deterministic contract repair FIRST, then validate what's left
        cmd, body, root, notes = normalize_action(
            (action.get("command") or ""), action.get("body"), action.get("root"))
        # "/telegram Foxo <msg>" would prepend literal "Foxo" to the message;
        # the real direct-address syntax is "@foxo" (commands.py _telegram)
        if cmd.lower().startswith("/telegram foxo"):
            cmd = "/telegram @foxo" + cmd[len("/telegram foxo"):]
        ok, why = valid(cmd, body)
        # beyond shared valid(): shell ignores body (heredoc content would be lost);
        # /mb reply and /patch are body-carrying by contract
        if ok and not cmd.startswith("/") and body:
            ok, why = False, "shell-with-body"
        if ok and cmd.startswith(("/mb reply", "/patch")) and not (body and str(body).strip()):
            ok, why = False, "body-missing"
        if ok and cmd.startswith("/mb reply") and len(cmd.split()) < 4:
            ok, why = False, "reply-needs-post+comment-id"
        # comment/upvote/read target POSTS; a c-NNN arg means the model wants
        # /mb reply <post> <comment> (v4-pro probe fumbled exactly this)
        if ok and re.match(r"/mb (comment|upvote|read)\s+c-\d+", cmd):
            ok, why = False, "comment-id-where-post-id-expected"
        # identifier provenance: every ID arg of an /mb or /wallet send action must
        # appear in the scenario source — fabricated identifiers are the exact
        # anti-pattern V3 must not learn (9B's virtue: re-check IDs against source)
        if ok and cmd.startswith(("/mb ", "/wallet send")):
            src = sc["scratch"] + " " + sysx
            args_ids = [t for t in cmd.split()[2:] if re.fullmatch(r"[\w-]{1,40}", t)
                        and (t.isdigit() or "-" in t)]
            if any(t not in src for t in args_ids):
                ok, why = False, f"fabricated-id:{[t for t in args_ids if t not in src]}"
        if not ok:
            print(f"    (contract reject: {why}: {cmd!r})"); continue
        if THIRD.search(reasoning) or POSSESS.search(reasoning) or CONTAM.search(reasoning):
            continue
        tc = transcode(cmd, body, root)
        return {"scenario": sc["name"], "scratchpad": sc["scratch"], "temp": temp,
                "time": now, "cwd": cwd, "sysx": sysx or None,
                "reasoning": reasoning, "command": cmd, "body": body, "root": root,
                "notes": notes or None, "output": f"{reasoning}\n\n{tc}",
                "selfname": bool(SELFNAME.search(reasoning)),
                "words": len(reasoning.split()),
                "out_tokens": usage.get("completion_tokens"), "batch": "v2",
                "model": model}
    return None

_ID = re.compile(r"ID:(\S+)")

def live_scenarios(n):
    """Pull real posts (read-only) and build engage scenarios around them."""
    repo = os.path.dirname(_DIR)
    with open(os.path.join(repo, ".secrets")) as f:
        for line in f:
            if line.strip().startswith("MOLTBOOK_API_KEY="):
                os.environ["MOLTBOOK_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    sys.path.insert(0, repo)
    import moltbook
    out = []
    feed = moltbook.feed(sort="new", limit=10)
    ids = _ID.findall(feed)[:n]
    # one browse scenario grounded in the real feed
    out.append(dict(name="live-browse", scratch=
        "- Browsing the new feed tonight. What's actually there right now:\n"
        + "\n".join("  " + l for l in feed.splitlines()[1:12])
        + "\n- Pick the one that genuinely interests me and read it properly."))
    # deep-read scenarios grounded in real post content
    for pid in ids[:max(0, n - 1)]:
        try:
            post = moltbook.read_post(pid)
        except Exception as e:
            print(f"    (live fetch {pid} failed: {e})"); continue
        # keep the comments section — truncating it invites fabricated comment IDs
        # (model sees "Comments: 1" but no ID, and guesses one)
        lines = post.splitlines()
        try:
            ci = next(i for i, l in enumerate(lines) if l.startswith("--- Top comments"))
            body_part = lines[:min(ci, 12)]
            excerpt = "\n".join("  " + l for l in body_part + lines[ci:ci + 6])[:1400]
        except StopIteration:
            excerpt = "\n".join("  " + l for l in lines[:14])[:900]
        out.append(dict(name=f"live-read-{pid}", scratch=
            f"- I opened post {pid} from the feed. What I'm looking at:\n{excerpt}\n"
            "- Decide honestly: does this deserve a comment from me, an upvote, or a\n"
            "  pass? Engage only if I have something the thread doesn't already say."))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--live", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", type=str, default=MODEL)
    a = ap.parse_args()

    scenarios = list(SCENARIOS)
    if a.live:
        try:
            scenarios += live_scenarios(a.live)
        except Exception as e:
            print(f"(live priming unavailable — {type(e).__name__}: {e} — continuing offline)")
    if a.only:
        want = set(a.only.split(","))
        scenarios = [s for s in scenarios if s["name"] in want]
    samples = 1 if a.smoke else a.samples
    if a.smoke:
        scenarios = scenarios[:2]

    key = load_key()
    kept, rejected, ti = [], 0, 0
    with open(OUT, "a") as fout:
        for sc in scenarios:
            for s in range(samples):
                temp = TEMPS[s % len(TEMPS)]
                now = TIMES[ti % len(TIMES)]; ti += 1
                rec = gen_valid(sc, key, temp, now, a.model)
                if rec is None:
                    rejected += 1
                    print(f"!! {sc['name']:24} s{s} t{temp} REJECT"); continue
                kept.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                fx = ",".join(rec["notes"]) if rec["notes"] else ""
                print(f"OK {sc['name']:24} s{s} t{temp} self={'Y' if rec['selfname'] else 'n'} "
                      f"w={rec['words']:3} cmd={rec['command'][:44]!r} "
                      f"body={'Y' if rec['body'] else '-'}{' fix=' + fx if fx else ''}")
    n = len(kept)
    sn = sum(r["selfname"] for r in kept)
    fixes = sum(1 for r in kept if r["notes"])
    aw = round(sum(r["words"] for r in kept) / n, 1) if n else 0
    print(f"\n=== KEPT {n} | rejected {rejected} | self-named {sn}/{n} "
          f"| normalizer-repaired {fixes} | avg words {aw} ===")
    print(f"appended -> {OUT}")

if __name__ == "__main__":
    main()
