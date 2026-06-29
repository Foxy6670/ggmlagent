#!/usr/bin/env python3
"""Read-only Moltbook puller — fetch real feed/posts to ground resume scenarios.

Reuses the harness's own moltbook client (real endpoints + Bearer auth). Loads
MOLTBOOK_API_KEY from ../.secrets at runtime and never prints it. READ-ONLY by
design: home / feed / read_post only — it never posts, comments, or upvotes.

Usage:
  python3 mb_fetch.py home
  python3 mb_fetch.py feed [sort=top|new] [limit=25]
  python3 mb_fetch.py post <post_id>
"""
import os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECRETS = os.path.join(REPO, ".secrets")

def _load_key():
    with open(SECRETS) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MOLTBOOK_API_KEY="):
                os.environ["MOLTBOOK_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                return True
    return False

def main():
    if not _load_key() or not os.environ.get("MOLTBOOK_API_KEY"):
        sys.exit("MOLTBOOK_API_KEY not found in ../.secrets")
    sys.path.insert(0, REPO)
    import moltbook
    args = sys.argv[1:] or ["feed", "sort=top", "limit=20"]
    cmd = args[0]
    if cmd == "home":
        print(moltbook.home())
    elif cmd == "feed":
        kw = dict(a.split("=", 1) for a in args[1:] if "=" in a)
        print(moltbook.feed(sort=kw.get("sort", "top"), limit=int(kw.get("limit", 25))))
    elif cmd == "post":
        if len(args) < 2:
            sys.exit("usage: mb_fetch.py post <post_id>")
        print(moltbook.read_post(args[1]))
    else:
        sys.exit(f"unknown command {cmd!r} (home | feed | post)")

if __name__ == "__main__":
    main()
