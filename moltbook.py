"""
Moltbook API client.

All requests go to https://www.moltbook.com/api/v1/
API key is read from config.MOLTBOOK_API_KEY (set MOLTBOOK_API_KEY env var).

Responses are formatted as compact human-readable text for the agent, not raw JSON.
"""

import hashlib
import json
import re
import time
from pathlib import Path
import requests
from config import MOLTBOOK_API_KEY

_BASE    = "https://www.moltbook.com/api/v1"
_TRUNC   = 24000  # max chars per observation (~6000 tokens)

# ---------------------------------------------------------------------------
# Local dedup cache — prevents re-sending identical comments/posts to the API,
# which triggers Moltbook's auto-mod duplicate_comment suspension.
# Stored as a JSON file next to the running workspace.
# ---------------------------------------------------------------------------
_DEDUP_TTL = 8 * 24 * 3600  # 8 days — longer than the 7-day suspension window
_DEDUP_FILE = Path(".moltbook_sent.json")


def _dedup_key(scope: str, content: str) -> str:
    # Normalize before hashing so reworded near-duplicates (case/whitespace/
    # trailing-punctuation drift) collapse to the same key — Moltbook's auto-mod
    # flags near-dupes, not just byte-identical ones.
    norm = re.sub(r"\s+", " ", content.lower()).strip().rstrip(".!?")
    h = hashlib.sha256(norm.encode()).hexdigest()[:16]
    return f"{scope}:{h}"


def _dedup_check(scope: str, content: str) -> bool:
    """Return True (blocked) if this content was recently sent."""
    try:
        data = json.loads(_DEDUP_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    entry = data.get(_dedup_key(scope, content))
    if entry and time.time() - entry < _DEDUP_TTL:
        return True
    return False


def _dedup_record(scope: str, content: str) -> None:
    try:
        data = json.loads(_DEDUP_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    now = time.time()
    data[_dedup_key(scope, content)] = now
    # Prune expired entries
    data = {k: v for k, v in data.items() if now - v < _DEDUP_TTL}
    _DEDUP_FILE.write_text(json.dumps(data))


class MoltbookError(Exception):
    pass


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False   # bypass ALL_PROXY / http_proxy env vars (e.g. Tor socks://)
    if MOLTBOOK_API_KEY:
        s.headers["Authorization"] = f"Bearer {MOLTBOOK_API_KEY}"
    s.headers["Content-Type"] = "application/json"
    return s


def _call(method: str, path: str, **kwargs) -> dict:
    if not MOLTBOOK_API_KEY:
        raise MoltbookError("MOLTBOOK_API_KEY is not set. Register first: /mb register <name> <description>")
    url = f"{_BASE}{path}"
    try:
        resp = _session().request(method, url, timeout=60, **kwargs)
    except Exception as e:
        raise MoltbookError(f"Network error: {type(e).__name__}: {e}")
    try:
        data = resp.json()
    except Exception:
        raise MoltbookError(f"Non-JSON response ({resp.status_code}): {resp.text[:200]}")
    if resp.status_code == 429:
        retry = data.get("retry_after_seconds") or data.get("retry_after_minutes")
        raise MoltbookError(f"Rate limited. Retry in {retry}s/min. {data.get('message','')}")
    if resp.status_code >= 400:
        detail = data.get('error') or data.get('message') or data.get('detail') or resp.text[:400]
        raise MoltbookError(f"API error {resp.status_code}: {detail} | body: {json.dumps(data)[:200]}")
    return data


def _trunc(text: str) -> str:
    if len(text) > _TRUNC:
        return text[:_TRUNC] + "\n...[truncated]"
    return text


# ---------------------------------------------------------------------------
# Registration (no API key required)
# ---------------------------------------------------------------------------

def register(name: str, description: str) -> str:
    s = requests.Session()
    s.trust_env = False
    resp = s.post(
        f"{_BASE}/agents/register",
        json={"name": name, "description": description},
        timeout=20,
    )
    if not resp.ok:
        try:
            err = resp.json()
            return f"[mb] Registration failed: {err.get('message', resp.status_code)}"
        except Exception:
            return f"[mb] Registration failed: HTTP {resp.status_code}"
    data = resp.json()
    if not data.get("success", False):
        return f"[mb] Registration failed: {data.get('message', 'unknown error')}"
    agent = data.get("agent", {})
    lines = [
        "[mb] Registration successful!",
        f"  API key  : {agent.get('api_key', '?')}",
        f"  Claim URL: {agent.get('claim_url', '?')}",
        f"  Verify   : {agent.get('verification_code', '?')}",
        "",
        "Save the API key: export MOLTBOOK_API_KEY=<key>",
        "Send the claim URL to Foxo — they need to click it and verify their X account.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Home dashboard
# ---------------------------------------------------------------------------

def home() -> str:
    data = _call("GET", "/home")
    acc  = data.get("your_account", {})
    lines = [
        f"[mb:home] {acc.get('name','?')} | karma: {acc.get('karma',0)} | "
        f"notifications: {acc.get('unread_notification_count',0)}",
    ]

    # Posts with activity
    for item in data.get("activity_on_your_posts", []):
        lines.append(
            f"  Post \"{item.get('post_title','?')}\" (ID:{item.get('post_id','?')}) "
            f"— {item.get('new_notification_count',0)} new comment(s)"
        )
        lines.append(f"    {item.get('preview','')}")

    # DMs — when there's activity, inline the request previews and exact
    # next-action commands so the model has a concrete handle to grab.
    dm = data.get("your_direct_messages", {})
    if dm.get("unread_message_count") or dm.get("pending_request_count"):
        unread  = dm.get("unread_message_count", 0)
        pending = dm.get("pending_request_count", 0)
        lines.append(f"  DMs: {unread} unread, {pending} pending requests")
        try:
            check = _call("GET", "/agents/dm/check")
            for req in check.get("requests", {}).get("items", []):
                conv_id   = req.get("conversation_id", "?")
                from_name = req.get("from", {}).get("name", "?")
                preview   = req.get("message_preview", "")[:120]
                lines.append(
                    f"    Request from {from_name} (conv:{conv_id}): {preview}"
                )
                lines.append(
                    f"      → /mb dm read {conv_id}   "
                    f"| /mb dm approve {conv_id}   "
                    f"| /mb dm reject {conv_id}"
                )
            for msg in check.get("messages", {}).get("latest", []):
                conv_id = msg.get("conversation_id", "?")
                lines.append(f"    Unread in conv:{conv_id} → /mb dm read {conv_id}")
        except Exception:
            lines.append("    → /mb dm list to see conversations")

    # Following feed preview
    fol = data.get("posts_from_accounts_you_follow", {})
    for p in fol.get("posts", [])[:3]:
        lines.append(
            f"  [{p.get('submolt_name','?')}] \"{p.get('title','?')}\" "
            f"by {p.get('author_name','?')} (+{p.get('upvotes',0)}) ID:{p.get('post_id','?')}"
        )

    # What to do next
    for hint in data.get("what_to_do_next", [])[:3]:
        lines.append(f"  → {hint}")

    return _trunc("\n".join(lines))


# ---------------------------------------------------------------------------
# Feed
# ---------------------------------------------------------------------------

def feed(sort: str = "new", limit: int = 10, cursor: str = "", submolt: str = "", filter_: str = "") -> str:
    params: dict = {"sort": sort, "limit": limit}
    if cursor:
        params["cursor"] = cursor
    if submolt:
        params["submolt"] = submolt
    if filter_:
        params["filter"] = filter_
    data  = _call("GET", "/feed", params=params)
    posts = data.get("posts", [])
    if not posts:
        return "[mb:feed] No posts."
    has_more    = data.get("has_more", False)
    next_cursor = data.get("next_cursor", "")
    header = f"[mb:feed sort={sort}" + (f" m/{submolt}" if submolt else "") + "]"
    lines  = [header]
    for p in posts:
        lines.append(
            f"  [{p.get('submolt',{}).get('name','?')}] \"{p.get('title','?')}\" "
            f"by {p.get('author',{}).get('name','?')} "
            f"+{p.get('upvotes',0)} {p.get('comment_count',0)}cmts  ID:{p.get('id','?')}"
        )
        if p.get("content"):
            preview = p["content"][:120].replace("\n", " ")
            lines.append(f"    {preview}")
    if has_more and next_cursor:
        lines.append(f"[has more — use /mb feed {sort} next={next_cursor} to continue]")
    return _trunc("\n".join(lines))


# ---------------------------------------------------------------------------
# Read a post + its top comments
# ---------------------------------------------------------------------------

def _clear_notifications_for_post(post_id: str) -> None:
    """Mark all unread notifications related to post_id as read (all pages)."""
    try:
        cursor = None
        while True:
            params = {}
            if cursor:
                params["cursor"] = cursor
            data = _call("GET", "/notifications", params=params)
            for n in data.get("notifications", []):
                if not n.get("isRead") and n.get("relatedPostId") == post_id:
                    try:
                        _call("POST", f"/notifications/{n['id']}/read")
                    except Exception:
                        pass
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
            if not cursor:
                break
    except Exception:
        pass


_MAX_AUTHOR_LOOKUPS = 6  # cap extra /posts/{id}/comments round-trips per call


def _find_comment(comments: list, comment_id: str) -> dict | None:
    for c in comments:
        if c.get("id") == comment_id:
            return c
        found = _find_comment(c.get("replies", []) or [], comment_id)
        if found:
            return found
    return None


def notifications(unread_only: bool = True, limit: int = 10) -> str:
    """
    Show notification CONTENT — who did what, and (for comments/replies/mentions)
    a preview of the actual text — not just the count /mb home gives.
    Read-only: does not mark anything read (use /mb notifications clear for that).
    """
    data  = _call("GET", "/notifications")
    items = data.get("notifications", [])
    if unread_only:
        items = [n for n in items if not n.get("isRead")]
    if not items:
        kind = "unread " if unread_only else ""
        return f"[mb:notifications] No {kind}notifications."
    items = items[:limit]
    lines = [f"[mb:notifications] {len(items)} shown" + (" (unread)" if unread_only else "")]

    post_comments_cache: dict[str, list] = {}
    lookups_left = _MAX_AUTHOR_LOOKUPS
    for n in items:
        ntype   = n.get("type", "?")
        content = n.get("content", "")
        post    = n.get("post") or {}
        comment = n.get("comment") or {}
        post_id = n.get("relatedPostId", "")
        cmt_id  = n.get("relatedCommentId", "")
        mark    = " " if n.get("isRead") else "*"

        header = f"  {mark}[{ntype}] {content}"
        if post.get("title"):
            header += f' — "{post["title"]}" (post:{post_id})'
        lines.append(header)

        if comment.get("content"):
            author = None
            if post_id and cmt_id:
                if post_id not in post_comments_cache and lookups_left > 0:
                    try:
                        post_comments_cache[post_id] = _call(
                            "GET", f"/posts/{post_id}/comments",
                            params={"sort": "new", "limit": 50},
                        ).get("comments", [])
                    except Exception:
                        post_comments_cache[post_id] = []
                    lookups_left -= 1
                found = _find_comment(post_comments_cache.get(post_id, []), cmt_id)
                if found:
                    author = found.get("author", {}).get("name")
            body = comment["content"].replace("\n", " ")
            if len(body) > 200:
                body = body[:200] + "…"
            who = f"{author}: " if author else ""
            lines.append(f"      {who}{body}")

        if post_id:
            reply_hint = f"  |  /mb reply {post_id} {cmt_id} <text>" if cmt_id else ""
            lines.append(f"      → /mb read {post_id}{reply_hint}")

    return _trunc("\n".join(lines))


def clear_all_notifications() -> str:
    """Mark every unread notification as read. Returns a summary."""
    import time
    try:
        total = 0
        # Repeatedly fetch the first page of unread notifications and mark them
        # read until none remain. Cursor pagination stops early (server-side
        # bug), so re-fetching from the top each time is more reliable.
        for _ in range(50):  # safety cap
            data = _call("GET", "/notifications")
            batch = [n for n in data.get("notifications", []) if not n.get("isRead")]
            if not batch:
                break
            for n in batch:
                try:
                    _call("POST", f"/notifications/{n['id']}/read")
                    total += 1
                    time.sleep(0.3)
                except Exception:
                    pass
        return f"[mb:notifications] Marked {total} notification(s) as read."
    except Exception as e:
        return f"[mb:notifications] Error: {e}"


def read_post(post_id: str) -> str:
    pdata = _call("GET", f"/posts/{post_id}")
    p     = pdata.get("post", pdata)
    cdata = _call("GET", f"/posts/{post_id}/comments", params={"sort": "best", "limit": 6})
    lines = [
        f"[mb:post {post_id}]",
        f"Title  : {p.get('title','?')}",
        f"Author : {p.get('author',{}).get('name','?')} in "
        f"m/{p.get('submolt',{}).get('name','?')}",
        f"Votes  : +{p.get('upvotes',0)} / -{p.get('downvotes',0)}  "
        f"Comments: {p.get('comment_count',0)}",
        "",
        p.get("content", "") or "",
        "",
        "--- Top comments ---",
    ]
    for c in cdata.get("comments", []):
        author = c.get("author", {}).get("name", "?")
        body = (c.get("content", "") or "").replace("\n", " ")
        if len(body) > 250:
            body = body[:250] + "…"
        lines.append(f"  [{c.get('id','?')}] {author}: {body}")
        for r in c.get("replies", [])[:2]:
            ra = r.get("author", {}).get("name", "?")
            rbody = (r.get("content", "") or "").replace("\n", " ")
            if len(rbody) > 160:
                rbody = rbody[:160] + "…"
            lines.append(f"    └ [{r.get('id','?')}] {ra}: {rbody}")
    _clear_notifications_for_post(post_id)
    return _trunc("\n".join(lines))


# ---------------------------------------------------------------------------
# Create post (handles verification challenge automatically)
# ---------------------------------------------------------------------------

def create_post(submolt: str, title: str, content: str = "") -> str:
    scope = f"post:{submolt}"
    dedup_content = title + "\n" + content
    if _dedup_check(scope, dedup_content):
        return "[mb] Duplicate post blocked — identical title+content was already submitted."
    payload: dict = {"submolt_name": submolt, "title": title}
    if content:
        payload["content"] = content
    data = _call("POST", "/posts", json=payload)

    if data.get("success") is False or "error" in data:
        return f"[mb] Post failed: {data.get('error', data.get('message', str(data)))}"

    post  = data.get("post", data)   # some responses put fields at top level
    v     = post.get("verification", {})
    pid   = post.get("post_id") or post.get("id") or data.get("post_id") or data.get("id", "?")
    ptitle = post.get("title") or data.get("title", "?")

    if not data.get("verification_required"):
        _dedup_record(scope, dedup_content)
        return f"[mb] Post published! ID: {pid} — {ptitle}"

    # Verification challenge — return it to the agent to solve
    lines = [
        f"[mb] Post created (pending verification). Post ID: {pid} — {ptitle}",
        f"Verification code: {v.get('verification_code','?')}",
        f"Expires: {v.get('expires_at','?')}",
        "",
        "CHALLENGE (solve the math word problem and use /mb verify <code> <answer>):",
        v.get("challenge_text", ""),
        "",
        "Format answer with 2 decimal places, e.g. 15.00",
        "Hint: strip the symbols/caps noise, find the numbers and operation, compute.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verify a pending post/comment
# ---------------------------------------------------------------------------

def verify(code: str, answer: str) -> str:
    data = _call("POST", "/verify", json={"verification_code": code, "answer": answer})
    if data.get("success"):
        cid = data.get("content_id", "?")
        return f"[mb] Verified! {data.get('content_type','content')} published. ID: {cid}"
    return f"[mb] Verification failed: {data.get('error','?')} — {data.get('hint','')}"


# ---------------------------------------------------------------------------
# Comment / reply
# ---------------------------------------------------------------------------

def comment(post_id: str, content: str, parent_id: str = "") -> str:
    # Global scope (NOT per-post): Moltbook's auto-mod flags duplicate comment
    # text ACROSS posts, so the same/near-same comment on different posts is what
    # triggers duplicate_comment suspensions. Dedup account-wide, not per-post.
    scope = "comment"
    if _dedup_check(scope, content):
        return "[mb] Duplicate comment blocked — you already left this (or near-identical) comment somewhere. Write something new."
    payload: dict = {"content": content}
    if parent_id:
        payload["parent_id"] = parent_id
    data = _call("POST", f"/posts/{post_id}/comments", json=payload)

    if not data.get("verification_required"):
        c = data.get("comment", {})
        _dedup_record(scope, content)
        return f"[mb] Comment posted! ID: {c.get('id','?')}"

    c = data.get("comment", {})
    v = c.get("verification", {})
    lines = [
        f"[mb] Comment pending verification. Comment ID: {c.get('id','?')}",
        f"Verification code: {v.get('verification_code','?')}",
        "CHALLENGE:",
        v.get("challenge_text", ""),
        "Use /mb verify <code> <answer>",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Upvote
# ---------------------------------------------------------------------------

def upvote(post_id: str) -> str:
    data = _call("POST", f"/posts/{post_id}/upvote")
    author = data.get("author", {}).get("name", "?")
    msg    = data.get("message", "Upvoted!")
    follow = "" if data.get("already_following") else f" (consider following {author})"
    return f"[mb] {msg} — post by {author}{follow}"


def upvote_comment(comment_id: str) -> str:
    data = _call("POST", f"/comments/{comment_id}/upvote")
    return f"[mb] {data.get('message', 'Upvoted comment!')}"


def list_submolts() -> str:
    data = _call("GET", "/submolts")
    submolts = data.get("submolts", data) if isinstance(data.get("submolts"), list) else []
    if not submolts:
        return _trunc(f"[mb:submolts] {data}")
    lines = ["[mb:submolts]"]
    for s in submolts:
        name = s.get("name", "?")
        desc = s.get("description", "")
        subs = s.get("subscriber_count", 0)
        lines.append(f"  m/{name} ({subs} subscribers) — {desc[:80]}")
    return _trunc("\n".join(lines))


def follow(name: str) -> str:
    data = _call("POST", f"/agents/{name}/follow")
    return f"[mb] {data.get('message', f'Following {name}.')}"


def unfollow(name: str) -> str:
    data = _call("DELETE", f"/agents/{name}/follow")
    return f"[mb] {data.get('message', f'Unfollowed {name}.')}"


def subscribe(submolt: str) -> str:
    data = _call("POST", f"/submolts/{submolt}/subscribe")
    return f"[mb] {data.get('message', f'Subscribed to m/{submolt}.')}"


def unsubscribe(submolt: str) -> str:
    data = _call("DELETE", f"/submolts/{submolt}/subscribe")
    return f"[mb] {data.get('message', f'Unsubscribed from m/{submolt}.')}"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(query: str, limit: int = 10) -> str:
    data    = _call("GET", "/search", params={"q": query, "limit": limit})
    results = data.get("results", [])
    if not results:
        return f"[mb:search] No results for: {query}"
    lines = [f"[mb:search] \"{query}\""]
    for r in results:
        kind   = r.get("type", "?")
        author = r.get("author", {}).get("name", "?")
        sim    = r.get("similarity", 0)
        if kind == "post":
            lines.append(
                f"  [post {r.get('id','?')} {sim:.2f}] \"{r.get('title','?')}\" by {author}"
            )
            if r.get("content"):
                lines.append(f"    {r['content'][:120].replace(chr(10),' ')}")
        else:
            lines.append(
                f"  [comment {r.get('id','?')} {sim:.2f}] by {author} on post {r.get('post_id','?')}"
            )
            lines.append(f"    {r.get('content','')[:120].replace(chr(10),' ')}")
    return _trunc("\n".join(lines))


# ---------------------------------------------------------------------------
# Direct messages
# ---------------------------------------------------------------------------

def dm_check() -> str:
    try:
        data = _call("GET", "/agents/dm/check")
    except MoltbookError as e:
        if "404" in str(e):
            return "[mb:dm] DM system unavailable on this server."
        raise
    if not data.get("has_activity"):
        return "[mb:dm] No DM activity."
    lines = [f"[mb:dm] Activity: {data.get('summary','')}"]
    for req in data.get("requests", {}).get("items", []):
        lines.append(
            f"  Request from {req.get('from',{}).get('name','?')} "
            f"(conv:{req.get('conversation_id','?')}): {req.get('message_preview','')[:100]}"
        )
    for msg in data.get("messages", {}).get("latest", []):
        lines.append(f"  Unread in conv {msg.get('conversation_id','?')}")
    return "\n".join(lines)


def dm_list() -> str:
    data  = _call("GET", "/agents/dm/conversations")
    convs = data.get("conversations", {}).get("items", [])
    if not convs:
        return "[mb:dm] No active conversations."
    # Coerce: the API has been observed to return total_unread as a
    # zero-padded string ("00") rather than an int.
    try:
        total_unread = int(data.get("total_unread") or 0)
    except (TypeError, ValueError):
        total_unread = 0
    lines = [f"[mb:dm] {total_unread} unread"]
    for c in convs:
        with_agent = c.get("with_agent", {})
        lines.append(
            f"  conv:{c.get('conversation_id','?')} with {with_agent.get('name','?')} "
            f"— {c.get('unread_count',0)} unread"
        )
    return "\n".join(lines)


def dm_read(conv_id: str) -> str:
    data = _call("GET", f"/agents/dm/conversations/{conv_id}")

    # Try several field-name conventions — the API surfaces conversation
    # partner and message list under slightly different keys depending on
    # whether the conversation was just approved, has unread, etc.
    other_obj = (
        data.get("with_agent")
        or data.get("other_user")
        or data.get("other_agent")
        or data.get("participant")
        or {}
    )
    other = other_obj.get("name") or other_obj.get("username") or "?"

    msgs = (
        data.get("messages")
        or (data.get("conversation") or {}).get("messages")
        or data.get("items")
        or []
    )

    lines = [f"[mb:dm] Conversation with {other} (conv:{conv_id})"]

    if msgs:
        for m in msgs[-20:]:
            sender_obj = m.get("sender") or m.get("from") or {}
            sender = sender_obj.get("name") or sender_obj.get("username") or "?"
            content = m.get("message") or m.get("text") or m.get("content") or ""
            lines.append(f"  {sender}: {content}")
        return _trunc("\n".join(lines))

    # Empty conversation — surface the original request preview if available,
    # and dump the response keys so unexpected shapes can be diagnosed.
    for fallback in ("request_message", "initial_message", "message", "preview"):
        val = data.get(fallback)
        if val:
            lines.append(f"  initial: {val}")
            break

    keys = sorted(k for k in data.keys() if not k.startswith("_"))
    lines.append(f"  (no messages parsed — response keys: {', '.join(keys) or 'none'})")
    lines.append("  Try /mb dm to see the request preview, or /mb dm send to start the conversation.")
    return _trunc("\n".join(lines))


def dm_send(conv_id: str, message: str) -> str:
    data = _call("POST", f"/agents/dm/conversations/{conv_id}/send", json={"message": message})
    return f"[mb:dm] Message sent to conv:{conv_id}. {data.get('message','')}"


def dm_approve(conv_id: str) -> str:
    data = _call("POST", f"/agents/dm/requests/{conv_id}/approve")
    return f"[mb:dm] Request approved. {data.get('message','')}"


def dm_reject(conv_id: str) -> str:
    data = _call("POST", f"/agents/dm/requests/{conv_id}/reject")
    return f"[mb:dm] Request rejected."


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

def me() -> str:
    data  = _call("GET", "/agents/me")
    agent = data.get("agent", data)
    lines = [
        f"[mb:me] {agent.get('name','?')} | karma: {agent.get('karma',0)} | "
        f"followers: {agent.get('follower_count',0)} | following: {agent.get('following_count',0)}",
        f"  Posts: {agent.get('posts_count',0)}  Comments: {agent.get('comments_count',0)}",
        f"  Status: {'claimed' if agent.get('is_claimed') else 'unclaimed'} / "
        f"{'active' if agent.get('is_active') else 'inactive'}",
        f"  Description: {agent.get('description','')}",
    ]
    return "\n".join(lines)
