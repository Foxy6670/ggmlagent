"""
Simulation mode for teleop and dry-runs.

When enabled, replaces the real Telegram handler and intercepts Moltbook
write actions so the operator can drive the agent through scenarios without
committing real social actions. File I/O, web I/O, and Moltbook *reads*
still hit the real systems — those are inputs to decision-making and the
agent should see real cause and effect for them.

Operator-only meta-commands (teleop only, parsed before dispatch):
  /tgin <text>    inject fake incoming Telegram message
  /tgout <text>   inject fake outgoing Telegram message (pre-load history)
"""

import time
from dataclasses import dataclass, field


_MB_WRITE_ACTIONS = {
    "post", "comment", "reply",
    "upvote", "upvote-comment",
    "follow", "unfollow",
    "subscribe", "unsubscribe",
    "verify",
}
_MB_DM_WRITE_ACTIONS = {"send", "approve", "reject"}


def is_mb_write(sub: str, dsub: "str | None" = None) -> bool:
    """True if /mb <sub> [<dsub>] is a write action that simulation intercepts."""
    if sub == "dm" and dsub is not None:
        return dsub in _MB_DM_WRITE_ACTIONS
    return sub in _MB_WRITE_ACTIONS


@dataclass
class SimState:
    """In-memory state for simulated Telegram and Moltbook writes."""

    _history:  list[dict] = field(default_factory=list)
    _undrained: list[int] = field(default_factory=list)  # indices into _history

    # ------------------------------------------------------------------
    # Telegram surface — duck-types telegram_handler module
    # ------------------------------------------------------------------

    def send(self, message: str) -> str:
        self._history.append({
            "direction": "out", "from": "Boonie",
            "text": message, "ts": time.time(), "sim": True,
        })
        return "[telegram] Message sent to Foxo. (simulated)"

    def drain_inbox(self) -> list[dict]:
        if not self._undrained:
            return []
        out = [self._history[i] for i in self._undrained]
        self._undrained.clear()
        return out

    def history(self) -> list[dict]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Operator injection (teleop meta-commands)
    # ------------------------------------------------------------------

    def inject_in(self, text: str, sender: str = "Foxo") -> None:
        self._history.append({
            "direction": "in", "from": sender,
            "text": text, "ts": time.time(), "sim": True,
        })
        self._undrained.append(len(self._history) - 1)

    def inject_out(self, text: str) -> None:
        self._history.append({
            "direction": "out", "from": "Boonie",
            "text": text, "ts": time.time(), "sim": True,
        })

    # ------------------------------------------------------------------
    # Moltbook write interception
    # ------------------------------------------------------------------

    def mb_write(self, sub: str, dsub: "str | None", args: list[str]) -> str:
        """Synthetic success response for a Moltbook write action.

        Format mimics the real moltbook.py responses so the agent sees a
        familiar observation shape — same `[mb] ...` prefix, same single-line
        terseness — with a trailing `(simulated)` marker for the operator's
        benefit. The marker would be a training-data leak in live mode, but
        simulated sessions shouldn't be used as training data anyway.
        """
        target = args[0] if args else "?"
        if sub == "post":
            return "[mb] Post created. (simulated — id=sim_post_42)"
        if sub == "comment":
            return "[mb] Comment posted. (simulated)"
        if sub == "reply":
            return "[mb] Reply posted. (simulated)"
        if sub == "upvote":
            return "[mb] Upvoted. (simulated)"
        if sub == "upvote-comment":
            return "[mb] Comment upvoted. (simulated)"
        if sub == "follow":
            return f"[mb] Now following {target}. (simulated)"
        if sub == "unfollow":
            return f"[mb] Unfollowed {target}. (simulated)"
        if sub == "subscribe":
            return f"[mb] Subscribed to {target}. (simulated)"
        if sub == "unsubscribe":
            return f"[mb] Unsubscribed from {target}. (simulated)"
        if sub == "verify":
            return "[mb] Verification accepted. (simulated)"
        if sub == "dm" and dsub == "send":
            return "[mb] DM sent. (simulated)"
        if sub == "dm" and dsub == "approve":
            return "[mb] DM request approved. (simulated)"
        if sub == "dm" and dsub == "reject":
            return "[mb] DM request rejected. (simulated)"
        return f"[mb] {sub} action simulated."
