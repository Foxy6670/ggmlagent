"""
Shared secret-redaction for anything that might end up in training data.

Broader than watch_*.py's log-display redaction, which only needs to catch
env-var-style KEY=value for a human skimming a terminal. Training data comes
from pmem/cmem prose, which states secrets in natural language --
"API key <value>", not "API_KEY=<value>" -- so the pattern has to match a
label followed by a token-shaped value, not just an env-var assignment.

Heuristic, not a real secret scanner: the value must be adjacent to a
label word (key/token/secret/password) AND contain a digit, so it catches
"API key cmrana3lh0004i6041zvqw7cg" without also eating "the key insight
is..." (no digit, doesn't match). This trades perfect precision for keeping
false positives on ordinary prose low -- it's a first automated pass, not a
substitute for a human spot-check before anything derived from real Boonie
logs goes into a corpus.

Deliberately does NOT touch wallet addresses. Those are public identifiers
Boonie shares on purpose (send-only, no spend authority) -- not secrets.
"""

import re

# "Bearer" is its own alternative, not just an optional prefix -- HTTP auth
# headers put the value directly after it ("Authorization: Bearer <token>"),
# no "key"/"token" word in between. Separator includes quote characters for
# JSON shape ("apiKey":"value") and allows zero chars between a compound
# label and "key" (apiKey, botToken -- camelCase, no separator at all).
_LABELED_TOKEN = re.compile(
    r"(?i)\b((?:(?:api|bot|access)[\s_-]?)?(?:key|token|secret|password)s?|bearer)\b"
    r"([\s:=\"']+)"
    r"(?=[A-Za-z0-9_\-.]*\d)([A-Za-z0-9_\-.]{8,})"
)
_ENV_STYLE = re.compile(r"(?i)([A-Z_]*(TOKEN|KEY|SECRET|PASSWORD)[A-Z_]*=)\S+")
_MOLTBOOK_KEY = re.compile(r"moltbook_sk_[A-Za-z0-9]+")
_TG_TOKEN = re.compile(r"\d{8,10}:[A-Za-z0-9_-]{35,}")


def redact(text: str) -> str:
    text = _LABELED_TOKEN.sub(r"\1\2[REDACTED]", text)
    text = _ENV_STYLE.sub(r"\1[REDACTED]", text)
    text = _MOLTBOOK_KEY.sub("[REDACTED:moltbook_key]", text)
    text = _TG_TOKEN.sub("[REDACTED:tg_token]", text)
    return text
