# datagen — frontier-generated training data for Boonie

Tooling for generating clean, first-person **identity / cmem-resume** training
turns in the qwen-toolcall format, using a frontier model (deepseek-v4-flash via
OpenRouter) instead of the dissociation-prone 14B. Motivation and findings live in
the `project-identity-dissociation` memory.

## The approach (why it's shaped this way)

Dissociation is partly *structural*: `agent.py:_build_messages` rebuilds history
from `agent_text` only — `<think>` is dropped after one turn (only the last 400
chars survive). So the first-person "I" thread evaporates, leaving a third-person-
looking transcript the model narrates as a spectator. Fix: put re-grounding
reasoning in the **persistent** channel (bare prose, not `<think>`).

The 14B can't self-repair its own dissociation, so we **generate** clean examples.
Two dead-ends ruled out by probe: a frontier model's **native tool mode** guts the
reasoning (terse, multi-call); **pasting Qwen3's `<tool_call>` as prompt text**
makes it fumble the grammar. The bridge that works:

> first-person **prose reasoning** + a trailing **JSON `{command, body, root}`**
> → we **transcode** to the Qwen3 `<tool_call>` envelope → **normalize** against
> the real harness command contract.

## Scripts

| file | what it does |
|---|---|
| `resume_gen.py` | the generator: 13 resume scenarios × temperature-varied samples → gated, transcoded turns. Importable library (SYSTEM, SCENARIOS, `gen_once`, parsers). Writes `data/resume_seed_v1.jsonl`. |
| `normalize_actions.py` | deterministic harness-contract fixer: slash-restore on known stems, body↔inline fold, `/mb post` title-extract, strips `root` from slash commands. Importable + standalone (`python3 normalize_actions.py <src> <dst>`). |
| `model_format_probe.py` | **the scenario/model benchmark.** Scores any OpenRouter model on our format+harness: dissociation-free, self-named, JSON-parse, contract-valid, reasoning depth. `python3 model_format_probe.py [model ...]`. |
| `mb_fetch.py` | read-only Moltbook puller (`home`/`feed`/`post`) to ground scenarios in real posts. Never writes to the account. |
| `data/resume_seed_v1.jsonl` | first clean seed: 52 cmem-resume turns, 52/52 dissociation-free, self-named, contract-valid. |

Keys are read from `.secrets` at runtime and never printed (OPENROUTER from
`frontier-boonie/.secrets`, MOLTBOOK from the repo `.secrets`). ZDR enforced via
account toggle + provider `data_collection:"deny"`.

## Still TODO

- Fold `normalize_actions` + a correct-syntax few-shot into `resume_gen` so raw
  output is contract-clean at the source (currently normalize is a post-pass).
- Rebuild content-engagement scenarios (comment-reply, post-compose, browsing,
  post-compaction) on **real** pulled posts — proven to prime deeper reasoning.
- Convert the seed turns to training-message format and merge into the new corpus.
