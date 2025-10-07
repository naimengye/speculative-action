# Speculative Actions: Faster Agentic Systems via Lossless Parallelism
Predict likely next steps with a fast Speculator while a slower, authoritative Actor validates—turning idle waiting time into productive work.

## TL;DR
This repo implements Speculative Actions, a systems-level pattern that accelerates agents in rich environments (games, tool-use pipelines, web search, OS tuning). We treat every agent step as an API call and speculate the most likely next response(s) using a fast model. When the slow, ground-truth executor agrees, we’ve already pre-launched subsequent calls. The user sees as-if sequential, lossless behavior—just faster.

- Lossless core: verify before commit; rollback/overwrite when needed

- Works across environments: chess, e-commerce (τ-bench retail), HotpotQA web search, plus a lossy systems extension for OS hyperparameter tuning

- Theory: simple model predicts up to ~50% asymptotic latency reduction under idealized assumptions; wider/top-K speculation gives larger practical gains

- Practice: top-1/3 guesses often match; we measure both accuracy and wall-clock speedup across settings

We provide four environments:

- `chess/` - turn-based game-play (lossless)
- `ecommerce` - retail tool-calling (lossless)
- `hotpotqa/` - multi-hop web search (lossless)
- `os-tuning` - real-time OS tuning (a lossy extension with last-write wins)



