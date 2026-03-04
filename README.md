# Speculative Actions: Faster Agentic Systems via Lossless Parallelism

Predict likely next steps with a fast **Speculator** while a slower, authoritative **Actor** validates—turning idle waiting time into productive work.

## TL;DR
This repo implements **Speculative Actions**, a systems-level pattern that accelerates agents in rich environments (games, tool-use pipelines, web search, OS tuning). We treat every agent step as an API call and speculate the most likely next response(s) using a fast model. When the slow, ground-truth executor agrees, we’ve already pre-launched subsequent calls. The user sees as-if sequential, lossless behavior—just faster.

- Lossless core: verify before commit; rollback/overwrite when needed

- Works across environments: chess, e-commerce (τ-bench retail), HotpotQA web search, plus a lossy systems extension for OS hyperparameter tuning

- Theory: simple model predicts up to ~50% asymptotic latency reduction under idealized assumptions; wider/top-K speculation gives larger practical gains

- Practice: top-1/3 guesses often match; we measure both accuracy and wall-clock speedup across settings

## Repository layout

Each environment is **self-contained**. All setup, config, and commands for an environment are run from that environment's directory.

```
speculative-action/
├── README.md                 # This file
├── chess-game/               # Turn-based game-play (lossless)
│   ├── README.md             # Setup and usage for chess
│   ├── config.yml            # API keys, models, paths (relative to chess-game/)
│   ├── trajectories/         # Default data dir (sample_trajectories, test_trajectories, …)
│   └── …
├── ecommerce/                # Retail tool-calling (lossless)
├── hotpotqa/                 # Multi-hop web search (lossless)
└── os-tuning/                # OS hyperparameter tuning (lossy extension)
```

**Working directory convention:** For any environment, **run all commands from that environment's root** (e.g. `cd chess-game` then run scripts or entry points). Config and default data paths (e.g. `config.yml`, `trajectories/`) are relative to that root. This keeps imports and paths consistent without hardcoded or cwd-dependent paths.

## Environments

- **`chess-game/`** — turn-based game-play (lossless). See [chess-game/README.md](chess-game/README.md).
- **`ecommerce/`** — retail tool-calling (lossless)
- **`hotpotqa/`** — multi-hop web search (lossless)
- **`os-tuning/`** — real-time OS tuning (lossy extension with last-write wins)



