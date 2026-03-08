# Speculative Execution for HotPotQA

Part of [Speculative Actions](../README.md) (root repo). This sub-project applies speculative execution to **multi-hop question answering** on the [HotPotQA](https://hotpotqa.github.io/) benchmark. A lightweight speculator LLM races ahead of the Wikipedia API, predicting observations before they arrive, so the agent can pre-compute its next reasoning step.

> **Note:** All commands assume you are inside this directory (`hotpotqa/`). Paths in the codebase are relative to here.

## Background

In a standard ReAct loop the agent waits for each Wikipedia response before reasoning about the next step. With speculative execution, a fast secondary model predicts the API response while the call is still in flight. When the prediction matches, the agent's next thought is already prepared — saving one round-trip of latency per correct guess.

### Pipeline

1. The **agent** (GPT-4, Gemini, etc.) follows a Thought → Action → Observation loop, querying Wikipedia via Search and Lookup calls
2. In parallel, a cheaper **speculator** (GPT-3.5, GPT-5-nano, etc.) predicts what each Wikipedia call will return
3. Every step records both the real and speculated observations side-by-side
4. After the run, we evaluate how often the speculator's top-k predictions matched the agent's actual next action, and how much wall-clock time could have been saved

## Repository Layout

```
hotpotqa/
├── src/
│   ├── runner.py               # Experiment driver (HotPotQARun)
│   ├── llm_client.py           # Multi-provider LLM client (OpenRouter, Gemini, OpenAI)
│   ├── environment.py          # Wikipedia environment (WikiEnv)
│   ├── wrappers.py             # Gymnasium wrappers (HotPotQA, Logging, History)
│   ├── metrics.py              # Prediction accuracy computation
│   ├── grapher.py              # Bar charts and accuracy visualizations
│   ├── prompts.py              # Prompt templates
│   ├── utils.py                # I/O helpers
│   └── constants.py            # Hyperparameters, model names, API keys
├── run.py                      # CLI entrypoint
├── prompts/                    # Few-shot examples (JSON)
├── data/                       # HotPotQA dataset splits
├── run_metrics/                # Saved results: per-sample trajectories, aggregate metrics, plots
├── trajs/                      # Raw logged trajectories
└── requirements.txt
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API keys and models

Open `src/constants.py` and fill in your OpenRouter credentials:

```python
openrouter_api_key = "your-openrouter-key"
openrouter_model_name = "openai/gpt-4"            # Agent model
openrouter_guess_model_name = "openai/gpt-5-nano"  # Speculator model
```

Direct Gemini and OpenAI clients are also supported in `src/llm_client.py` — add the relevant API keys to `constants.py` to use them.

## Running Experiments

### Launch a run

```bash
python run.py
```

This evaluates the agent on HotPotQA questions with speculative simulation active. Trajectories, observations, and per-sample metrics land in `run_metrics/`.

You can swap models on the fly:

```bash
python run.py --modelname "openai/gpt-4" --guessmodelname "openai/gpt-3.5-turbo"
```

Pass `--norun` to skip execution and only post-process existing results. Pass `--noprint` for silent mode.

### Compute metrics

```bash
# Aggregate accuracy across every agent/speculator pair
python run.py --norun --getmetric --savemetrics

# Per-sample top-1 and top-3 accuracy lists
python run.py --norun --getmetric2 --savemetrics
```

### Plot results

```bash
# Wall-clock time comparison: normal API vs speculative
python run.py --norun --graph

# Top-1 vs top-3 prediction accuracy per speculator
python run.py --norun --graph3
```

## Hyperparameters

Adjustable in `src/constants.py`:

| Parameter | Default | Description |
|---|---|---|
| `n_samples_to_run` | 20 | Questions to evaluate per run |
| `n_steps_to_run` | 8 | Max reasoning steps per question |
| `guess_num_actions` | 3 | Top-k speculative candidates |
| `temperature` | 1 | Agent sampling temperature |
| `guess_temperature` | 0.1 | Speculator sampling temperature |
| `max_agent_retries` | 1 | Agent retries on malformed output |
| `max_guess_retries` | 3 | Speculator retries on malformed output |

## Sample Data

A curated set of trajectories and metrics is included under `run_metrics/` and `trajs/` to illustrate the output format across different agent/speculator model pairings.
