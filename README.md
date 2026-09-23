# jev-playground

Experiments with [Jev](https://typesafe.ai), TypeSafe's classifier, reached through OpenRouter's alpha decisions endpoint. Jev answers typed questions (yes/no, choice, score) about a state; it does not act. Each experiment compares Jev with a general LLM doing the same job.

| experiment | question | docs |
|---|---|---|
| `triage` | Does Jev help an incident-triage agent, as a tool or as a tool-risk gate? | [docs/triage.md](docs/triage.md) |
| `router` | Can Jev route tasks to the cheapest model that still answers correctly, compared with an LLM router? | [docs/router.md](docs/router.md) |

## Setup

```bash
uv sync
export OPENROUTER_API_KEY=...   # required for every live run
```

## Usage

```bash
uv run jev-playground triage --mode gate --scenario risky-bait   # same arguments as before
uv run jev-playground router calibrate --repeat 3               # find each task's gold tier
uv run jev-playground router route --router all --repeat 3      # routing accuracy, no task execution
uv run jev-playground router e2e --router all                   # route, answer, grade
```

`uv run jev-demo-triage ...` still works and is the same as `jev-playground triage ...`.

Run output (`runs/*.json`) is gitignored and stays local.

## Tests

```bash
uv run pytest               # offline, with coverage
uv run pytest -m live       # real OpenRouter calls, costs cents
```
