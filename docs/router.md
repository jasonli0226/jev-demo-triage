# Smart router: Jev vs LLM

A router reads a task and picks one of three OpenRouter models. The goal is the cheapest model that still answers correctly. Two routers answer the same `Choice` question with the same criteria:

- `jev`: Jev (`~typesafe/jev-latest`) via the decisions endpoint.
- `llm`: `z-ai/glm-4.7` at temperature 0, asked for JSON `{"choice", "probabilities"}`. An answer that does not parse is recorded as ERROR; there is no retry and no default tier.

## Model pool

| tier | model | input $/1M | output $/1M |
|---|---|---|---|
| cheap | `mistralai/mistral-small-3.2-24b-instruct` | 0.075 | 0.20 |
| mid | `z-ai/glm-4.7` | 0.40 | 1.75 |
| strong | `moonshotai/kimi-k3` | 3.00 | 15.00 |

Prices are OpenRouter's model list prices on 2026-09-24, hard-coded in `src/jev_router_bench/pool.py`; costs are estimates from token counts, and the provider that serves a request may charge less than the list price. All three models have zero-data-retention (ZDR) endpoints, so the pool works on an OpenRouter account that enforces ZDR. Every model call is capped at 8192 output tokens (`MAX_TOKENS`), because reasoning models spend part of the budget before they answer.

## Tasks and grading

30 single-turn tasks in `src/jev_router_bench/tasks.py`, about 10 aimed at each tier (the aim is a guess used only for reporting). Answers are checked in code: exact text, number with tolerance, unordered list, or hidden asserts for code tasks. Routers see only the task prompt.

**Security note:** code tasks run model-written code on your machine in a `python -I` subprocess with a temporary working directory, an environment without secrets and a 5 s timeout. Stdin is closed (`DEVNULL`), so code that calls `input()` fails fast instead of hanging. A pass requires both exit code 0 and a random nonce printed after the hidden asserts, so code that calls `sys.exit(0)` or `unittest.main()` before the asserts run can no longer look like a pass. On timeout the whole process group is killed. This is not a sandbox.

## Commands

0. `jev-playground router preflight` sends one short call to each tier and asks each router to route one task. It prints a table of the results and exits 1 if any call fails. `calibrate`, `route` and `e2e` run the same check first and stop before the real run if it fails; pass `--no-preflight` to skip it.
1. `jev-playground router calibrate --repeat 3` runs every task on every tier. A task's gold tier is the cheapest tier that passes 3/3; `none` if no tier does. It warns when more than 80% of tasks have gold `cheap` (the set is too easy) or more than 30% have `none`.
2. `jev-playground router route --router all --repeat 3` asks each router for each task and scores against gold: accuracy, 3x3 confusion, under-route (picked cheaper than gold: quality risk), over-route (picked dearer: wasted money), expected task cost from calibration, router cost and latency.
3. `jev-playground router e2e --router all` routes, runs the chosen model and grades it, next to `always-cheap`, `always-mid`, `always-strong` and `oracle` (gold tier; needs calibration). Reports pass rate, total cost, savings and quality kept versus `always-strong`, latency and tier mix.

`--task ID` (repeatable) limits any command to some tasks; `--calib PATH` picks a calibration file (default: newest `runs/calib-*.json`).

Each saved run keeps the model's answer text in `reply` (on calibration runs and on the `run` of e2e records) and the LLM router's raw answer in `decision.reply`. When the LLM router's answer does not parse, the error message includes the first 2000 characters of it. Calibration files saved before `reply` existed still load.

## Reading the numbers

- `route` accuracy counts router errors as wrong: the denominator is all routed rows, not just decided ones. `under` and `over` are computed over decided rows only, so `accuracy + under + over` need not sum to 100%.
- In `e2e`, an errored run costs $0 and does not count as passed, so always check the `errors` column before reading `savings` or `pass_rate` — a router with many errors can look artificially cheap.
- A calibration run that errors makes that task's gold tier unknown (excluded from routing accuracy), even if other tiers passed cleanly. A tier whose calibration runs all errored has an unknown cost, shown as `unpriced` in `route` and as `E<k>` in the `calibrate` table's tier cell (e.g. `0/3 E1`).
- `route` only routes tasks that are present in the calibration file with a known (non-`none`) gold tier; tasks with gold `none` or missing from calibration are skipped and reported on stderr, never silently sent to a router.

## Known limits

- GLM-4.7 is both the LLM router and the `mid` tier, so it may favour itself.
- Gold tiers come from 3 runs at temperature 0; providers are not fully deterministic.
- The routing criteria are ours; results show how well each router follows them, not an optimal policy.

## Results

None yet. Add runs here with the date, commit and the command used.
