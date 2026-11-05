# Agent Eval Harness

Replays tool-calling agent tasks against a model, scores each run, and blocks regressions before a prompt change ships.

<img src="screenshots/1.png" alt="Dashboard showing a 94.2% pass rate, per-task scores and 11 regressions against the baseline" width="800">

## Why

Changing a prompt or swapping a model in a tool-calling assistant makes it different, not necessarily better, and reading transcripts by hand misses the regressions that matter. Teams that ship agents need one number they can compare across runs and a per-task list of what broke. This harness replays a fixed, versioned suite of tasks, checks the tool calls and the final answer, and fails a build when the score drops.

## How it works

A run starts from the API, which stores a queued row and hands the work to a background task. The runner loads the suite from `data/`, runs every task through the agent in parallel, scores each transcript with the deterministic checks and the judge, and writes the results back to PostgreSQL. A comparison endpoint lines two runs up task by task so a CI job can see exactly which tasks regressed.

```mermaid
flowchart LR
    ci["CI job"] --> api["FastAPI app"]
    api --> db[("PostgreSQL")]
    api --> runner["Runner"]
    runner --> suite["Task suite JSON"]
    runner --> agent["Agent loop"]
    agent --> model["Model endpoint"]
    agent --> tools["Tool stubs"]
    runner --> scoring["Checks and judge"]
    scoring --> judge["Judge model"]
    runner --> db
    db --> compare["Compare endpoint"]
    compare --> ci
```

The agent loop in `harness/agent.py` sends the task prompt with the tool schemas, executes each tool call against deterministic stand-ins so runs are reproducible, and stops when the model answers or `max_steps` is reached. `harness/scoring.py` treats the tool and text checks as gates: a run that calls the wrong tool or misses the required text fails regardless of the judge, and a run that passes both gets the mean of the check score and the judge score.

## Key decisions

- Deterministic checks gate the judge instead of being averaged with it, so a fluent but wrong answer cannot pass on the judge's opinion alone; the cost is that a correct answer phrased differently from `must_contain` fails.
- Tool results come from fixed stand-ins in the agent rather than live backends, which keeps runs reproducible and cheap; the cost is that the harness does not test the real tools.
- Tasks are stored in the database and upserted from the suite file on every run, so a suite edit is picked up without a migration; the cost is that old runs are compared against the current task text.
- The API queues a run and returns 202 with a background task, which keeps the request short; the cost is that a restart mid-run leaves the row marked running.

## Tech stack

- Python 3.12 with FastAPI and uvicorn
- SQLAlchemy 2 with PostgreSQL 16 through psycopg
- The OpenAI Python client against any OpenAI-compatible endpoint
- pytest and pytest-asyncio for tests, ruff for linting
- Docker and Kubernetes manifests for deployment

## Quick start

Prerequisites: Python 3.12 or newer, Docker with Compose, and an API key for an OpenAI-compatible endpoint.

```bash
docker compose up --build
```

```bash
pip install -e ".[dev]"
pytest
```

```bash
kubectl apply -f deploy/k8s.yaml
```

| Variable | What it sets | Default |
| --- | --- | --- |
| `DATABASE_URL` | SQLAlchemy URL of the PostgreSQL database | `postgresql+psycopg://harness:harness@localhost:5432/harness` |
| `OPENAI_API_KEY` | Key sent to the model endpoint | empty |
| `OPENAI_BASE_URL` | Base URL of an OpenAI-compatible endpoint | unset, provider default |
| `MODEL` | Model under test | `gpt-4o-mini` |
| `JUDGE_MODEL` | Model used as the judge | `gpt-4o-mini` |
| `PROMPT_VERSION` | Label stored on each run | `v1` |
| `PASS_THRESHOLD` | Score a task must reach to pass | `0.75` |
| `MAX_STEPS` | Tool-calling turns allowed per task | `6` |
| `REQUEST_TIMEOUT` | Seconds before a model call fails | `30.0` |
| `SUITE_PATH` | Task suite loaded by the runner | `data/suite.json` |

## Usage

Start a run and read it back:

```bash
curl -s -X POST localhost:8000/runs -H 'content-type: application/json' -d '{"suite": "suite", "prompt_version": "v2"}'
```

```json
{"id": 42, "suite": "suite", "model": "gpt-4o-mini", "prompt_version": "v2", "status": "queued", "started_at": "2025-03-11T09:14:02.118Z", "finished_at": null, "pass_rate": null}
```

Compare it against the previous baseline:

```bash
curl -s 'localhost:8000/compare?baseline=41&candidate=42'
```

```json
[{"task_id": "missing-order", "baseline_passed": true, "candidate_passed": false, "delta": -1}]
```

## Project structure

```text
harness/     API, agent loop, scoring, runner, models, config
data/        versioned task suite
deploy/      Kubernetes Deployment and Service
tests/       tests for the checks and score aggregation
ui/          single-page dashboard
```

## Limitations

- Tool calls run against fixed stand-ins, so the harness scores the model's decisions, not the real tools.
- The judge is a model call and can drift between provider versions; its reasons are stored but not re-checked.
- A run is executed in the API process, so a restart mid-run leaves the row marked running.
- The suite holds six sample tasks; a real suite needs many more to be meaningful.

## License

MIT. The project builds on the OpenAI Python client and FastAPI, both under their own licences.
