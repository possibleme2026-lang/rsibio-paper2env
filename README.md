# rsibio-paper2env

**Automated construction of life-science agent environments from published literature.**
A pipeline that turns open-access bioinformatics papers into [Harbor](https://pypi.org/project/harbor/)-format
task directories, and a four-stage protocol for checking whether those environments are
actually worth training on.

RisingSI · Apache-2.0

> **Status.** This release is the framework and its offline test suite. No environment has
> been constructed from a live paper and no model has been run yet: every empirical number
> in `claims.json` is `status=planned` on purpose. Structural claims (`C1`–`C8`) are measured
> and point at the test that measures them.

## Why

Reinforcement-learning environments for scientific agents are written by hand, one at a time,
and the bottleneck is not ideas — it is labour. Papers already contain the method, the data,
the metric and the evaluation protocol; the gap is turning prose into a runnable, *verifiable*
task. rsibio-paper2env closes that gap with two properties that matter more than throughput:

- **No fake green.** A task that does not pass verification is reported as failed, never shipped
  quietly. A missing Docker daemon or an absent `harbor` install yields `UNVERIFIED`, not a pass.
- **The evaluation contract is frozen before weights move.** The held-out split is locked with a
  corpus digest; a later corpus change fails verification instead of drifting to match the result.

## What it builds

A Harbor task directory per paper:

```text
tasks/<slug>/
  task.toml              # Harbor TaskConfig, schema_version = "1.4"
  instruction.md         # the task as the agent sees it
  environment/Dockerfile
  tests/test.sh          # runs the grader
  tests/grade.py         # metric computation
  solution/              # reference solution
```

plus a `manifest.jsonl` line per environment, which is what the Stage-1 tooling consumes.

## Install

```bash
python -m pip install -e .          # stdlib only: zero hard dependencies
python tests/run_checks.py          # 169 offline tests, no network, no Docker, no model
```

Python ≥ 3.12 is required (`tomllib`). Everything else is optional: `pip install -e .[harbor]`
enables the real-Harbor cross-check gate, `.[jsonschema]` enables richer schema diagnostics.

## Quickstart (offline)

```bash
# screen a corpus: deterministic rules first, model only for boundary papers
bioforge curate --corpus tests/fixtures/corpus/mini_corpus.json --select-n 3

# run the five stages over a corpus with a canned backend (no model, no network)
bioforge run --corpus corpus.json --runs-dir runs \
  --backend mock --mock-responses tests/mock_responses

# score a set of environments from a results manifest
bioforge validate-envs --out stage1.json --manifest results.json

# freeze the held-out set BEFORE training, then audit it for leakage
bioforge stage4 split-holdout --corpus corpus.json --lock holdout.lock.json
bioforge stage4 check-contamination --lock holdout.lock.json --texts-dir texts/
```

## The five agents

The pipeline mirrors the five roles in the paper: one deterministic screener and four
model-backed agents, each with its own prompt and its own failure mode.

| Agent | Stage | What it decides |
|---|---|---|
| Curator | `curate` | Which papers are screenable (licence, year, computational content) |
| Construction | `construct` | The `TaskBlueprint` for one paper |
| Reproduction | `reproduce` | A baseline score, re-derived through a second channel |
| Review | `review` | Fidelity, shortcut-resistance, and a dual-use denylist |
| Packager | `package` | Promotion into `tasks/<slug>`, re-verified at the last gate |

## Pipeline stages

`curate → construct → reproduce → review → package`, persisted per paper in `state.json`
after every stage. A crashed run resumes; a resumed run issues **zero** backend calls
(claim `C4`). `--force` re-runs, `--stages` restricts the pass.

## Verification: four gates

Every emitted task directory is checked before it can be packaged:

1. **Strict schema** — an embedded JSON Schema with `additionalProperties: false` at every
   level. This catches the Harbor trap where a mistyped `TaskConfig` key (`cpu` for `cpus`)
   is silently dropped by pydantic's `extra='ignore'` (claim `C1`).
2. **Layout sanity** — the files `harbor run` actually needs, in the right places.
3. **Real-Harbor cross-check** — optional; `UNVERIFIED` when `harbor` is not installed
   (claim `C2`).
4. **Hygiene** — LF line endings and no path escape out of the task directory.

## Four-stage validation protocol

| Stage | Question | Command |
|---|---|---|
| 1 | Are the environments real? | `bioforge validate-envs` |
| 2 | Do agents improve without weight updates? | `bioforge run-weightfree` |
| 3 | Is the spread across models or across harnesses? | `bioforge compare-harness` |
| 4 | Do held-out benchmarks move, and did they leak? | `bioforge stage4` |

Metrics live in `src/bioforge_harbor/validation/metrics.py` and mirror the paper's tables
one-to-one. Every metric returns `None` on empty input, never `0.0`, so an empty set can
never be read as a score of zero (claim `C5`).

## CLI

```bash
bioforge --version
bioforge curate           --corpus FILE [--select-n N] [--year-max Y] [--out FILE]
bioforge run              --corpus FILE [--backend mock|dryrun|pi] [--stages a,b] [--force]
bioforge validate-envs    --out FILE --manifest FILE [--executor manifest|docker|hybrid]
bioforge run-weightfree   --config FILE --out FILE [--dry-run | --replay FILE]
bioforge compare-harness  --baseline FILE --variant FILE [--out FILE]
bioforge stage4           split-holdout | verify-lock | check-contamination
```

## Repository layout

```text
src/bioforge_harbor/
  backends/      agent backends (mock replay, dry-run synthesize, pi headless)
  harbor_emit/   emitter + four-gate verifier + deterministic TOML writer
  models/        PaperRecord, TaskBlueprint, stage reports
  pipeline/      five stages, resumable state machine, runner
  retrieval/     Europe PMC client with offline cache
  schemas/       strict JSON Schema for task.toml and artifacts
  validation/    the four-stage protocol and its metrics
claims.json      every claim the paper is allowed to make
tools/           guards: claims, CRLF/BOM, README i18n, real-Harbor gate
```

## Known limitations

Read these before quoting any number:

- **Nothing empirical has been measured yet.** All `P*` claims are `planned`.
- `variance_decomposition` is only interpretable on a *complete* rectangular
  model × harness grid; missing cells add different constant offsets to the two
  components, which can flip "which axis dominates".
- `ngram_contamination` can miss a verbatim leak when the probe text is shorter
  than the n-gram window: a short probe's single gram can never match an 8-gram.
- `harness_delta` treats an absent key as `0.0`, so "did not participate" can read
  as "regressed".
- `stage1_summary` does not expose the review and probe denominators, so the
  "reviewed only" denominator of `expert_approval_rate` is not auditable from the
  report alone.

## Provenance and licensing

Paper metadata is retrieved from [Europe PMC](https://europepmc.org/RestfulWebService);
attribution and source terms are in `NOTICE`. Each constructed environment carries its
source paper's identifier and licence in `task.toml` provenance. Code is Apache-2.0
([LICENSE](https://www.apache.org/licenses/LICENSE-2.0)).

## Citation

```text
RisingSI. rsibio-paper2env: Automating the Construction of Life-Science Agent
Environments from Published Literature. 2026.
https://github.com/possibleme2026-lang/rsibio-paper2env
```
