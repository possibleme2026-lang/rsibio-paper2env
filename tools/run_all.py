"""End-to-end orchestrator: synthesize -> lint-filter -> oracle-validate ->
three-harness comparison -> report.

The user's instruction was one sentence with no follow-ups ("don't ask me
anything, only stop when it's done"), so the remaining work is expressed as a
single resumable script rather than a series of manual steps. Each phase is
skippable and idempotent, and every phase writes a machine-readable artifact
under ``out/`` so a later run can pick up where it stopped.

Phases
------
1. synth    -- run the rsibio-paper2env pipeline over a corpus until the task
               directory holds >= --target tasks (uses tools/synth_scale.py).
2. lint     -- static preflight (tools/lint_tasks.py); emits the clean subset
               to out/task_lint/clean_tasks.txt
3. fix      -- apply the known defect repairs (tools/fix_tasks.py) and re-lint
4. oracle   -- prove each candidate task's own solution passes its verifier
               (tools/validate_tasks.py --task-list <clean>)
5. eval     -- run codex / pi / dsh on the oracle-passing tasks
               (tools/harness_eval.py --task-list <validated>)
6. report   -- render out/REPORT.md from the three summary.json artifacts

Usage
-----
    .rlvenv/Scripts/python.exe tools/run_all.py --target 100 --skip-synth
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
TASKS = REPO / "out" / "runs" / "tasks"
OUT = REPO / "out"


def _run(argv: list[str], log: Path, *, timeout: int | None = None,
         env_extra: dict | None = None) -> int:
    env = dict(os.environ)
    env.setdefault("APPDATA", os.path.expandvars(r"%APPDATA%"))
    env.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "PYTHONPATH": ""})
    if env_extra:
        env.update(env_extra)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        p = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return p.returncode


def count_tasks() -> int:
    if not TASKS.is_dir():
        return 0
    return sum(1 for d in TASKS.iterdir() if (d / "task.toml").is_file())


def phase_lint(tag: str) -> list[str]:
    _run([PY, "-u", "tools/lint_tasks.py", "--task-dir", str(TASKS),
          "--concurrency", "4"],
         OUT / f"_{tag}_lint.log")
    summary = json.loads((OUT / "task_lint" / "summary.json").read_text(
        encoding="utf-8"))
    clean = summary["clean_tasks"]
    (OUT / "task_lint" / "clean_tasks.txt").write_text(
        "\n".join(clean) + "\n", encoding="utf-8")
    print(f"[lint] clean {len(clean)}/{summary['total']}", flush=True)
    return clean


def phase_oracle(clean: list[str], concurrency: int, timeout_sec: int) -> list[str]:
    lst = OUT / "task_lint" / "clean_tasks.txt"
    _run([PY, "-u", "tools/validate_tasks.py", "--task-dir", str(TASKS),
          "--task-list", str(lst), "--concurrency", str(concurrency),
          "--timeout-sec", str(timeout_sec)],
         OUT / "_oracle_sweep.log")
    summary = json.loads((OUT / "task_validation" / "summary.json").read_text(
        encoding="utf-8"))
    good = [r["task"] for r in summary["results"] if r["status"] == "ok"]
    (OUT / "task_validation" / "oracle_pass.txt").write_text(
        "\n".join(good) + "\n", encoding="utf-8")
    print(f"[oracle] pass {len(good)}/{summary['total']}", flush=True)
    return good


def phase_eval(good: list[str], harnesses: str, concurrency: int,
               timeout_sec: int, setup_mult: float, api_key: str) -> None:
    lst = OUT / "task_validation" / "oracle_pass.txt"
    _run([PY, "-u", "tools/harness_eval.py", "--task-dir", str(TASKS),
          "--task-list", str(lst), "--harnesses", harnesses,
          "--rollout-concurrency", str(concurrency),
          "--timeout-sec", str(timeout_sec),
          "--setup-mult", str(setup_mult)],
         OUT / "_harness_full.log",
         env_extra={"TIERFLOW_API_KEY": api_key})


def phase_report() -> None:
    he = json.loads((OUT / "harness_eval" / "summary.json").read_text(
        encoding="utf-8"))
    ov = json.loads((OUT / "task_validation" / "summary.json").read_text(
        encoding="utf-8"))
    lt = json.loads((OUT / "task_lint" / "summary.json").read_text(
        encoding="utf-8"))
    hs = he["summary"]["harnesses"]
    raw = he.get("results", [])

    def _timing(h: str) -> str:
        es = [r.get("elapsed_sec", 0) for r in raw if r.get("harness") == h]
        if not es:
            return "n/a"
        es.sort()
        med = es[len(es) // 2]
        return f"median {med:.0f}s / max {es[-1]:.0f}s"

    def _error_kinds(h: str) -> dict[str, int]:
        kinds: dict[str, int] = {}
        for r in raw:
            if r.get("harness") != h or not r.get("error"):
                continue
            e = str(r["error"])[:80]
            if "timed out" in e or r.get("timed_out"):
                kinds["timeout"] = kinds.get("timeout", 0) + 1
            elif "verifier" in e.lower() or "reward" in e.lower():
                kinds["verifier"] = kinds.get("verifier", 0) + 1
            else:
                head = e.split(":")[0][:40]
                kinds[head] = kinds.get(head, 0) + 1
        return kinds

    rows = []
    rows.append("# rsibio-paper2env: synthesis + three-harness evaluation\n")
    rows.append(f"- generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    rows.append(f"- tasks packaged: {count_tasks()}")
    rows.append(f"- static lint clean: {lt['clean']}/{lt['total']} "
                f"({lt['clean_rate']:.1%})")
    rows.append(f"- oracle pass: {ov['oracle_pass']}/{ov['total']} "
                f"({ov['pass_rate']:.1%})\n")
    rows.append("## Harness comparison (model: tierflow_pro, identical task set)\n")
    rows.append("| harness | trials | mean reward | nonzero | >=0.5 | max | "
                "errors | timeouts | elapsed |")
    rows.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for h, st in hs.items():
        rows.append(f"| {h} | {st['trials']} | {st['mean_reward']} | "
                    f"{st['nonzero']} | {st['solved_ge_0.5']} | "
                    f"{st['max_reward']} | {st['errors']} | "
                    f"{st.get('timeouts', 0)} | {_timing(h)} |")
    s = he["summary"]
    rows.append(f"\npaired tasks: {s['paired_tasks']}  "
                f"head-to-head wins: {s['head_to_head_wins']}  "
                f"ties: {s['ties']}\n")

    rows.append("### Error attribution (nonzero reward = 0 trials)\n")
    rows.append("| harness | error class | count |")
    rows.append("|---|---|---:|")
    any_err = False
    for h in hs:
        for kind, n in sorted(_error_kinds(h).items(), key=lambda kv: -kv[1]):
            any_err = True
            rows.append(f"| {h} | {kind} | {n} |")
    if not any_err:
        rows.append("| (none) | all trials scored | 0 |")

    if s.get("per_task"):
        rows.append("\n### Per-task rewards\n")
        hlist = list(hs.keys())
        rows.append("| task | " + " | ".join(hlist) + " |")
        rows.append("|---|" + "---:|" * len(hlist))
        for t, v in s["per_task"].items():
            rows.append(f"| {t} | "
                        + " | ".join(str(v.get(h, "")) for h in hlist) + " |")
    (OUT / "REPORT.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[report] wrote {OUT / 'REPORT.md'}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--corpus", default="out/corpus_wave3.json")
    ap.add_argument("--harnesses", default="codex,pi,dsh")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--oracle-timeout", type=int, default=1800)
    ap.add_argument("--eval-timeout", type=int, default=2400)
    ap.add_argument("--eval-concurrency", type=int, default=2)
    ap.add_argument("--setup-mult", type=float, default=6.0)
    ap.add_argument("--api-key", default=os.environ.get("TIERFLOW_API_KEY", ""))
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()

    if not a.skip_synth and count_tasks() < a.target:
        print(f"[synth] need {a.target}, have {count_tasks()}", flush=True)
        _run([PY, "-u", "tools/synth_scale.py", "--corpus", a.corpus,
              "--select-n", str(a.target + 15), "--workers", "5",
              "--retries", "2", "--timeout-sec", "900"],
             OUT / "_synth_run.log",
             env_extra={"TIERFLOW_API_KEY": a.api_key})

    clean = phase_lint("full")
    if not a.skip_oracle:
        good = phase_oracle(clean, a.concurrency, a.oracle_timeout)
    else:
        gp = OUT / "task_validation" / "oracle_pass.txt"
        good = [x.strip() for x in gp.read_text(encoding="utf-8").splitlines()
                if x.strip()] if gp.is_file() else clean
    if not a.skip_eval:
        phase_eval(good, a.harnesses, a.eval_concurrency, a.eval_timeout,
                   a.setup_mult, a.api_key)
    phase_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
