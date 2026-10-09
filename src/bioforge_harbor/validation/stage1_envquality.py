"""Stage 1 runner: environment quality verification (paper 4.1).

Two execution modes, same metric code path:

  --executor manifest   read a JSON/JSONL file of per-env result records that
                        some *other* system (a Harbor run, a CI job, a human)
                        filled in. Fully offline; this is what tests use.

  --executor docker     build each task dir and run its tests/test.sh, then
                        parse /app/rewards.json. Requires Docker; if the
                        binary is missing the runner degrades to UNVERIFIED
                        records (built=false, ran=false with a note) instead
                        of pretending the environments are good.

The report is written as JSON so Stage-3/4 tooling and the paper's tables can
consume it directly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .metrics import stage1_summary
from ..util.subproc import run_to_file


def load_manifest(path: Path | str) -> list[dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("records", [])
    return data


def collect_env_records(envs_dir: Path | str) -> list[dict[str, Any]]:
    """Manifest of *what exists*: one record per task dir with task.toml."""
    out = []
    for d in sorted(Path(envs_dir).iterdir()):
        if d.is_dir() and (d / "task.toml").is_file():
            out.append({"env": d.name, "path": str(d)})
    return out


def docker_probe(record: dict[str, Any], *, timeout_sec: float = 1800.0) -> dict[str, Any]:
    """Build + run one environment. Returns a Stage-1 record.

    Only used with --executor docker. Never called by the offline tests.
    """
    env_dir = Path(record.get("path", ""))
    name = record.get("env", env_dir.name)
    rec: dict[str, Any] = {"env": name, "built": False, "ran": False}
    if not shutil.which("docker"):
        rec["note"] = "UNVERIFIED: docker not available"
        return rec
    image = f"bioforge-probe-{name}"
    build = run_to_file(["docker", "build", "-t", image, str(env_dir)],
                        timeout_sec=timeout_sec)
    if not build.ok:
        rec["note"] = f"build failed (exit={build.returncode})"
        return rec
    rec["built"] = True
    run = run_to_file(["docker", "run", "--rm", image, "bash", "/tests/test.sh"],
                      timeout_sec=timeout_sec)
    rec["ran"] = run.ok
    if not run.ok:
        rec["note"] = f"run failed (exit={run.returncode})"
    return rec


def run(envs_dir: Path | str, *, out: Path | str, executor: str = "manifest",
        manifest: Path | str | None = None,
        reviews: Path | str | None = None,
        probes: Path | str | None = None) -> dict[str, Any]:
    """Compute the Stage-1 report and write it to *out*."""
    if executor == "manifest":
        if not manifest:
            raise ValueError("--executor manifest requires --manifest FILE")
        env_records = load_manifest(manifest)
    elif executor == "docker":
        base = collect_env_records(envs_dir) if envs_dir else []
        env_records = [docker_probe(r) for r in base]
    elif executor == "hybrid":
        # env existence from disk, execution facts from a manifest filled in
        # by an external Harbor job (the recommended production mode).
        disk = {r["env"]: r for r in collect_env_records(envs_dir)}
        facts = {r.get("env"): r for r in (load_manifest(manifest) if manifest else [])}
        env_records = []
        for name in sorted(set(disk) | set(facts)):
            merged = dict(disk.get(name, {}))
            merged.update(facts.get(name, {}))
            merged.setdefault("env", name)
            env_records.append(merged)
    else:
        raise ValueError(f"unknown executor {executor!r}")

    review_records = load_manifest(reviews) if reviews else []
    probe_records = load_manifest(probes) if probes else []

    report = stage1_summary(env_records, review_records, probe_records)
    report["unverified"] = sum(1 for r in env_records
                               if str(r.get("note", "")).startswith("UNVERIFIED"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="bioforge validate-envs",
                                 description="Stage 1: environment quality verification (paper 4.1)")
    ap.add_argument("--envs-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--executor", choices=["manifest", "docker", "hybrid"], default="manifest")
    ap.add_argument("--manifest", default=None, help="per-env result records (JSON or JSONL)")
    ap.add_argument("--reviews", default=None, help="expert review records")
    ap.add_argument("--probes", default=None, help="shortcut probe records")
    a = ap.parse_args(argv)
    rep = run(a.envs_dir, out=a.out, executor=a.executor, manifest=a.manifest,
              reviews=a.reviews, probes=a.probes)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
