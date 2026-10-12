"""Static preflight on packaged tasks: catch build-breakers without Docker.

Why this exists
---------------
The pipeline's ``reproduce`` stage never builds the task image -- it asks the
backend for baseline numbers and compares them. So a task can pass curate /
construct / reproduce / review / package and still be *unbuildable*. Two defect
classes were found only after running harbor's oracle agent (2026-10-12):

1. **Unresolvable apt package.** A generated Dockerfile did
   ``apt-get install -y --no-install-recommends vienna-rna``; Debian trixie
   (what ``python:3.11-slim`` resolves to now) has no such package::

       E: Package 'vienna-rna' has no installation candidate

2. **Generator scripts that cannot run.** ``gen_inputs.py`` mixed the two
   random APIs -- it created ``rng = random.Random(42)`` but called
   ``rng.normal(size=3)``, which only exists on ``numpy.random.Generator``::

       AttributeError: 'Random' object has no attribute 'normal'

Both are *static*: the first is a name lookup against the distro's package
index, the second is an attribute that provably does not exist on the object
being constructed. Checking them offline turns a ~90 s container build failure
into a millisecond lint, and turns "oracle scored 0" from an ambiguous result
into a precise, actionable one.

A third class only shows up when the grader runs: a ``reward.json`` that
carries a nested object, which harbor rejects with a pydantic
``ValidationError`` *after* the agent has finished -- see ``fix_tasks.py``.

What is checked
---------------
* every ``apt-get install`` package in ``environment/Dockerfile`` is resolved
  against the real Debian/Ubuntu package index for the declared base image
  (``docker run <image> apt-cache policy``), cached per image;
* every ``RUN`` line that invokes a ``.py`` file is executed **with its real
  arguments** (an earlier version dropped them, which turned every
  argparse-based generator into a false "missing --seed/--outdir" finding).
  Runs locally first; if the only failure is a missing third-party module it
  retries inside the task's own base image with the task's own pip packages,
  so a bare local interpreter (no rdkit/scipy/biopython) cannot masquerade as
  a task defect;
* ``python -m py_compile`` over every ``.py`` in the task (syntax gate).

Usage
-----
    .rlvenv/Scripts/python.exe tools/lint_tasks.py --task-dir out/runs/tasks
Output: out/task_lint/{summary.json, <task>.txt}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "out" / "task_lint"

DOCKER = shutil.which("docker") or "docker"

# Packages that exist in some Debian suites but not the one the base image
# currently resolves to. Recorded so the report explains *why* a name failed.
KNOWN_ABSENT = {"vienna-rna"}


def _apt_cache_policy(image: str, packages: list[str]) -> dict[str, bool]:
    """Return {package: available} by asking the image's own apt index."""
    if not packages:
        return {}
    script = (
        "apt-get update -qq >/dev/null 2>&1; "
        "for p in " + " ".join(packages) + "; do "
        '  c=$(apt-cache policy "$p" 2>/dev/null | '
        "    awk '/Candidate:/{print $2}'); "
        '  [ -n "$c" ] && [ "$c" != "(none)" ] && echo "$p OK" || echo "$p MISSING"; '
        "done"
    )
    env = dict(os.environ)
    env.update({"MSYS_NO_PATHCONV": "1", "MSYS2_ARG_CONV_EXCL": "*"})
    try:
        r = subprocess.run(
            [DOCKER, "run", "--rm", image, "bash", "-c", script],
            capture_output=True, text=True, timeout=600, env=env)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {p: True for p in packages}  # cannot check -> do not fail the task
    out = {}
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.endswith(" OK"):
            out[line[:-3]] = True
        elif line.endswith(" MISSING"):
            out[line[:-8]] = False
    for p in packages:
        out.setdefault(p, True)
    return out


def _logical_lines(text: str) -> list[str]:
    """Join backslash continuations, keep RUN bodies as single logical lines."""
    text = re.sub(r"\\\s*\n\s*", " ", text)
    return text.splitlines()


def _dockerfile_packages(text: str) -> list[str]:
    """apt packages, parsed per logical line.

    Must be line-scoped: the previous ``apt-get\\s+install\\s+[^&|]*`` form
    happily ran past the end of the RUN line (``[^&|]`` matches newlines), so
    a Dockerfile whose *last* apt line had no trailing ``&&`` had every later
    token swallowed as a "package" -- ``mkdir``, ``pip``, ``python``, even a
    seed value like ``20260730`` all showed up as unresolvable apt names.
    """
    pkgs: set[str] = set()
    for line in _logical_lines(text):
        if "apt-get install" not in line and "apt install" not in line:
            continue
        seg = re.split(r"&&|;", line)
        for part in seg:
            m = re.search(r"apt(?:-get)?\s+install\s+(.*)", part)
            if not m:
                continue
            for tok in m.group(1).split():
                if tok.startswith("-") or tok in ("&&", "\\"):
                    continue
                tok = tok.strip("'\"")
                if re.fullmatch(r"[a-z0-9][a-z0-9.+-]*", tok):
                    pkgs.add(tok)
    return sorted(pkgs)


def _base_image(text: str) -> str:
    m = re.search(r"^FROM\s+(\S+)", text, re.MULTILINE)
    return m.group(1) if m else "python:3.11-slim"


def _pip_install_cmds(text: str) -> list[str]:
    """Verbatim ``pip install ...`` commands from the Dockerfile.

    Replayed as-is during the in-image retry, so the linter's scratch run
    installs exactly what the real build installs. Extracting bare package
    names and re-guessing the flags was how ``pillow``/``scikit-image``
    silently went missing and produced bogus ``No module named 'PIL'``
    findings.
    """
    cmds: list[str] = []
    for line in _logical_lines(text):
        for part in re.split(r"&&|;", line):
            part = part.strip()
            m = re.search(r"(?:python3?\s+-m\s+)?pip3?\s+install\s+.*", part)
            if m:
                cmds.append(m.group(0).strip())
    return cmds


def _run_one_generator(task_dir: Path, workdir: Path, rel: str, argv: list[str],
                       image: str, pip_cmds: list[str]) -> str | None:
    """Return an error string, or None when the generator runs clean.

    Two-tier: run locally first (fast, no network). If the local failure is
    purely a missing third-party module, retry inside the task's own base
    image with the task's own pip packages -- otherwise a missing rdkit/scipy
    in the *linter's* interpreter would be misreported as a task defect.
    """
    cand = task_dir / "environment" / rel
    if not cand.is_file():
        cand = task_dir / "environment" / "data" / Path(rel).name
    if not cand.is_file():
        # generators sometimes COPY'd from the environment root to /tmp
        cand = task_dir / "environment" / Path(rel).name
    if not cand.is_file():
        return f"generator not found on disk: {rel}"

    local = workdir / Path(rel).name
    shutil.copyfile(cand, local)
    for sibling in (task_dir / "environment" / "data").iterdir():
        dst = workdir / sibling.name
        if sibling.is_file() and not dst.exists():
            shutil.copyfile(sibling, dst)
    # A generator that imports a sibling module needs it importable.
    for py in (task_dir / "environment" / "data").glob("*.py"):
        dst = workdir / py.name
        if not dst.exists():
            shutil.copyfile(py, dst)

    def _invoke(script_cwd: Path, argv: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=300, cwd=str(script_cwd))

    try:
        r = _invoke(workdir, [sys.executable, str(local), *argv])
    except subprocess.TimeoutExpired:
        return f"{rel}: timeout after 300s"
    if r.returncode == 0:
        return None

    combined = (r.stderr or "") + (r.stdout or "")
    if "ModuleNotFoundError" not in combined and "ImportError" not in combined:
        tail = combined.strip().splitlines()
        return f"{rel}: exit {r.returncode} :: " + " | ".join(tail[-3:])

    # ---- retry inside the declared base image with the task's pip packages.
    # Mirrors the real Dockerfile layout: data files under /app/data, the
    # generator invoked from wherever the RUN line says. A generator that
    # hardcodes /app/data must find that directory.
    env = dict(os.environ)
    env.update({"MSYS_NO_PATHCONV": "1", "MSYS2_ARG_CONV_EXCL": "*"})
    argv_line = " ".join(shlex.quote(x) for x in argv)
    installs = " && ".join(pip_cmds) if pip_cmds else \
        "python3 -m pip install -q --no-cache-dir numpy scipy pandas"
    shell = (
        "set -e; "
        "command -v python3 >/dev/null 2>&1 || "
        "(apt-get update -qq >/dev/null 2>&1; "
        "apt-get install -y -qq python3 python3-pip >/dev/null 2>&1) || true; "
        "command -v pip3 >/dev/null 2>&1 || "
        "(apt-get update -qq >/dev/null 2>&1; "
        "apt-get install -y -qq python3-pip >/dev/null 2>&1) || true; "
        "mkdir -p /app/data; cp -a /gen/. /app/data/ 2>/dev/null || true; "
        f"{installs} >/dev/null 2>&1 || true; "
        f"cd /app/data && python3 gen.py {argv_line} "
        f"|| cd /gen && python3 {Path(rel).name} {argv_line}"
    )
    (workdir / "gen.py").write_text(cand.read_text(encoding="utf-8",
                                                  errors="replace"),
                                    encoding="utf-8")
    try:
        d = subprocess.run(
            [DOCKER, "run", "--rm", "-v", f"{workdir.resolve()}:/gen", "-w", "/gen",
             image, "bash", "-c", shell],
            capture_output=True, text=True, timeout=900, env=env)
    except (subprocess.TimeoutExpired, OSError):
        return None  # cannot verify in-image -> do not claim a defect
    if d.returncode == 0:
        return None
    tail = ((d.stderr or "") + (d.stdout or "")).strip().splitlines()
    return f"{rel}: exit {d.returncode} (in {image}) :: " + " | ".join(tail[-3:])


def _run_generators(task_dir: Path, workdir: Path, image: str,
                    pip_cmds: list[str]) -> list[str]:
    """Execute the generators named in Dockerfile RUN lines, with their args.

    Argv is produced with ``shlex.split`` rather than ``str.split``: a
    generator may take a quoted JSON blob (``--config '{"a": 1}'``), and
    naive splitting shreds it into ``{, "a":, 1}`` and makes the generator
    die on ``JSONDecodeError``. Docker itself runs RUN through ``/bin/sh -c``,
    so the quoting is meaningful and must be preserved.
    """
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return []
    errors: list[str] = []
    text = dockerfile.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r"RUN\s+(.*?)(?=\nRUN|\nCOPY|\nADD|\nFROM|\Z)",
                         text, re.DOTALL):
        cmd = m.group(1).replace("\\\n", " ").strip()
        # one RUN may chain with && ; check each simple command
        for part in re.split(r"&&|;", cmd):
            try:
                toks = shlex.split(part.strip(), posix=True)
            except ValueError:
                continue
            for i, tok in enumerate(toks):
                if not re.fullmatch(r"python3?", Path(tok).name):
                    continue
                # collect the .py target + its args
                j = i + 1
                while j < len(toks) and toks[j].startswith("-"):
                    j += 1
                if j >= len(toks):
                    continue
                script = toks[j]
                if not script.endswith(".py"):
                    continue
                argv = toks[j + 1:]
                rel = script.replace("/app/", "", 1)
                err = _run_one_generator(task_dir, workdir, rel, argv, image,
                                         pip_cmds)
                if err:
                    errors.append(err)
    return errors


def lint_task(task_dir: Path, image_cache: dict[str, dict[str, bool]]) -> dict:
    findings: list[str] = []
    dockerfile = task_dir / "environment" / "Dockerfile"
    pkgs_missing: list[str] = []
    image = "python:3.11-slim"
    pip_cmds: list[str] = []

    if dockerfile.is_file():
        text = dockerfile.read_text(encoding="utf-8", errors="replace")
        pkgs = _dockerfile_packages(text)
        image = _base_image(text)
        pip_cmds = _pip_install_cmds(text)
        if image not in image_cache:
            image_cache[image] = _apt_cache_policy(image, pkgs)
        else:
            newp = [p for p in pkgs if p not in image_cache[image]]
            if newp:
                image_cache[image].update(_apt_cache_policy(image, newp))
        for p in pkgs:
            if not image_cache[image].get(p, True):
                pkgs_missing.append(p)
                findings.append(
                    f"apt package '{p}' has no installation candidate in {image}")

    # syntax gate over every python file in the task
    for py in task_dir.rglob("*.py"):
        if any(part in (".git", "__pycache__") for part in py.parts):
            continue
        try:
            r = subprocess.run([sys.executable, "-m", "py_compile", str(py)],
                               capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            findings.append(f"py_compile timeout: {py.relative_to(task_dir)}")
            continue
        if r.returncode != 0:
            findings.append(
                f"syntax error in {py.relative_to(task_dir)}: "
                + (r.stderr.strip().splitlines() or [""])[-1][:160])

    # generator execution (the class of bug that killed several tasks)
    with tempfile.TemporaryDirectory() as td:
        gen_errors = _run_generators(task_dir, Path(td), image, pip_cmds)
    for e in gen_errors:
        findings.append(f"generator failed: {e}")

    return {
        "task": task_dir.name,
        "ok": not findings,
        "findings": findings,
        "missing_apt_packages": pkgs_missing,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", default="out/runs/tasks")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--task-list", default="",
                    help="file with one task slug per line (subset to run)")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="parallel tasks (bounded by docker/pip I/O, not CPU)")
    ap.add_argument("--no-docker", action="store_true",
                    help="skip apt index checks (offline/fast mode)")
    a = ap.parse_args()

    root = Path(a.task_dir)
    tasks = sorted(d for d in root.iterdir() if (d / "task.toml").is_file())
    if a.task_list:
        wanted = {x.strip() for x in Path(a.task_list).read_text(
            encoding="utf-8").splitlines() if x.strip()}
        tasks = [t for t in tasks if t.name in wanted]
    if a.limit:
        tasks = tasks[:a.limit]
    OUT.mkdir(parents=True, exist_ok=True)

    if a.no_docker:
        global _apt_cache_policy
        _apt_cache_policy = lambda img, pkgs: {p: True for p in pkgs}  # noqa: E731

    image_cache: dict[str, dict[str, bool]] = {}
    results: list[dict] = []

    def _one(t: Path) -> dict:
        r = lint_task(t, image_cache)
        flag = "ok  " if r["ok"] else "FAIL"
        print(f"[lint] {flag} {r['task']}"
              + ("" if r["ok"] else f"  ({len(r['findings'])} findings)"), flush=True)
        if not r["ok"]:
            (OUT / f"{r['task']}.txt").write_text(
                "\n".join(r["findings"]), encoding="utf-8")
        return r

    if a.concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            results = list(ex.map(_one, tasks))
    else:
        results = [_one(t) for t in tasks]

    clean = [r for r in results if r["ok"]]
    summary = {
        "total": len(results),
        "clean": len(clean),
        "dirty": len(results) - len(clean),
        "clean_rate": round(len(clean) / len(results), 4) if results else 0.0,
        "clean_tasks": [r["task"] for r in clean],
        "findings_by_task": {r["task"]: r["findings"] for r in results if not r["ok"]},
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "clean_tasks.txt").write_text(
        "\n".join(summary["clean_tasks"]) + "\n", encoding="utf-8")
    print(f"\n=== static lint ===\nclean {len(clean)}/{len(results)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
