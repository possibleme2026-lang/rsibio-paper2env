"""One-shot repair pass over packaged tasks (2026-10-12 defect taxonomy).

Defect classes found by the oracle sweep + static lint, and what this tool
does about each. Everything is idempotent; a task already clean for a class
is left untouched. A backup of every modified file is kept in
``out/task_fixes/backup/<task>/<relpath>`` so the "before" state survives.

Class A -- generator aborts at image-build time (breaks the trial before
           the agent even starts):
  A1  allostery-transfer-entropy-alphabet
        ``rng = random.Random(42)`` but ``rng.normal(size=3)`` (numpy API on
        a stdlib object). Fix: swap to ``np.random.default_rng(42)`` -- it
        also provides .random()/.choice()/.randint(), so the rest of the
        file keeps working and the seed semantics are preserved.
  A2  rar1-sgt1-ancestral-reconstruction
        ``tree.write(f, format=1)`` on a hand-built ``BaseTree.Tree``.
        Bio.Phylo's writer lives on the module: ``Bio.Phylo.write(tree, f,
        'newick')``. Fix: rewrite the call site + import.
  A3  sca7-longitudinal-fc
        ``edge_names`` is built from i<j network pairs, so
        ``"Cerebellar-DefaultMode"`` can never exist (Cerebellar is last and
        is therefore never the *first* element). Fix: use
        "DefaultMode-Cerebellar".
  A4  t7-promoter-redesign
        ``CONSENSUS`` is 23 nt but ``positions`` is longer; ``CONSENSUS[i]``
        runs off the end. Fix: ``CONSENSUS[i % len(CONSENSUS)]``.
  A5  confdock-conformal-docking
        rdkit ``AtomValenceException`` on a hand-written SMILES at generator
        time. Fix: add bonds only while the atom is below its element's max
        degree. NOTE: the same ``generate_ligand`` body is duplicated in
        ``environment/data/gen_calibration.py`` AND ``tests/gold_gen.py`` --
        patching one left the other to crash the trial, so the fixer now
        rewrites every .py under the task containing the pattern.
  A6  3d-membrane-curvature-reconstruction
        ``trimesh.curvature.discrete_mean_curvature_measure(mesh, 3.0)`` --
        current trimesh wants ``(mesh, radius)`` positionally ambiguous with
        newer ``(mesh, radius, n_rings=1)``. Fix: pass ``radius=3.0`` by
        keyword.
  A7  data-driven-disease-dynamics
        ``np.array(all_samples)`` over per-sample vectors of unequal length;
        modern numpy raises ``ValueError: ... inhomogeneous shape`` instead of
        degrading to object dtype. Fix: nan-padded matrix + nanmean/nanstd.
  A8  boltz2esi-enzyme-substrate-ranking
        ``json.dump`` of a payload whose ``details.constraints`` values are
        ``numpy.bool_`` -> ``TypeError: Object of type bool is not JSON
        serializable``, raised *before* reward.json exists, so harbor reports
        ``RewardFileNotFoundError``. Fix: ``default=_jsonable`` converter.
  A9  solve.sh referencing mounted solution files via /app/solution
        harbor mounts ``solution/`` at ``/solution`` (``EnvironmentPaths``).
        Fix only when the referenced filename really exists in ``solution/``
        -- a solve.sh that *creates* ``/app/solution/x.py`` must be left alone
        (its grader execs that path by convention). allostery hits ``python
        /app/solution/solution.py`` for a file that is only ever mounted.

Class B -- unresolvable apt package:
  B1  biographx-rna-localization: ``vienna-rna`` is absent from Debian
      trixie *and* bookworm (verified live: Candidate: (none) on both --
      Debian dropped the package after stretch). The task needs the
      **RNAfold CLI**, which pip ``ViennaRNA`` does not ship. Fix: rebuild on
      ``ubuntu:22.04``, whose universe archive still carries ``vienna-rna``.

Class C -- reward.json carries non-numeric values.
  Harbor validates the file against ``dict[str, float | int]``
  (``harbor/models/verifier/result.py``). A nested object or string makes
  pydantic abort AFTER the agent has already finished, so the agent's whole
  rollout is discarded and the trial is recorded as an *exception*, not a
  score. Nine tasks did this -- the single biggest loss bucket.

  Fix (generic, not per-grader): append a **reward sanitizer** to every
  affected task's ``tests/test.sh``. It reads /logs/verifier/reward.json,
  keeps only numeric ``score``/``reward``, moves the rest into
  /logs/verifier/report.json, and re-writes a numeric-only reward.json. This
  is deliberately kept out of ``grade.py``: rewriting nine differently-shaped
  graders by string substitution is fragile, whereas one post-step that runs
  after every grader cannot corrupt the grading logic itself.
  ``test.sh`` also drops ``set -e`` so the sanitizer still runs when a grader
  raises (which is how granpa/cpplocpred lost their reward file).

Class E -- graders that break on their own data (found by oracle sweep):
  E   pandas DataFrame attribute shadowing. A column named like a DataFrame
      attribute (``sample``) is unreachable via dotted access: ``df.sample``
      returns the bound method, so ``df[df.sample == 'A']`` is an all-False
      mask and ``df[False]`` raises ``KeyError: False``. VERIFIED on
      auc-vs-spc-metaproteomics grade.py:112. Fix: bracket access, applied only
      to names *confirmed as columns* elsewhere in the same file (a bare
      ``df.size > 3`` is legitimate and must not be rewritten).

Class A (cont.) -- oracle-solution defects found by the second oracle sweep:
  A10 eola1-acc-salvage-lcms-sensitivity
        The instruction states calibration covers only 5 of 8 channels BY
        DESIGN, but the reference solve converts concentrations for every
        sample row, so ``fits[262.0834]`` raises ``KeyError`` for the three
        uncalibrated channels and the oracle dies before writing any output.
        Fix: restrict the conversion to channels that have a calibration fit.

Usage:
    .rlvenv/Scripts/python.exe tools/fix_tasks.py --task-dir out/runs/tasks
Output: out/task_fixes/{summary.json, applied.md}
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASKS_REL = Path("out") / "runs" / "tasks"
OUT = REPO / "out" / "task_fixes"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _backup(task: str, path: Path) -> None:
    path = path.resolve()
    base = (REPO / TASKS_REL).resolve()
    try:
        rel = path.relative_to(base)
    except ValueError:
        rel = Path(task) / path.name
    dst = OUT / "backup" / str(rel)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copyfile(path, dst)


def _rewrite(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        return False
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


# --------------------------------------------------------------------------
# Class A: generator crashes
# --------------------------------------------------------------------------

def fix_a1(task: str, root: Path, applied: list) -> None:
    g = root / task / "environment" / "data" / "gen_inputs.py"
    if not g.is_file():
        return
    _backup(task, g)
    if _rewrite(g, "    rng = random.Random(42)",
                "    rng = np.random.default_rng(42)"):
        applied.append(f"{task}: A1 random.Random -> np.random.default_rng "
                       "(so rng.normal() exists; seed preserved)")


def fix_a2(task: str, root: Path, applied: list) -> None:
    g = root / task / "environment" / "data" / "gen_inputs.py"
    if not g.is_file():
        return
    text = g.read_text(encoding="utf-8")
    if "tree.write(" not in text or "Bio.Phylo.write" in text:
        return
    _backup(task, g)
    new = text.replace("import random\nimport os\n",
                       "import random\nimport os\nfrom Bio import Phylo\n", 1)
    new = re.sub(r"tree\.write\(\s*([^,]+),\s*format=1\s*\)",
                 r"Bio.Phylo.write(tree, \1, 'newick')", new)
    g.write_text(new, encoding="utf-8")
    applied.append(f"{task}: A2 tree.write -> Bio.Phylo.write(tree, f, 'newick')")


def fix_a3(task: str, root: Path, applied: list) -> None:
    g = root / task / "environment" / "data" / "gen_inputs.py"
    if not g.is_file():
        return
    _backup(task, g)
    if _rewrite(g, 'edge_names.index("Cerebellar-DefaultMode")',
                'edge_names.index("DefaultMode-Cerebellar")'):
        applied.append(f"{task}: A3 edge_names index Cerebellar-DefaultMode -> "
                       "DefaultMode-Cerebellar (i<j ordering)")


def fix_a4(task: str, root: Path, applied: list) -> None:
    g = root / task / "environment" / "data" / "gen_inputs.py"
    if not g.is_file():
        return
    _backup(task, g)
    if _rewrite(g, "        c = CONSENSUS[i]",
                "        c = CONSENSUS[i % len(CONSENSUS)]"):
        applied.append(f"{task}: A4 CONSENSUS[i] -> CONSENSUS[i % len(CONSENSUS)]")


def fix_a5(task: str, root: Path, applied: list) -> None:
    """confdock: rdkit AtomValenceException from an unbounded random graph.

    ``generate_ligand`` builds a chain plus ``num_atoms // 6`` extra random
    single bonds with no regard for valence, so O/N/S routinely exceed their
    permitted degree and ``Chem.SanitizeMol`` raises. Fix: add bonds only
    while the atom is below its element's max degree.

    The SAME ``generate_ligand`` body is duplicated across the task's files
    (``environment/data/gen_calibration.py`` for image build and
    ``tests/gold_gen.py`` for grading), and patching only one leaves the other
    to crash the trial. So this rewrites *every* .py under the task that
    contains the pattern -- a missed copy is exactly how confdock failed
    oracle the first time.
    """
    old = (
        "    for _ in range(num_atoms // 6):\n"
        "        i, j = rng.sample(range(num_atoms), 2)\n"
        "        if not mol.GetBondBetweenAtoms(i, j):\n"
        "            mol.AddBond(i, j, Chem.BondType.SINGLE)\n"
    )
    new = (
        "    _maxdeg = {'C': 4, 'N': 3, 'O': 2, 'S': 2}\n"
        "    for _ in range(num_atoms // 6):\n"
        "        i, j = rng.sample(range(num_atoms), 2)\n"
        "        if mol.GetBondBetweenAtoms(i, j):\n"
        "            continue\n"
        "        ei = mol.GetAtomWithIdx(i).GetSymbol()\n"
        "        ej = mol.GetAtomWithIdx(j).GetSymbol()\n"
        "        if mol.GetAtomWithIdx(i).GetDegree() >= _maxdeg[ei]:\n"
        "            continue\n"
        "        if mol.GetAtomWithIdx(j).GetDegree() >= _maxdeg[ej]:\n"
        "            continue\n"
        "        mol.AddBond(i, j, Chem.BondType.SINGLE)\n"
    )
    for g in sorted((root / task).rglob("*.py")):
        if any(part in (".git", "__pycache__") for part in g.parts):
            continue
        text = g.read_text(encoding="utf-8", errors="replace")
        if old not in text:
            continue
        _backup(task, g)
        g.write_text(text.replace(old, new), encoding="utf-8")
        applied.append(f"{task}: A5 {g.relative_to(root / task)} random-bond "
                       "loop now respects per-element max degree")


# --------------------------------------------------------------------------
# Class B: apt package absent from base suite
# --------------------------------------------------------------------------

def fix_b1(task: str, root: Path, applied: list) -> None:
    d = root / task / "environment" / "Dockerfile"
    if not d.is_file():
        return
    text = d.read_text(encoding="utf-8")
    if "vienna-rna" not in text or "ubuntu:22.04" in text:
        return
    _backup(task, d)
    d.write_text(
        "FROM ubuntu:22.04\n"
        "WORKDIR /app\n"
        "COPY data/ /app/data/\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
        "        vienna-rna python3 python3-pip \\\n"
        "    && rm -rf /var/lib/apt/lists/*\n"
        "RUN pip3 install --no-cache-dir numpy\n"
        "RUN python3 /app/data/gen_inputs.py && chmod -R 444 /app/data\n"
        "RUN mkdir -p /app/output && chmod 777 /app/output\n",
        encoding="utf-8")
    applied.append(f"{task}: B1 base -> ubuntu:22.04 (vienna-rna lives in "
                   "universe; Debian trixie/bookworm have no candidate)")


# --------------------------------------------------------------------------
# Class A (cont.): grader-time defects found by the first oracle sweep
# --------------------------------------------------------------------------

def fix_a6(task: str, root: Path, applied: list) -> None:
    """3d-membrane: trimesh curvature call is missing the ``points`` argument.

    VERIFIED in-container (trimesh 5.1.1): the signature is
    ``discrete_mean_curvature_measure(mesh, points, radius)`` -- three required
    positionals. The task calls ``(...)(mesh, 3.0)``, so ``3.0`` lands in
    ``points`` and ``radius`` is missing -> ``TypeError: ... missing 1 required
    positional argument: 'radius'``. The intent is per-vertex curvature, so the
    correct call is ``(mesh, mesh.vertices, 3.0)``.
    """
    for g in sorted((root / task).rglob("*.py")):
        if any(part in (".git", "__pycache__") for part in g.parts):
            continue
        text = g.read_text(encoding="utf-8", errors="replace")
        changed = False
        for fn in ("discrete_mean_curvature_measure",
                   "discrete_gaussian_curvature_measure"):
            old = f"trimesh.curvature.{fn}(mesh, 3.0)"
            new = f"trimesh.curvature.{fn}(mesh, mesh.vertices, 3.0)"
            if old in text:
                text = text.replace(old, new)
                changed = True
        if changed:
            _backup(task, g)
            g.write_text(text, encoding="utf-8")
            applied.append(f"{task}: A6 {g.relative_to(root / task)} curvature "
                           "call passes points=mesh.vertices (signature is "
                           "(mesh, points, radius))")
    # trimesh.curvature also imports rtree at call time; without it the call
    # raises ModuleNotFoundError: No module named 'rtree'. VERIFIED in a
    # python:3.11-slim container.
    d = root / task / "environment" / "Dockerfile"
    if d.is_file():
        dt = d.read_text(encoding="utf-8")
        if "trimesh" in dt and "rtree" not in dt:
            _backup(task, d)
            d.write_text(dt.replace("trimesh", "trimesh rtree", 1),
                         encoding="utf-8")
            applied.append(f"{task}: A6 Dockerfile installs rtree "
                           "(trimesh.curvature imports it at call time)")


def fix_a7(task: str, root: Path, applied: list) -> None:
    """data-driven-disease-dynamics: np.array of ragged per-sample arrays.

    ``np.array(all_samples)`` -- a list of per-(patient,biomarker) time-series
    vectors of *different* lengths -- raises "inhomogeneous shape" on modern
    numpy, which no longer silently degrades to object dtype.

    The critical constraint: **the grader and the oracle solution must stay
    byte-identical in this block**. The oracle scores 1.0 only by reproducing
    the grader's own gold, so a fix applied to one file and not the other
    turns a crashing task into an unsolvable one. This fixer therefore
    normalises BOTH files to one canonical block, replacing whichever variant
    is present (the original, or an earlier partial fix).

    Note the broadcast: ``mean`` is a length-``_ncol`` vector reduced over
    samples, so ``mean[i]`` (``i`` = biomarker index) is a *scalar* -- the
    subtraction must not be sliced.
    """
    variants = [
        # original, crashing
        ("all_samples = np.array(all_samples)\n"
         "mean = all_samples.mean(axis=0)\n"
         "std = all_samples.std(axis=0)\n", ""),
        ("    all_samples = np.array(all_samples)\n"
         "    mean = all_samples.mean(axis=0)\n"
         "    std = all_samples.std(axis=0)\n", "    "),
        # earlier partial fix (padded, no atleast_1d)
        ("    # per-column stats; np.array() would raise on ragged lengths\n"
         "    _arrs = [np.asarray(a, dtype=float) for a in all_samples]\n"
         "    _ncol = max(a.shape[0] for a in _arrs)\n"
         "    _mat = np.full((len(_arrs), _ncol), np.nan)\n"
         "    for _i, _a in enumerate(_arrs):\n"
         "        _mat[_i, :_a.shape[0]] = _a\n"
         "    mean = np.nanmean(_mat, axis=0)\n"
         "    std = np.nanstd(_mat, axis=0)\n", "    "),
    ]
    canonical = (
        "{I}# per-column stats; np.array() would raise on ragged lengths\n"
        "{I}_arrs = [np.atleast_1d(np.asarray(a, dtype=float))\n"
        "{I}         for a in all_samples]\n"
        "{I}_ncol = max(a.shape[0] for a in _arrs)\n"
        "{I}_mat = np.full((len(_arrs), _ncol), np.nan)\n"
        "{I}for _i, _a in enumerate(_arrs):\n"
        "{I}    _mat[_i, :_a.shape[0]] = _a\n"
        "{I}mean = np.nanmean(_mat, axis=0)\n"
        "{I}std = np.nanstd(_mat, axis=0)\n")

    for sub in ("solution/solve.sh", "tests/grade.py", "tests/gold_gen.py",
                "solution/solve.py"):
        p = root / task / sub
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        # undo the bad slice if a previous run of this fixer wrote it
        # (regex so it works at any indent; mean[i] is a scalar per the
        # broadcast analysis -- slicing it is the regression, not the fix)
        text = re.sub(
            r"\(d\[b\] - mean\[i\]\[?:?d\[b\]\.shape\[0\]\]?\)? ?"
            r"/ ?std\[i\]\[?:?d\[b\]\.shape\[0\]\]?\)?",
            "(d[b] - mean[i]) / std[i]", text)
        text = text.replace(
            "(d[b] - mean[i][:d[b].shape[0]]) / std[i][:d[b].shape[0]]",
            "(d[b] - mean[i]) / std[i]")
        hit = None
        for old, indent in variants:
            if old in text:
                hit = (old, indent)
                break
        if hit is None:
            if "np.nanmean(_mat, axis=0)" in text:
                continue  # already canonical
            continue
        old, indent = hit
        _backup(task, p)
        text = text.replace(old, canonical.format(I=indent), 1)
        p.write_text(text, encoding="utf-8")
        applied.append(f"{task}: A7 canonicalised ragged np.array fix in {sub} "
                       f"(grader and oracle kept byte-identical)")


def fix_a8(task: str, root: Path, applied: list) -> None:
    """boltz2esi: numpy bool not JSON serializable in the reward payload.

    ``check_*`` helpers return ``numpy.bool_``; the grader then does
    ``json.dump({'details': {'constraints': {name: ok ...}}})`` which raises
    ``TypeError: Object of type bool is not JSON serializable`` -- BEFORE
    reward.json exists, so the trial dies with RewardFileNotFoundError. Fix:
    give json.dump a default= converter so any numpy scalar/bool is coerced.
    """
    g = root / task / "tests" / "grade.py"
    if not g.is_file():
        return
    text = g.read_text(encoding="utf-8")
    changed = False
    if "def _jsonable(" not in text:
        for old in ('json.dump(result, f, indent=2)',):
            if old in text:
                helper = (
                    "def _jsonable(o):\n"
                    "    try:\n"
                    "        import numpy as _np\n"
                    "        if isinstance(o, (_np.bool_, _np.integer, _np.floating)):\n"
                    "            return o.item()\n"
                    "        if isinstance(o, _np.ndarray):\n"
                    "            return o.tolist()\n"
                    "    except Exception:\n"
                    "        pass\n"
                    "    return str(o)\n\n\n")
                # insert helper before the first def
                idx = text.index("def ")
                text = text[:idx] + helper + text[idx:]
                text = text.replace(old, "json.dump(result, f, indent=2, default=_jsonable)")
                changed = True
                break
    # ALWAYS also fix the stdout echo: ``print(json.dumps(result, ...))``
    # without default= crashes AFTER reward.json was written, and the
    # container exit code then marks the verifier run failed even though a
    # valid reward exists. VERIFIED on boltz2esi: reward.json was never read
    # because test.sh died on this print.
    out_lines = []
    for line in text.splitlines():
        if ("print(json.dumps(" in line or "print(json.dumps" in line) \
                and "default=" not in line:
            m = re.match(r"^(\s*print\(json\.dumps\((.*)\)\)\s*)$", line)
            if m:
                line = f"{m.group(1)[:-2]}, default=_jsonable))"
                changed = True
        out_lines.append(line)
    text = "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")
    if changed:
        _backup(task, g)
        g.write_text(text, encoding="utf-8")
        applied.append(f"{task}: A8 json.dump/dumps(default=_jsonable) coerces "
                       "numpy scalars/bools (incl. the stdout echo)")


def fix_a9(task: str, root: Path, applied: list) -> None:
    """solve.sh references /app/solution for a file harbor actually mounts.

    ``harbor/models/trial/paths.py`` sets
    ``EnvironmentPaths.solution_dir = PurePosixPath("/solution")`` and the
    oracle agent uploads the task's ``solution/`` there. A solve.sh that runs
    ``python solution/solution.py`` relative to ``/app`` therefore fails with
    ENOENT.

    Critical precision note: most tasks' solve.sh *writes its own* helper to
    ``/app/solution/x.py`` and runs that path -- which is correct and must NOT
    be rewritten (their graders even exec ``/app/solution/x.py`` by
    convention). So only rewrite a reference when the referenced filename
    actually exists in the task's ``solution/`` dir, i.e. it is genuinely
    referring to the mounted content rather than a file the script creates.
    """
    sol_dir = root / task / "solution"
    sh = sol_dir / "solve.sh"
    if not sh.is_file():
        return
    mounted = {p.name for p in sol_dir.iterdir()
               if p.is_file() and p.name != "solve.sh"}
    if not mounted:
        return
    text = sh.read_text(encoding="utf-8")
    new = text
    for name in sorted(mounted):
        # `python solution/<name>` / `python3 solution/<name>` -> /solution
        for py in ("python", "python3"):
            new = new.replace(f"{py} solution/{name}", f"{py} /solution/{name}")
        # explicit /app/solution/<name> -> /solution/<name>
        new = new.replace(f"/app/solution/{name}", f"/solution/{name}")
        # bare /app/<name> for a file that only exists in solution/ -> /solution
        # (e.g. `python /app/solve.py` when solution/solve.py is the mounted file)
        new = new.replace(f"/app/{name}", f"/solution/{name}")
    if new == text:
        return
    _backup(task, sh)
    sh.write_text(new, encoding="utf-8")
    applied.append(f"{task}: A9 solve.sh references mounted solution files via "
                   "/solution (harbor's actual mount point), not /app")


def fix_a10(task: str, root: Path, applied: list) -> None:
    """eola1: reference solve converts concentrations for uncalibrated channels.

    The task's own instruction says the calibration file covers "5 of the 8
    channels", so three sample channels legitimately have no fit. The solve
    script still looks up ``fits[r['mz4']]`` for every row, raising
    ``KeyError`` on the first uncalibrated channel and aborting before any
    output file is written -- so the oracle scored 0 on a solvable task.
    Fix: drop rows whose channel has no calibration fit.
    """
    sh = root / task / "solution" / "solve.sh"
    if not sh.is_file():
        return
    text = sh.read_text(encoding="utf-8")
    old = ("sm = sm.dropna(subset=['analyte'])\n"
           "sm['conc_nM'] = [(r['area'] - fits[r['mz4']][1]) / fits[r['mz4']][0]"
           " for _, r in sm.iterrows()]\n")
    if old not in text:
        return
    new = ("sm = sm.dropna(subset=['analyte'])\n"
           "# only channels covered by the external calibration can be\n"
           "# quantified (the file calibrates 5 of the 8 channels by design)\n"
           "sm = sm[sm['mz4'].isin(fits.keys())]\n"
           "sm['conc_nM'] = [(r['area'] - fits[r['mz4']][1]) / fits[r['mz4']][0]"
           " for _, r in sm.iterrows()]\n")
    _backup(task, sh)
    sh.write_text(text.replace(old, new, 1), encoding="utf-8")
    applied.append(f"{task}: A10 reference solve now skips channels without a "
                   "calibration fit (was KeyError on uncalibrated m/z)")


def fix_a11(task: str, root: Path, applied: list) -> None:
    """hic-depth: grader's key-normaliser assumes every key is numeric.

    ``keyed()`` probes ``(str(key), str(float(key)), key)`` to tolerate a value
    stored under "5" vs 5 vs 5.0. But it is also called with *metric names*
    ("loop_density"), where ``float(key)`` raises
    ``ValueError: could not convert string to float`` and the whole grader
    aborts -- so a correct solution scored 0. Fix: only attempt the float form
    when the key actually parses as a number.
    """
    g = root / task / "tests" / "grade.py"
    if not g.is_file():
        return
    text = g.read_text(encoding="utf-8")
    old = ("    for k in (str(key), str(float(key)), key):\n"
           "        if k in d:\n"
           "            return d[k]\n"
           "    return None\n")
    if old not in text:
        return
    new = ("    candidates = [str(key), key]\n"
           "    try:\n"
           "        candidates.insert(1, str(float(key)))\n"
           "    except (TypeError, ValueError):\n"
           "        pass  # key is a metric name, not a number\n"
           "    for k in candidates:\n"
           "        if k in d:\n"
           "            return d[k]\n"
           "    return None\n")
    _backup(task, g)
    g.write_text(text.replace(old, new, 1), encoding="utf-8")
    applied.append(f"{task}: A11 keyed() tolerates non-numeric metric-name "
                   "keys (was ValueError in str(float(key)))")


# --------------------------------------------------------------------------
# Class C: reward.json must be numeric-only  (generic sanitizer)
# --------------------------------------------------------------------------

SANITIZER = '''#!/usr/bin/env python3
"""Force /logs/verifier/reward.json to be numeric-only.

Harbor type-checks the reward file as ``dict[str, float | int]``; a nested
object or string anywhere in it makes pydantic abort the trial *after* the
agent has finished, so the rollout is recorded as an exception rather than a
score. This step runs after the grader and moves anything non-numeric into
/logs/verifier/report.json, leaving a valid reward.json behind.

Also covers the "grader died before writing anything" case: it synthesises
{"score": 0.0, "reward": 0.0} so the trial reports a real zero instead of a
RewardFileNotFoundError exception.
"""
import json
import os

REWARD = "/logs/verifier/reward.json"
REPORT = "/logs/verifier/report.json"


def _numeric(v):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def main():
    os.makedirs("/logs/verifier", exist_ok=True)
    payload = {}
    if os.path.exists(REWARD):
        try:
            payload = json.load(open(REWARD))
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        payload = {"score": payload, "reward": payload}

    score = _numeric(payload.get("score"))
    reward = _numeric(payload.get("reward"))
    if reward is None:
        reward = score
    if score is None:
        score = reward
    if score is None:
        score = reward = 0.0

    # keep every scalar entry (no information loss for numeric metrics) and
    # move only the non-numeric ones -- those are what break pydantic.
    clean = {"score": score, "reward": reward}
    extra = {}
    for k, v in payload.items():
        if k in ("score", "reward"):
            continue
        num = _numeric(v)
        if num is None or isinstance(v, (dict, list)):
            extra[k] = v
        else:
            clean[k] = num
    if extra:
        prev = {}
        if os.path.exists(REPORT):
            try:
                prev = json.load(open(REPORT))
            except Exception:
                prev = {}
        if isinstance(prev, dict):
            prev.update(extra)
        else:
            prev = extra
        with open(REPORT, "w") as fh:
            json.dump(prev, fh, indent=2, default=str)

    with open(REWARD, "w") as fh:
        json.dump(clean, fh)
    print("sanitized reward.json ->", clean)


if __name__ == "__main__":
    main()
'''


def fix_class_c(task: str, root: Path, applied: list) -> None:
    tdir = root / task
    tests = tdir / "tests"
    if not tests.is_dir():
        return
    reward_writers = [p for p in tests.glob("*.py")
                      if "reward.json" in p.read_text(encoding="utf-8",
                                                     errors="replace")]
    if not reward_writers:
        return

    # is any non-numeric value actually emitted? cheap textual sniff
    suspect = False
    for p in reward_writers:
        src = p.read_text(encoding="utf-8", errors="replace")
        if re.search(r"(details|checks|constraints|metrics|messages|passed|"
                     r"total|report|error)\s*[:=]", src):
            suspect = True
            break
    sanitizer = tests / "_fix_reward.py"
    if not sanitizer.is_file():
        sanitizer.write_text(SANITIZER, encoding="utf-8")

    sh = tests / "test.sh"
    if not sh.is_file():
        return
    body = sh.read_text(encoding="utf-8")
    if "_fix_reward.py" in body:
        return
    _backup(task, sh)
    body = body.replace("set -euo pipefail", "set -uo pipefail")
    body = body.replace("set -e\n", "set -u\n")
    if not body.endswith("\n"):
        body += "\n"
    body += "python3 /tests/_fix_reward.py 2>&1 || python /tests/_fix_reward.py 2>&1 || true\n"
    sh.write_text(body, encoding="utf-8")
    applied.append(f"{task}: C reward sanitizer appended to tests/test.sh "
                   f"(suspect_payload={suspect})")


# --------------------------------------------------------------------------
# Class D: syntax error
# --------------------------------------------------------------------------

def fix_d1(task: str, root: Path, applied: list) -> None:
    g = root / task / "tests" / "grade.py"
    if not g.is_file():
        return
    text = g.read_text(encoding="utf-8")
    m = re.search(
        r"\n?[ \t]*required = \[\('n_lof',\), \('n_wt',\),\n"
        r"[ \t]*\('immune_markers', g, 'delta'\) for g in \[\]\]\n", text)
    if not m:
        return
    _backup(task, g)
    g.write_text(text.replace(m.group(0), "\n", 1), encoding="utf-8")
    applied.append(f"{task}: D removed invalid statement "
                   "`required = [('n_lof',), ('n_wt',), (...) for g in []]`")


# --------------------------------------------------------------------------
# Class E: pandas DataFrame attribute shadowing in graders
# --------------------------------------------------------------------------

# A DataFrame column whose name collides with a DataFrame *attribute* (a
# method or property) is unreachable via dotted access: ``df.sample`` returns
# the bound ``DataFrame.sample`` method, so ``df[df.sample == 'A']`` compares a
# method object to a string -> an all-False boolean mask -> ``df[False]`` ->
# ``KeyError: False``. VERIFIED on auc-vs-spc-metaproteomics: grade.py:112
# ``sub = df[df.sample == sample]`` raised exactly that, the grader died before
# writing reward.json, and harbor recorded ``RewardFileNotFoundError`` -- the
# task looked unsolvable when only the grader was broken.
#
# The fix is mechanical and semantics-preserving: dotted column access ->
# bracket access (``df["sample"] == sample``). Only names that are real pandas
# attributes are touched, so ordinary columns are never rewritten.
_PANDAS_DF_ATTRS = (
    "sample", "size", "shape", "index", "columns", "values", "name", "dtypes",
    "ndim", "empty", "T", "axes", "flags", "at", "iat", "loc", "iloc",
    "count", "sum", "mean", "std", "min", "max", "abs", "all", "any",
    "mode", "rank", "mask", "where", "query", "filter", "sort", "drop",
    "head", "tail", "copy", "pop", "items", "keys", "map", "apply", "diff",
    "shift", "clip", "round", "add", "insert", "update", "dot", "prod",
    "describe", "groupby", "merge", "join", "rename", "reset_index",
)


def _dataframe_attr_shadow_re(df_var: str, col: str) -> re.Pattern:
    # df.sample ==  /  df.sample  <  /  df.sample.isin(  ; never match on the
    # LHS of an assignment and never match df.sample( (a real method call).
    return re.compile(rf"\b{re.escape(df_var)}\.{re.escape(col)}\b(?!\s*\()")


def _strip_pandas_comments(text: str) -> list[tuple[int, str]]:
    """Return (lineno0, code_before_comment) for each source line."""
    out = []
    for i, line in enumerate(text.splitlines()):
        # good enough for graders: strip a trailing comment not inside quotes
        code = line
        in_s = in_d = False
        for j, ch in enumerate(line):
            if ch == "'" and not in_d:
                in_s = not in_s
            elif ch == '"' and not in_s:
                in_d = not in_d
            elif ch == "#" and not in_s and not in_d:
                code = line[:j]
                break
        out.append((i, code))
    return out


def _confirmed_columns(text: str) -> set[str]:
    """Column names the file itself proves are columns.

    A bare ``df.size > 3`` is *legitimate* (DataFrame.size) and must NOT be
    rewritten, whereas ``df.sample == 'A'`` is broken. The signal that tells
    them apart is elsewhere in the same file: a real column shows up as a
    quoted string inside bracket access / ``groupby`` / ``rename`` / ``merge``
    / ``columns=``. Collect those, and only rewrite shadowed attributes that
    appear in this set.
    """
    cols: set[str] = set()
    patterns = [
        r"\[\s*\[?\s*['\"]([A-Za-z_]\w*)['\"]",          # df[['a']] / df['a']
        r"groupby\(\s*\[?\s*['\"]([A-Za-z_]\w*)['\"]",
        r"rename\([^)]*['\"]([A-Za-z_]\w*)['\"]",
        r"merge\([^)]*['\"]([A-Za-z_]\w*)['\"]",
        r"columns\s*=\s*\[[^\]]*['\"]([A-Za-z_]\w*)['\"]",
        r"sort_values\([^)]*['\"]([A-Za-z_]\w*)['\"]",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            cols.add(m.group(1))
    # merge(on='x') / merge(on=['x','y'])
    for m in re.finditer(r"on\s*=\s*(\[[^\]]*\]|['\"][A-Za-z_]\w*['\"])", text):
        for q in re.finditer(r"['\"]([A-Za-z_]\w*)['\"]", m.group(1)):
            cols.add(q.group(1))
    return cols


def fix_class_e(task: str, root: Path, applied: list) -> None:
    """Rewrite dotted access to attribute-shadowing DataFrame columns.

    A DataFrame column whose name collides with a DataFrame attribute (a
    method or property) is unreachable via dotted access: ``df.sample`` returns
    the bound method, so ``df[df.sample == 'A']`` builds an all-False mask and
    ``df[False]`` raises ``KeyError: False``. The fix is ``df["sample"]``.

    Precision guards (a wrong rewrite is worse than no rewrite):
      * the attribute must be a real pandas attribute, AND
      * the same name must be *confirmed as a column* elsewhere in the file, AND
      * the access must be a comparison (==/!=/</<=/>/>=/.isin/.isna/.notna),
        never a method call ``df.sample(...)`` or an assignment.
    """
    g = root / task / "tests" / "grade.py"
    if not g.is_file():
        return
    text = g.read_text(encoding="utf-8")
    confirmed = _confirmed_columns(text)
    if not confirmed:
        return
    lines = text.splitlines()
    changed = 0
    for idx, code in _strip_pandas_comments(text):
        for m in re.finditer(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)", code):
            var, attr = m.group(1), m.group(2)
            if attr not in _PANDAS_DF_ATTRS or attr not in confirmed:
                continue
            after = code[m.end():]
            if after.startswith("(") or re.match(r"\s*=[^=]", after):
                continue  # method call or assignment, not a column access
            if not re.match(r"\s*(==|!=|<=|>=|<|>|\.isin\b|\.isna\b|\.notna\b)",
                            after):
                continue
            old = f"{var}.{attr}"
            new = f'{var}["{attr}"]'
            if old in lines[idx]:
                lines[idx] = lines[idx].replace(old, new)
                changed += 1
    if changed:
        _backup(task, g)
        g.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""),
                     encoding="utf-8")
        applied.append(
            f"{task}: E rewrote {changed} attribute-shadowed column access(es) "
            f"in tests/grade.py to bracket form")


# --------------------------------------------------------------------------
# Class F: grader-crash swallowed into score 0 ("silent swallow")
# --------------------------------------------------------------------------
#
# VERIFIED on gonnect-go-binn. The grader is wrapped as
#
#     score = 0.0
#     try:
#         ...score the agent output...
#     except Exception:
#         score = 0.0
#     write(score)
#
# so a *bug in the grader itself* (here: ``acts[i][j]`` where the schema is
# samples-major, giving IndexError) produces exactly the same reward as a
# genuinely wrong agent answer: 0.0 with no trace. Across the 100-task set
# this makes 0-scoring tasks unusable -- you cannot tell "task is
# unsolvable" from "grader is broken", which invalidates every harness
# comparison built on it.
#
# The repair makes crashes *diagnosable without changing the score*: the
# except block records the traceback to /logs/verifier/grader_error.txt and
# prints it to stdout (which harbor captures in verifier/test-stdout.txt).
# A crashed grader now leaves evidence; a legitimately wrong answer still
# scores 0 but writes no error file.
def fix_class_f(task: str, root: Path, applied: list) -> None:
    g = root / task / "tests" / "grade.py"
    if not g.is_file():
        return
    text = g.read_text(encoding="utf-8")
    if "grader_error.txt" in text:
        return  # already instrumented

    lines = text.splitlines()
    out_lines: list[str] = []
    changed = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)except(\s+Exception)?(\s+as\s+\w+)?\s*:\s*$", line)
        if not m:
            out_lines.append(line); i += 1; continue
        base = m.group(1)
        # collect the except block body
        j = i + 1
        body: list[str] = []
        while j < len(lines):
            nxt = lines[j]
            if nxt.strip() == "":
                body.append(nxt); j += 1; continue
            ind = re.match(r"^(\s*)", nxt).group(1)
            if len(ind) <= len(base):
                break
            body.append(nxt); j += 1
        block = "\n".join(body)
        # only instrument a *silent* swallow: maps to score 0, records nothing
        is_zero = re.search(r"(score|reward|result|total)\s*=\s*0(\.0)?\b", block)
        records = re.search(
            r"(error|err\b|exc|exception|traceback|repr|print|details|report|"
            r"message|str\()", block, re.I)
        if not (is_zero and not records):
            out_lines.append(line); out_lines.extend(body); i = j; continue

        inner = base + "    "
        out_lines.append(line)
        out_lines.append(inner + "import traceback as _tb, sys as _sys")
        out_lines.append(inner + "_err = _tb.format_exc()")
        out_lines.append(inner + "try:")
        out_lines.append(inner + "    import os as _os")
        out_lines.append(inner + "    _os.makedirs('/logs/verifier', exist_ok=True)")
        out_lines.append(inner + "    open('/logs/verifier/grader_error.txt','w')"
                                 ".write(_err)")
        out_lines.append(inner + "except Exception:")
        out_lines.append(inner + "    pass")
        out_lines.append(inner + "print('GRADER_EXCEPTION\\n' + _err,"
                                 " file=_sys.stderr, flush=True)")
        out_lines.extend(body)
        changed += 1
        i = j

    if changed:
        _backup(task, g)
        g.write_text("\n".join(out_lines) + ("\n" if text.endswith("\n") else ""),
                     encoding="utf-8")
        applied.append(
            f"{task}: F instrumented {changed} silent swallow(s) -> grader "
            f"crashes now recorded to /logs/verifier/grader_error.txt")


# --------------------------------------------------------------------------
# Class G: schema-transposed index in grader (sample-major vs term-major)
# --------------------------------------------------------------------------
#
# VERIFIED on gonnect-go-binn. instruction.md says ``activations`` is
# sample-major: "Rows in activations must align with samples; columns must
# align with terms." The grader instead indexed ``acts[i][j]`` under
# ``for i,t in enumerate(terms) / for j,s in enumerate(samples)`` -- i.e. it
# read term-major -- so on any task with more samples than terms it raised
# IndexError at the third sample. Oracle scored 0.0; with the correct index
# the same solve.sh scores 0.994.
#
# The fix is a pure local transpose swap and is only applied when the
# enclosing comprehension's enumerate variables prove the mismatch.
_G_TRANSPOSE = (
    # (old, new)
    ("float(acts[i][j])", "float(acts[j][i])"),
    ("acts[i][j]", "acts[j][i]"),
    ("float(preds[i][j])", "float(preds[j][i])"),
)


def _g_is_transposed(text: str) -> bool:
    """True when a single statement pairs i<->terms and j<->samples.

    The tell-tale is the enumerate pairing, in either textual order:
      for i,t in enumerate(terms)  ...  for j,s in enumerate(samples)
      for j,s in enumerate(samples) ...  for i,t in enumerate(terms)
    Only then is ``acts[i][j]`` provably the transposed read of the
    sample-major schema declared in instruction.md.
    """
    i_terms = re.compile(r"for\s+i\s*,\s*\w+\s+in\s+enumerate\(\s*terms\s*\)")
    j_samples = re.compile(r"for\s+j\s*,\s*\w+\s+in\s+enumerate\(\s*samples\s*\)")
    for line in text.splitlines():
        if "acts[" not in line:
            continue
        if i_terms.search(line) and j_samples.search(line):
            return True
    return False


def fix_class_g(task: str, root: Path, applied: list) -> None:
    g = root / task / "tests" / "grade.py"
    if not g.is_file():
        return
    text = g.read_text(encoding="utf-8")
    if not _g_is_transposed(text):
        return
    changed = 0
    for old, new in _G_TRANSPOSE:
        if old not in text:
            continue
        _backup(task, g)
        text = text.replace(old, new)
        changed += 1
    if changed:
        g.write_text(text, encoding="utf-8")
        applied.append(
            f"{task}: G swapped transposed index {changed}x "
            f"(schema is sample-major per instruction.md)")


# --------------------------------------------------------------------------
# Class B2: bare ``python`` in scripts, image only ships ``python3``
# --------------------------------------------------------------------------
#
# VERIFIED on biographx-rna-localization: Dockerfile is ubuntu:22.04 which
# installs only ``python3``/``python3-pip`` (Debian/Ubuntu have no ``python``
# package), yet solve.sh runs ``python /app/solve.py`` and the verifier's
# test.sh runs ``python /tests/grade.py`` -- BOTH exit 127, so oracle,
# codex, pi and dsh all see reward 0 for a provisioning reason unrelated to
# the task itself. The fix is to rewrite the invocation to ``python3`` ONLY
# where the image provably has no ``python`` (no python: base, no
# python-is-python3, no symlink).
_B2_WHITELIST_SKIP = re.compile(
    r"(FROM\s+python:|continuumio/|miniconda|condaforge|mambaorg/|"
    r"python-is-python3|ln\s+-s[^\n]*python3[^\n]*\bpython\b|"
    r"update-alternatives[^\n]*python)")


def _b2_image_lacks_python(root: Path, task: str) -> bool:
    df = root / task / "environment" / "Dockerfile"
    if not df.is_file():
        return False
    return not bool(_B2_WHITELIST_SKIP.search(
        df.read_text(encoding="utf-8", errors="replace")))


def fix_b2(task: str, root: Path, applied: list) -> None:
    if not _b2_image_lacks_python(root, task):
        return
    for sub in ("solution/solve.sh", "tests/test.sh"):
        p = root / task / sub
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"(?<![\w./-])python\s+(?!-)", text):
            continue
        if _B2_WHITELIST_SKIP.search(text):
            # e.g. the fixer's own instrumentation mentions python-is-python3
            continue
        new_text = re.sub(r"(?<![\w./-])python(\s+)(?!-)",
                          r"python3\1", text)
        if new_text == text:
            continue
        _backup(task, p)
        p.write_text(new_text, encoding="utf-8")
        applied.append(
            f"{task}: B2 {sub}: bare ``python`` -> ``python3`` "
            f"(image provides no python alias)")


# --------------------------------------------------------------------------
# Class A12: dependency version constraint admits an API-incompatible release
# --------------------------------------------------------------------------
#
# VERIFIED on hippo-tead-kidney-transcriptomics. The Dockerfile pins
# ``pydeseq2>=0.4,<0.5`` but BOTH solve.sh and tests/grade.py call
# ``DeseqDataSet(..., design='~condition')``. The ``design`` keyword only
# exists from pydeseq2 0.5.0 -- the 0.4.x line uses ``design_factors`` -- so
# pip installs 0.4.12 and the call raises
# ``TypeError: DeseqDataSet.__init__() got an unexpected keyword argument
# 'design'``. Both the oracle solution and the grader die, and the task looks
# unsolvable. The fix widens the pin to the line that matches the code.
_A12_PINS = [
    # (regex on the Dockerfile requirement, replacement, proof regex on code)
    (re.compile(r"'pydeseq2>=0\.4,<0\.5'"),
     "'pydeseq2>=0.5,<0.6'",
     re.compile(r"DeseqDataSet\([^)]*\bdesign\s*=")),
]


def fix_a12(task: str, root: Path, applied: list) -> None:
    df = root / task / "environment" / "Dockerfile"
    if not df.is_file():
        return
    dockerfile = df.read_text(encoding="utf-8")
    code = ""
    for sub in ("solution/solve.sh", "solution/solve.py", "tests/grade.py",
                "tests/gold_gen.py"):
        p = root / task / sub
        if p.is_file():
            code += p.read_text(encoding="utf-8", errors="replace")
    new = dockerfile
    changed = 0
    for pat, repl, proof in _A12_PINS:
        if not pat.search(new) or not proof.search(code):
            continue
        new = pat.sub(repl, new)
        changed += 1
    if changed:
        _backup(task, df)
        df.write_text(new, encoding="utf-8")
        applied.append(
            f"{task}: A12 widened dependency pin to match the API actually "
            f"used by solve/grade (pydeseq2 >=0.5 for ``design=``)")


# --------------------------------------------------------------------------
# Class A13: numpy array used as a plain-Python sequence index
# --------------------------------------------------------------------------
#
# VERIFIED on fish-morphology-life-history-phylo-prediction:
#   tri = np.array([...], int)
#   y = lh_idx.loc[tree_names[tri], "col"]      # tree_names is a *list*
# -> TypeError: only integer scalar arrays can be converted to a scalar index
# Indexing a Python list with a numpy integer array is not supported; the
# intent is clearly element-wise gather. The safe, semantics-preserving
# rewrite is to materialise the gather explicitly.
_A13_RE = re.compile(
    r"(?P<seq>[A-Za-z_]\w*)\[(?P<idx>[A-Za-z_]\w*)\]")


def fix_a13(task: str, root: Path, applied: list) -> None:
    for sub in ("solution/solve.sh", "solution/solve.py",
                "tests/gold_gen.py", "tests/grade.py"):
        p = root / task / sub
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        # need a python list name and a numpy int-array name in the same file
        lists = set(re.findall(r"^([A-Za-z_]\w*)\s*=\s*\[", text, re.M))
        np_arrays = set(re.findall(
            r"^([A-Za-z_]\w*)\s*=\s*np\.array\(.*?,\s*int\)", text, re.M))
        if not (lists and np_arrays):
            continue
        changed = 0
        lines = text.splitlines()
        for i, line in enumerate(lines):
            for seq in lists:
                for idx in np_arrays:
                    old = f"{seq}[{idx}]"
                    if old in line:
                        lines[i] = line = line.replace(
                            old, f"[{seq}[_k] for _k in {idx}]")
                        changed += 1
        if changed:
            _backup(task, p)
            p.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""),
                         encoding="utf-8")
            applied.append(
                f"{task}: A13 replaced list[np-array] gather with explicit "
                f"comprehension ({changed}x)")


# --------------------------------------------------------------------------
# Class A14: ternary-in-tuple unpacking bug (dbd-class-annotation)
# --------------------------------------------------------------------------
#
# VERIFIED on dbd-class-annotation. Line 133 of the oracle solve reads
#
#   start, end = pos, pos + quant[...]["w"] if False else (pos, pos + ...)
#
# Python parses the RHS as the 2-tuple ``(pos, ((pos, pos + ...)))`` -- the
# parenthesized ternary produces a *nested* tuple, so ``end`` is a tuple and
# ``end - start`` raises ``TypeError: unsupported operand type(s) for -:
# 'tuple' and 'int'``. The dead ``if False`` arm and the stray parentheses
# are clearly leftover scaffolding: the intent is
#
#   if cls != "None": start, end = pos, pos + quant[cls]["w"]
#   else:             start, end = pos, pos + len(models[ranked[0][0]])
_A14_OLD = (
    'start, end = pos, pos + quant[ranked[0][0]]["w"] if False else '
    '(pos, pos + quant[cls]["w"] if cls != "None" else '
    'pos + len(models[ranked[0][0]]))'
)
_A14_NEW = (
    'if cls != "None":\n'
    '            start, end = pos, pos + quant[cls]["w"]\n'
    '        else:\n'
    '            start, end = pos, pos + len(models[ranked[0][0]])'
)


def fix_a14(task: str, root: Path, applied: list) -> None:
    p = root / task / "solution" / "solve.sh"
    if not p.is_file():
        return
    text = p.read_text(encoding="utf-8", errors="replace")
    if _A14_OLD not in text:
        return
    _backup(task, p)
    p.write_text(text.replace(_A14_OLD, _A14_NEW, 1), encoding="utf-8")
    applied.append(
        f"{task}: A14 fixed tuple-unpacking of a parenthesized ternary "
        f"(end became a nested tuple -> tuple - int TypeError)")


# --------------------------------------------------------------------------
# Class A15: singular normal-equations solve in gold generation
# --------------------------------------------------------------------------
#
# VERIFIED on edgemap-spatial-lr-heritability: tests/grade.py caught
# ``LinAlgError('Singular matrix')`` from gold_gen.py's ``np.linalg.solve`` --
# the GOLD ITSELF could not be generated, so every harness scores 0 for a
# reason internal to the task. The design matrix in the per-gene loop
# (``X = [1, lv, li]``) is rank-deficient for genes whose liver annotation
# aligns with the background column. The numerically safe equivalent that
# matches least squares even under rank deficiency is the pseudo-inverse
# solve -- same solution when XtX is full rank, defined when it is not.
_A15_SOLVE = re.compile(
    r"beta = np\.linalg\.solve\(XtX, X\.T @ y\)")


def fix_a15(task: str, root: Path, applied: list) -> None:
    for sub in ("tests/gold_gen.py", "solution/solve.py", "solution/solve.sh"):
        p = root / task / sub
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if not _A15_SOLVE.search(text):
            continue
        _backup(task, p)
        text = _A15_SOLVE.sub(
            "beta = np.linalg.pinv(XtX) @ (X.T @ y)", text)
        # inv(XtX) for the standard errors must degrade the same way
        text = text.replace("np.linalg.inv(XtX)", "np.linalg.pinv(XtX)")
        p.write_text(text, encoding="utf-8")
        applied.append(
            f"{task}: A15 replaced singular normal-equations solve/inv with "
            f"pinv in {sub} (gold generation crashed on LinAlgError)")


# --------------------------------------------------------------------------
# Class A16: scalar call applied to the vector instead of its reduction
# --------------------------------------------------------------------------
#
# VERIFIED on hpabc-ml-directed-evolution tests/grade.py:49
#   return float(math.sqrt(float((np.log(p) - np.log(g)) ** 2).mean()))
# ``float(...)`` wraps the squared VECTOR, so ``.mean()`` is called on a
# Python float -> AttributeError in general and, before that, the float()
# call raises ``TypeError: only length-1 arrays can be converted to Python
# scalars``. The grader dies computing its very first RMSE and the task
# scores 0 for every harness. Fix: wrap the scalar AFTER the reduction.
_A16_PAIRS = [
    ("float(math.sqrt(float((np.log(p) - np.log(g)) ** 2).mean()))",
     "float(math.sqrt(((np.log(p) - np.log(g)) ** 2).mean()))"),
]


def fix_a16(task: str, root: Path, applied: list) -> None:
    for sub in ("tests/grade.py", "tests/gold_gen.py",
                "solution/solve.py", "solution/solve.sh"):
        p = root / task / sub
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        changed = 0
        for old, new in _A16_PAIRS:
            if old in text:
                text = text.replace(old, new)
                changed += 1
        if changed:
            _backup(task, p)
            p.write_text(text, encoding="utf-8")
            applied.append(
                f"{task}: A16 moved float() after the reduction "
                f"({changed}x in {sub}; grader crashed on vector->scalar)")


# --------------------------------------------------------------------------
# Class A17: batch vector reshaped as a single row (crispaipe)
# --------------------------------------------------------------------------
#
# VERIFIED on crispaipe-conformal-pegRNA solution/solve.sh. The density
# helper is called two ways:
#     dirichlet_density_logpdf(obs[cal_idx], alpha_cal)   # alpha (300, 3)
#     dirichlet_density_logpdf(G, alpha_eval[i])          # alpha (3,)
# but it unconditionally does ``alpha.reshape(1, 3)`` -> for the batch case
# that is ``ValueError: cannot reshape array of size 900 into shape (1,3)``
# and the oracle dies before writing any output. The parameter is already
# shaped correctly for the row-broadcast formula below (``sum(axis=1)``),
# so accepting both shapes is a no-op for the scalar path and a fix for the
# batch path.
_A17_OLD = """def dirichlet_density_logpdf(y, alpha):
    a = alpha.reshape(1, 3)"""
_A17_NEW = """def dirichlet_density_logpdf(y, alpha):
    a = np.atleast_2d(alpha)
    if a.shape[1] != 3:
        a = a.reshape(-1, 3)"""


def fix_a17(task: str, root: Path, applied: list) -> None:
    for sub in ("solution/solve.sh", "solution/solve.py",
                "tests/gold_gen.py", "tests/grade.py"):
        p = root / task / sub
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if _A17_OLD not in text:
            continue
        _backup(task, p)
        p.write_text(text.replace(_A17_OLD, _A17_NEW, 1), encoding="utf-8")
        applied.append(
            f"{task}: A17 accepted both batch (n,3) and row (3,) alpha in "
            f"{sub} (reshape(1,3) crashed on the 900-element batch)")


# --------------------------------------------------------------------------
# Class A18: skbio DistanceMatrix.from_iterable needs a *callable* metric
# --------------------------------------------------------------------------
#
# VERIFIED on housing-facility-arsenic-microbiome solution/solve.sh:64:
#   DistanceMatrix.from_iterable(sp.values, metric='braycurtis', keys=...)
# Modern scikit-bio's ``from_iterable`` does ``dm[i, j] = metric(a, b)`` and
# therefore requires a callable -- a string is a TypeError ('str' object is
# not callable). Nothing is written to the output dir, the grader's
# downstream ``subprocess.run(['python', '/app/analysis.py', ...])`` fails,
# and every harness scores 0. The equivalent callable is
# ``scipy.spatial.distance.braycurtis``.
_A18_PAIRS = [
    ("metric='braycurtis'", "metric=braycurtis"),
    ('metric="braycurtis"', "metric=braycurtis"),
]


def fix_a18(task: str, root: Path, applied: list) -> None:
    for sub in ("solution/solve.sh", "solution/solve.py",
                "tests/gold_gen.py", "tests/grade.py"):
        p = root / task / sub
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "DistanceMatrix.from_iterable" not in text:
            continue
        changed = 0
        for old, new in _A18_PAIRS:
            if old in text:
                text = text.replace(old, new)
                changed += 1
        if not changed:
            continue
        # make sure the callable is importable in that file
        if "from scipy.spatial.distance import" in text:
            if not re.search(r"from scipy\.spatial\.distance import [^\n]*\bbraycurtis\b",
                             text):
                text = re.sub(r"from scipy\.spatial\.distance import ",
                              "from scipy.spatial.distance import braycurtis, ",
                              text, count=1)
        else:
            anchor = "import numpy as np"
            if anchor in text:
                text = text.replace(
                    anchor, anchor + "\nfrom scipy.spatial.distance import braycurtis",
                    1)
        _backup(task, p)
        p.write_text(text, encoding="utf-8")
        applied.append(
            f"{task}: A18 passed a callable metric to DistanceMatrix."
            f"from_iterable ({changed}x in {sub}; string metric is a TypeError)")


# --------------------------------------------------------------------------
# Class A19: Biopython aligned-blocks are numpy arrays, not objects
# --------------------------------------------------------------------------
#
# VERIFIED on nesso-binding-affinity-eval. BOTH solution/solve.sh and
# tests/grade.py do
#     for (a, b) in zip(best.aligned[0], best.aligned[1]):
#         for i in range(a.start, a.end):
# In current Biopython ``Alignment.aligned`` is a 2xNx2 *numpy array* of
# [start, end) block coordinates -- each ``a`` is a length-2 ndarray, and
# ``a.start`` raises ``AttributeError: 'numpy.ndarray' object has no
# attribute 'start'``. The oracle and the grader die at the first identity
# computation. Fix: index the endpoints positionally.
_A19_OLD = """    for (a, b) in zip(best.aligned[0], best.aligned[1]):
        for i in range(a.start, a.end):"""
_A19_NEW = """    for (a, b) in zip(best.aligned[0], best.aligned[1]):
        for i in range(int(a[0]), int(a[1])):"""


def fix_a19(task: str, root: Path, applied: list) -> None:
    for sub in ("tests/grade.py", "tests/gold_gen.py",
                "solution/solve.sh", "solution/solve.py"):
        p = root / task / sub
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if _A19_OLD not in text:
            continue
        _backup(task, p)
        p.write_text(text.replace(_A19_OLD, _A19_NEW), encoding="utf-8")
        applied.append(
            f"{task}: A19 read Biopython aligned blocks as ndarrays "
            f"({sub}: a.start/a.end -> a[0]/a[1])")


# --------------------------------------------------------------------------
# Class A20-A25 (second sweep, after the oracle run that produced
# out/task_validation/summary.json with 61/98 pass). Every one of these was
# confirmed by running the failing code locally against the real generated
# data -- none of them is a guess.
# --------------------------------------------------------------------------

def fix_a20(task: str, root: Path, applied: list) -> None:
    """PDB writer/parser column misalignment (ndh1ms).

    VERIFIED end-to-end: the generator's ``write_pdb`` emits
    ``{name:^4s}{resname:>3s}`` (name at cols 12-15, resName at 16-18) while
    every consumer parses the *standard* PDB columns (name 12-15, resName
    17-19, chainID 21, resSeq 22-25). The off-by-one means ``resname`` is
    read as the last two letters of the real residue plus a space, so no atom
    ever matches ``'CO2'`` / ``'HOH'`` / a polar residue. The graph ends up
    with **zero nodes**, the BFS returns ``best = None``, and the oracle dies
    with ``TypeError: 'NoneType' object is not iterable``.

    Local proof: before the fix ``nodes: 0``; after it ``nodes: 14`` and the
    BFS finds the expected 7-member proton wire
    ``CO2 -> HOH:1..HOH:5 -> NdhF4:410:HIS:NE2``.

    The fix inserts the missing altLoc column (one space) between the atom
    name and the residue name, in BOTH the ATOM and HETATM branches.
    """
    for sub in ("environment/data/gen_inputs.py", "tests/gold_gen.py",
                "tests/gen_inputs.py"):
        p = root / task / sub
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        # NOTE: the literal record differs per branch -- "ATOM  " carries two
        # trailing spaces (6 cols) while "HETATM" is exactly 6 -- so an exact
        # string match silently skips one branch. Match structurally instead.
        new, n = re.subn(
            r"(ATOM\s*|HETATM)\{atm\['serial'\]:5d\} \{atm\['name'\]:\^4s\}"
            r"\{atm\['resname'\]:>3s\}",
            lambda m: (m.group(1) + "{atm['serial']:5d} "
                       "{atm['name']:<4s} {atm['resname']:>3s}"),
            text)
        if n:
            _backup(task, p)
            p.write_text(new, encoding="utf-8")
            applied.append(
                f"{task}: A20 {sub} write_pdb inserts the missing altLoc "
                f"column ({n} call site(s)) so resName/chainID/resSeq land in "
                "the standard PDB columns the parsers actually read")


def fix_a21(task: str, root: Path, applied: list) -> None:
    """``exit 0`` in test.sh short-circuits the reward sanitizer.

    ``_fix_reward.py`` exists precisely to move nested objects out of
    reward.json (harbor type-checks it as ``dict[str, float | int]`` and a
    nested dict aborts the trial with a pydantic ValidationError *after* the
    agent finished). But ten tasks have ``exit 0`` *before* the sanitizer
    line, so the sanitizer never runs and the trial dies as an exception.

    The fix removes the premature ``exit 0`` and lets the sanitizer run; the
    script still ends ``|| true`` so a missing sanitizer is not fatal.
    """
    t = root / task / "tests" / "test.sh"
    if not t.is_file():
        return
    lines = t.read_text(encoding="utf-8", errors="replace").splitlines()
    exit_i = next((i for i, l in enumerate(lines) if l.strip() == "exit 0"),
                  None)
    fix_i = next((i for i, l in enumerate(lines)
                  if "_fix_reward.py" in l), None)
    if exit_i is None or fix_i is None or exit_i > fix_i:
        return
    del lines[exit_i]
    _backup(task, t)
    t.write_text("\n".join(lines) + "\n", encoding="utf-8")
    applied.append(f"{task}: A21 removed premature `exit 0` from test.sh so "
                   "the _fix_reward.py sanitizer actually runs (nested "
                   "reward payloads were aborting the trial)")


def fix_a22(task: str, root: Path, applied: list) -> None:
    """ppo2: ``get_aligned_pairs()`` yields ``qpos=None`` on deletions.

    A read with a deletion at the mutation site produces ``(None, rpos)``;
    ``read.query_sequence[None]`` then raises ``TypeError: string indices
    must be integers, not 'NoneType'``. Guard the lookup.
    """
    for sub in ("solution/solve.sh", "solution/solve.py",
                "tests/grade.py", "tests/gold_gen.py"):
        p = root / task / sub
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "get_aligned_pairs()" not in text:
            continue
        new = text
        # guard the qpos lookup inside the rpos == mut_idx branch
        new = new.replace(
            "            if rpos == mut_idx:\n"
            "                base = read.query_sequence[qpos].upper()\n",
            "            if rpos == mut_idx and qpos is not None:\n"
            "                base = read.query_sequence[qpos].upper()\n")
        new = new.replace(
            "        if rpos == mut_idx:\n"
            "            base = read.query_sequence[qpos].upper()\n",
            "        if rpos == mut_idx and qpos is not None:\n"
            "            base = read.query_sequence[qpos].upper()\n")
        if new != text:
            _backup(task, p)
            p.write_text(new, encoding="utf-8")
            applied.append(f"{task}: A22 {sub} guards qpos is not None in "
                           "get_aligned_pairs (deletions yield (None, rpos))")


def fix_a23(task: str, root: Path, applied: list) -> None:
    """sca7: ``edge_names`` lookup uses the wrong member order.

    ``edge_names`` is built as ``f"{NETWORKS[i]}-{NETWORKS[j]}"`` for ``i<j``
    over a list whose order puts *DefaultMode* (index 6) before *Cerebellar*
    (index 7). So the canonical name is ``DefaultMode-Cerebellar`` and the
    literal ``"Cerebellar-DefaultMode"`` is never in the list ->
    ``ValueError: 'Cerebellar-DefaultMode' is not in list`` in the grader's
    own gold generator.

    The original fix_a3 targeted ``environment/data/gen_inputs.py``, but the
    code that actually runs is ``tests/gold_gen.py`` -- so the repair landed
    in a file nothing executes. This fixer covers BOTH files and rewrites
    every occurrence (the earlier one used a single-shot replace).
    """
    for sub in ("tests/gold_gen.py", "environment/data/gen_inputs.py",
                "tests/grade.py", "solution/solve.sh"):
        p = root / task / sub
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        n = text.count('edge_names.index("Cerebellar-DefaultMode")')
        if not n:
            continue
        _backup(task, p)
        text = text.replace('edge_names.index("Cerebellar-DefaultMode")',
                            'edge_names.index("DefaultMode-Cerebellar")')
        p.write_text(text, encoding="utf-8")
        applied.append(f"{task}: A23 {sub} rewrote {n} Cerebellar-DefaultMode "
                       "lookup(s) to DefaultMode-Cerebellar (i<j ordering)")


def fix_a24(task: str, root: Path, applied: list) -> None:
    """rar1: Ubuntu 22.04's iqtree package installs ``iqtree2``, not ``iqtree``.

    VERIFIED in a ubuntu:22.04 container: ``apt-get install iqtree`` places
    ``/usr/bin/iqtree2`` (plus ``-mpi``/``-omp`` variants); there is no
    ``iqtree`` on PATH. The oracle therefore dies at
    ``/solution/solve.sh: line 6: iqtree: command not found`` (exit 127).

    The fix adds a symlink in the Dockerfile rather than editing the solve
    script, so the task instruction (which names IQ-TREE) and any agent
    writing ``iqtree`` both work.
    """
    df = root / task / "environment" / "Dockerfile"
    if not df.is_file():
        return
    text = df.read_text(encoding="utf-8", errors="replace")
    if "iqtree" not in text or "ln -s" in text:
        return
    _backup(task, df)
    # append a dedicated RUN layer so the apt continuation lines stay
    # untouched (a blind in-place string replace would leave a dangling
    # backslash and re-associate the rm -rf with the wrong command)
    df.write_text(text.rstrip("\n") +
                  "\nRUN ln -sf /usr/bin/iqtree2 /usr/local/bin/iqtree\n",
                  encoding="utf-8")
    applied.append(f"{task}: A24 Dockerfile symlinks /usr/local/bin/iqtree -> "
                   "/usr/bin/iqtree2 (ubuntu:22.04 ships iqtree2)")


def fix_a25(task: str, root: Path, applied: list) -> None:
    """3d-membrane: the trimesh signature fix never reached solve.sh.

    fix_a6 rglobs ``*.py``, but the oracle's own reference solution lives in
    ``solution/solve.sh`` (an inline heredoc), which still calls
    ``discrete_*_curvature_measure(mesh, 3.0)``. The grader's gold_gen was
    fixed, so gold and oracle disagree -- exactly the "grader and oracle must
    stay byte-identical" hazard. Extend the repair to every text file that
    embeds the call, not just .py.
    """
    d = root / task
    if not d.is_dir():
        return
    for p in sorted(d.rglob("*")):
        if not p.is_file() or any(
                part in (".git", "__pycache__") for part in p.parts):
            continue
        if p.suffix not in (".py", ".sh", ""):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        changed = False
        for fn in ("discrete_mean_curvature_measure",
                   "discrete_gaussian_curvature_measure"):
            old = f"trimesh.curvature.{fn}(mesh, 3.0)"
            new = f"trimesh.curvature.{fn}(mesh, mesh.vertices, 3.0)"
            if old in text:
                text = text.replace(old, new)
                changed = True
        if changed:
            _backup(task, p)
            p.write_text(text, encoding="utf-8")
            applied.append(f"{task}: A25 {p.relative_to(d)} curvature call "
                           "passes points=mesh.vertices (was missed by A6, "
                           "which only globbed *.py)")


def fix_a26(task: str, root: Path, applied: list) -> None:
    """``None in [series, ...]`` is an ambiguous Series truth test.

    ``rows.iloc[0]`` returns a **Series**, and ``None in [sa, sb]`` compares
    against each element with ``==`` then reduces the resulting Series with
    ``bool()`` -- pandas raises
    ``ValueError: The truth value of a Series is ambiguous``. The intent is a
    plain identity test, so use ``any(x is None ...)``.
    """
    for sub in ("tests/grade.py", "tests/gold_gen.py",
                "solution/solve.sh", "solution/solve.py"):
        p = root / task / sub
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "None in [" not in text:
            continue
        new, n = re.subn(r"if None in \[([^\]]+)\]:",
                         lambda m: "if any(_x is None for _x in "
                                   f"({m.group(1)})):", text)
        if n:
            _backup(task, p)
            p.write_text(new, encoding="utf-8")
            applied.append(f"{task}: A26 {sub} `None in [...]` -> "
                           f"`any(x is None ...)` ({n} site(s)); a Series in "
                           "that list makes the truth value ambiguous")


def fix_a27(task: str, root: Path, applied: list) -> None:
    """A module the reference solution imports is not installed in the image.

    ``ModuleNotFoundError: No module named 'pandas'`` (tmvec2s) -- the grader
    and the oracle agree, but the Dockerfile's pip line omits a dependency
    that ``solve.sh`` imports at line 1. Detect by cross-referencing the
    top-level imports of every solution/tests file against the Dockerfile's
    ``pip install`` line, and add what is missing.
    """
    KNOWN = ("numpy", "pandas", "scipy", "sklearn", "scikit-learn",
             "biopython", "statsmodels", "networkx", "matplotlib",
             "seaborn", "rdkit", "pyteomics", "catboost", "hmmlearn",
             "primer3", "pysam", "trimesh", "rtree", "skbio")
    # import name -> PyPI distribution name(s). ``sklearn`` is installed by
    # the ``scikit-learn`` distribution, so a naive token match reports a
    # false positive on every task that already has it.
    ALIAS = {
        "sklearn": ("scikit-learn", "sklearn"),
        "skbio": ("scikit-bio", "skbio"),
    }
    df = root / task / "environment" / "Dockerfile"
    if not df.is_file():
        return
    df_text = df.read_text(encoding="utf-8", errors="replace")
    installed_l = {x.lower() for x in re.findall(r"[a-zA-Z0-9_.-]+", df_text)}

    wanted: set[str] = set()
    for sub in ("solution/solve.sh", "solution/solve.py",
                "tests/grade.py", "tests/gold_gen.py"):
        p = root / task / sub
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8",
                                errors="replace").splitlines():
            m = re.match(r"\s*(?:import|from)\s+([a-zA-Z0-9_]+)", line)
            if not m:
                continue
            mod = m.group(1)
            if mod not in KNOWN:
                continue
            names = ALIAS.get(mod, (mod,))
            if not any(n.lower() in installed_l for n in names):
                wanted.add(mod)
    if not wanted:
        return
    # sklearn is the PyPI name for the `sklearn` import
    add = sorted(w.replace("sklearn", "scikit-learn") for w in wanted)
    _backup(task, df)
    m = re.search(r"pip install --no-cache-dir ([^\n\\]+)", df_text)
    if m and "\\" not in m.group(0):
        line = m.group(0)
        df.write_text(df_text.replace(
            line, line + " " + " ".join(add), 1), encoding="utf-8")
    else:
        df.write_text(
            df_text.rstrip("\n") +
            "\nRUN pip install --no-cache-dir " + " ".join(add) + "\n",
            encoding="utf-8")
    applied.append(f"{task}: A27 Dockerfile installs missing module(s) "
                   f"{', '.join(add)} (imported by the solution/grader but "
                   "absent from the image)")


def fix_a28(task: str, root: Path, applied: list) -> None:
    """A helper module shipped in ``environment/data`` is not importable.

    Two distinct ways this bites:
      * the Dockerfile's cleanup line deletes it (``rm ... simulation.py``), or
      * it stays under ``/app/data`` which is not on ``sys.path``.

    The verifier runs with cwd ``/tests`` and the agent with cwd ``/app``, so
    a module imported by ``tests/grade.py`` must land in ``/tests`` while one
    imported by ``solution/solve.sh`` must land in ``/app``. Detect the
    imports that actually occur (not "every .py in data/"), copy each helper
    to both locations, and drop it from any ``rm`` list so the copy cannot
    reference a deleted file.
    """
    data = root / task / "environment" / "data"
    df = root / task / "environment" / "Dockerfile"
    if not data.is_dir() or not df.is_file():
        return
    df_text = df.read_text(encoding="utf-8", errors="replace")
    if "cp /app/data/" in df_text:
        return

    helpers = {p.stem for p in data.glob("*.py")
               if not p.stem.startswith(("gen_", "make_", "__init__"))}
    if not helpers:
        return

    # which helpers are actually imported, and by which side?
    imported: set[str] = set()
    for sub in ("solution/solve.sh", "solution/solve.py",
                "tests/grade.py", "tests/gold_gen.py"):
        p = root / task / sub
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8",
                                errors="replace").splitlines():
            m = re.match(r"\s*(?:import|from)\s+([a-zA-Z0-9_]+)", line)
            if m and m.group(1) in helpers:
                imported.add(m.group(1))
    if not imported:
        return

    _backup(task, df)
    lines = [f"RUN cp /app/data/{h}.py /app/{h}.py && "
             f"cp /app/data/{h}.py /tests/{h}.py" for h in sorted(imported)]
    block = "\n".join(lines)
    # Insert immediately after the data COPY, i.e. BEFORE any cleanup RUN that
    # deletes the helper -- copying after the delete would fail the build.
    # If there is no `COPY data/` line there is no safe anchor: appending at
    # the end can land after a `rm`, so skip rather than corrupt the image.
    anchor = re.search(r"^COPY data/ /app/data/\s*$", df_text, re.M)
    if anchor is None:
        return
    new = (df_text[:anchor.end()] + "\n" + block +
           df_text[anchor.end():])
    df.write_text(new, encoding="utf-8")
    applied.append(f"{task}: A28 Dockerfile copies imported helper module(s) "
                   f"{', '.join(sorted(imported))} to /app and /tests (they "
                   "were either deleted by the cleanup RUN or left in "
                   "/app/data, off sys.path)")


_A29_OLD = (
    "        if roi_expr == 'all':\n"
    "            vert_mask = np.ones(len(mesh.vertices), dtype=bool)\n"
    "        else:\n"
    "            vert_mask = eval(roi_expr)\n")
_A29_NEW = (
    "        if roi_expr == 'all':\n"
    "            vert_mask = np.ones(len(mesh.vertices), dtype=bool)\n"
    "        else:\n"
    "            # x,y,z hold FACE centroids (used for face_mask); the vertex\n"
    "            # ROI must be evaluated on the vertex coordinates\n"
    "            # vx,vy,vz, otherwise the mask has len(faces) entries and\n"
    "            # vertex_areas[vert_mask] raises IndexError.\n"
    "            x, y, z = vx, vy, vz\n"
    "            vert_mask = eval(roi_expr)\n")


def fix_a29(task: str, root: Path, applied: list) -> None:
    """3d-membrane: the vertex ROI mask is built from face centroids.

    ``x, y, z`` are assigned from ``mesh.triangles_center`` (one entry per
    FACE) a few lines above, then reused for the *vertex* ROI:

        face_mask = eval(roi_expr)     # correct: len(faces)
        vert_mask = eval(roi_expr)     # WRONG: also len(faces)
        w = vertex_areas[vert_mask]    # IndexError: len(vertices) != len(faces)

    VERIFIED in-container (trimesh 5.1.1): ``IndexError: boolean index did
    not match indexed array along axis 0; size of axis is 13428 but size of
    corresponding boolean axis is 26848`` -- i.e. 13428 vertices vs 26848
    faces, and it only triggers for vol_01/vol_03/vol_04 whose ROI is not
    'all'.

    Fix: rebind ``x, y, z`` to the vertex coordinates before evaluating the
    vertex mask. ``face_mask`` is computed earlier, so the rebind cannot
    affect it.
    """
    for sub in ("tests/gold_gen.py", "tests/grade.py"):
        p = root / task / sub
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if _A29_OLD not in text or "x, y, z = vx, vy, vz" in text:
            continue
        _backup(task, p)
        p.write_text(text.replace(_A29_OLD, _A29_NEW, 1), encoding="utf-8")
        applied.append(f"{task}: A29 {sub} evaluates the vertex ROI on "
                       "vx,vy,vz instead of the face-centroid x,y,z "
                       "(mask length now matches vertex_areas)")


# The statistics block, from ``all_samples = []`` up to (not including)
# ``tau = 1.0``. Matched structurally so it works at any indent level.
_A30_RE = re.compile(
    r"([ \t]*)all_samples = \[\]\n"
    r"(?:[ \t]*[^\n]*\n)*?"
    r"[ \t]*std\[std == 0\] = 1\.0\n")


def _a30_canonical(indent: str) -> str:
    return (
        f"{indent}all_samples = []\n"
        f"{indent}for pid, d in interp_data.items():\n"
        f"{indent}    for b in biomarkers:\n"
        f"{indent}        all_samples.append(d[b])\n"
        f"{indent}# Per-biomarker pooled statistics (population, ddof=0), as the\n"
        f"{indent}# instruction requires: pool every interpolated sample of a\n"
        f"{indent}# biomarker across all patients. ``all_samples`` is ordered\n"
        f"{indent}# patient-major / biomarker-minor, so one biomarker's samples\n"
        f"{indent}# sit at a constant stride of len(biomarkers). A previous fix\n"
        f"{indent}# padded the ragged arrays into a matrix and reduced over\n"
        f"{indent}# axis=0, which yields a per-TIME-COLUMN statistic (length\n"
        f"{indent}# max(len(series))) instead of a per-biomarker one -- that\n"
        f"{indent}# made ``mean`` shape (121,) while X is (n, 12), so the\n"
        f"{indent}# broadcast in ``(X - mean) / std`` failed.\n"
        f"{indent}_n_b = len(biomarkers)\n"
        f"{indent}mean = np.zeros(_n_b)\n"
        f"{indent}std = np.zeros(_n_b)\n"
        f"{indent}for _i in range(_n_b):\n"
        f"{indent}    _pool = np.concatenate([np.atleast_1d(np.asarray(a, dtype=float))\n"
        f"{indent}                            for a in all_samples[_i::_n_b]])\n"
        f"{indent}    mean[_i] = _pool.mean()\n"
        f"{indent}    std[_i] = _pool.std()\n"
        f"{indent}    if std[_i] == 0:\n"
        f"{indent}        std[_i] = 1.0\n")


def fix_a30(task: str, root: Path, applied: list) -> None:
    """data-driven: the ragged-array fix computed the WRONG statistic.

    The task instruction says: *"Compute the mean and standard deviation
    (population, ddof=0) of each biomarker across all interpolated samples
    (pooled over all patients)."*  The previous repair padded the ragged
    per-sample series into an ``(n_samples, ncol)`` matrix and reduced with
    ``nanmean(axis=0)`` -- a per-TIME-COLUMN statistic. With variable-length
    series ``ncol`` is ``max(len(series))`` = 121, so ``mean`` came out shape
    ``(121,)`` and the later ``(X - mean) / std`` (X is ``(n_visits, 12)``)
    raised ``ValueError: operands could not be broadcast together``.

    The correct statistic is one mean/std **per biomarker**, obtained by
    pooling that biomarker's samples across all patients. Since
    ``all_samples`` is appended patient-major / biomarker-minor, a
    biomarker's samples are at a constant stride of ``len(biomarkers)``.

    Both the grader and the oracle solution are rewritten to the same
    canonical block (the "grader and oracle must stay byte-identical"
    rule), and any lingering ``mean[i][:d[b].shape[0]]`` slice is removed --
    ``mean[i]`` is a scalar, slicing it raises
    ``IndexError: invalid index to scalar variable``.
    """
    for sub in ("solution/solve.sh", "tests/grade.py", "tests/gold_gen.py",
                "solution/solve.py"):
        p = root / task / sub
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "all_samples = []" not in text or "_n_b = len(biomarkers)" in text:
            continue
        m = _A30_RE.search(text)
        if not m:
            continue
        indent = m.group(1)
        _backup(task, p)
        new = text[:m.start()] + _a30_canonical(indent) + text[m.end():]
        # the normalisation line must not slice a scalar mean[i]. The bad
        # form nests brackets inside the subscript, so a character-class
        # pattern cannot match it -- use literal replacements.
        for bad, good in (
            ("(d[b] - mean[i][:d[b].shape[0]]) / std[i][:d[b].shape[0]]",
             "(d[b] - mean[i]) / std[i]"),
            ("(d[b] - mean[i][:d[b].shape[0]]) / std[i]",
             "(d[b] - mean[i]) / std[i]"),
            ("(d[b] - mean[i]) / std[i][:d[b].shape[0]]",
             "(d[b] - mean[i]) / std[i]"),
        ):
            new = new.replace(bad, good)
        p.write_text(new, encoding="utf-8")
        applied.append(f"{task}: A30 {sub} computes per-biomarker pooled "
                       "mean/std (stride len(biomarkers)) instead of a "
                       "per-time-column reduction, matching the instruction")


def _fix_scalar_slice(task: str, root: Path, applied: list) -> None:
    """Remove ``mean[i][:d[b].shape[0]]``-style slices of a scalar.

    ``mean[i]`` (i = biomarker index) is a **scalar** once the statistics are
    per-biomarker; slicing it raises
    ``IndexError: invalid index to scalar variable``. This is a separate
    idempotent pass because A30's guard skips files it already rewrote.
    """
    for sub in ("solution/solve.sh", "solution/solve.py",
                "tests/grade.py", "tests/gold_gen.py"):
        p = root / task / sub
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "mean[i][:" not in text and "std[i][:" not in text:
            continue
        _backup(task, p)
        new = text
        for bad, good in (
            ("(d[b] - mean[i][:d[b].shape[0]]) / std[i][:d[b].shape[0]]",
             "(d[b] - mean[i]) / std[i]"),
            ("(d[b] - mean[i][:d[b].shape[0]]) / std[i]",
             "(d[b] - mean[i]) / std[i]"),
            ("(d[b] - mean[i]) / std[i][:d[b].shape[0]]",
             "(d[b] - mean[i]) / std[i]"),
        ):
            new = new.replace(bad, good)
        if new != text:
            p.write_text(new, encoding="utf-8")
            applied.append(f"{task}: A32 {sub} removes the scalar slice "
                           "mean[i][:d[b].shape[0]] (mean[i] is a scalar)")


def fix_a31(task: str, root: Path, applied: list) -> None:
    """``sort_values(['padj', 'gene'])`` on a frame whose INDEX is named gene.

    The reference solution builds a tie-break column from the index
    (``tmp['gene'] = tmp.index.astype(str)``) but ``hi``'s index level is
    itself named ``gene``. Since pandas 2.0 a sort key that names both an
    index level and a column raises
    ``ValueError: 'gene' is both an index level and a column label, which is
    ambiguous`` -- the whole grader aborts before any reward is written.

    Fix: clear the index level name so the sort key resolves unambiguously
    to the new column. Applied to every ``sort_values`` in the file that
    references ``gene``, in both the grader and the oracle solution.
    """
    for sub in ("solution/solve.sh", "solution/solve.py",
                "tests/grade.py", "tests/gold_gen.py"):
        p = root / task / sub
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "sort_values" not in text or "'gene'" not in text:
            continue
        lines = text.split("\n")
        out: list[str] = []
        n = 0
        for ln in lines:
            # NOT anchored at line start: the real line is
            # `tmp = tmp.sort_values(['padj', 'gene'], ...)`.
            m = re.search(r"\b([A-Za-z_]\w*)\.sort_values\(\[[^\]]*'gene'"
                          r"[^\]]*\]", ln)
            if m and f"{m.group(1)}.index.name = None" not in text:
                out.append(f"{m.group(1)}.index.name = None")
                n += 1
            out.append(ln)
        if not n:
            continue
        _backup(task, p)
        p.write_text("\n".join(out), encoding="utf-8")
        applied.append(f"{task}: A31 {sub} clears the index level name before "
                       f"{n} sort_values call(s) keyed on 'gene' (pandas>=2.0 "
                       "rejects a key that is both index level and column)")


# --------------------------------------------------------------------------

FIXERS = {
    "allostery-transfer-entropy-alphabet": [fix_a1],
    "rar1-sgt1-ancestral-reconstruction": [fix_a2],
    "sca7-longitudinal-fc": [fix_a3],
    "t7-promoter-redesign": [fix_a4],
    "confdock-conformal-docking": [fix_a5],
    "3d-membrane-curvature-reconstruction": [fix_a6],
    "data-driven-disease-dynamics": [fix_a7],
    "boltz2esi-enzyme-substrate-ranking": [fix_a8],
    "eola1-acc-salvage-lcms-sensitivity": [fix_a10],
    "hic-depth-requirements": [fix_a11],
    "biographx-rna-localization": [fix_b1],
    "kmt2d-crc-immune-deconvolution": [fix_d1],
}

ALL_TASKS = [
    "granpa-node-phenotype-association", "human-sound-recognition-modeling",
    "hyperspherical-microbiome-clustering", "lrt-microbiome-sampling-methods",
    "massgat-lcms-annotation", "mcsa-cyclic-peptide-motif",
    "phylofm-lineage-transport", "prompt-guided-multiplex-virtual-staining",
    "star-suite-flex-probe-assignment", "cpplocpred-subcellular-localization",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", default=str(TASKS_REL))
    a = ap.parse_args()
    root = Path(a.task_dir)

    applied: list[str] = []
    for task, fns in FIXERS.items():
        for fn in fns:
            fn(task, root, applied)
    # class C applies to every task: it is a no-op when the payload is already
    # numeric, and a safety net when the grader crashes before writing
    # class E is likewise generic (a DataFrame column shadowing an attribute)
    # class A9 is generic too: any solve.sh using /app/solution is wrong
    # class F is generic: instrument silent grade-crash -> score 0 swallows
    # class G is generic: swap sample-major/term-major transposed indexing
    for task in sorted({d.name for d in root.iterdir()
                        if (d / "tests").is_dir()}):
        fix_class_c(task, root, applied)
        fix_class_e(task, root, applied)
        fix_a9(task, root, applied)
        fix_class_f(task, root, applied)
        fix_class_g(task, root, applied)
        fix_b2(task, root, applied)
        fix_a12(task, root, applied)
        fix_a13(task, root, applied)
        fix_a14(task, root, applied)
        fix_a15(task, root, applied)
        fix_a16(task, root, applied)
        fix_a17(task, root, applied)
        fix_a18(task, root, applied)
        fix_a19(task, root, applied)
        fix_a20(task, root, applied)
        fix_a21(task, root, applied)
        fix_a22(task, root, applied)
        fix_a23(task, root, applied)
        fix_a24(task, root, applied)
        fix_a25(task, root, applied)
        fix_a26(task, root, applied)
        fix_a27(task, root, applied)
        fix_a28(task, root, applied)
        fix_a29(task, root, applied)
        fix_a30(task, root, applied)
        _fix_scalar_slice(task, root, applied)
        fix_a31(task, root, applied)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps({"applied": applied, "n": len(applied)}, indent=2,
                   ensure_ascii=False), encoding="utf-8")
    lines = ["# Task repairs applied (2026-10-12)", ""]
    lines += [f"- {x}" for x in applied]
    (OUT / "applied.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"applied {len(applied)} fixes:")
    for x in applied:
        print("  -", x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
