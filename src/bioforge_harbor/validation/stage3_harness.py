"""Stage 3 runner: harness optimization, no weight updates (paper 4.3).

Compare harness variants (prompt/tool/memory/control-flow configurations)
across the same (model, task) grid. The key numeric question the paper poses:
does *model* variance exceed *harness* variance (HarnessOpt-Bench's finding),
or do domain toolchains flip the answer?

Inputs are MatrixReport JSON files: {"cells": {"<model>||<harness>": mean_reward}}
produced by any rollout system (Stage 2 logs, Harbor jobs, or hand-written).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .metrics import variance_decomposition, harness_delta


def load_matrix(path: Path | str) -> dict[tuple[str, str], float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = data.get("cells", data)
    out: dict[tuple[str, str], float] = {}
    for k, v in cells.items():
        if "||" not in k:
            raise ValueError(f"cell key must be 'model||harness', got {k!r}")
        m, h = k.split("||", 1)
        out[(m, h)] = float(v)
    return out


def dump_matrix(cells: dict[tuple[str, str], float]) -> str:
    return json.dumps({"cells": {f"{m}||{h}": v for (m, h), v in sorted(cells.items())}},
                      indent=2) + "\n"


def compare(baseline_cells: dict[tuple[str, str], float],
            variant_cells: dict[tuple[str, str], float]) -> dict[str, Any]:
    """Variance decomposition on each side + per-cell delta table."""
    base_vd = variance_decomposition(baseline_cells)
    var_vd = variance_decomposition(variant_cells)
    # per-model average across harnesses, then delta variant-vs-baseline
    from statistics import mean

    def _per_model(cells: dict[tuple[str, str], float]) -> dict[str, float]:
        acc: dict[str, list[float]] = {}
        for (m, _h), v in cells.items():
            acc.setdefault(m, []).append(v)
        return {m: mean(vs) for m, vs in acc.items()}

    delta = harness_delta(_per_model(baseline_cells), _per_model(variant_cells))
    return {
        "stage": 3,
        "baseline_variance": base_vd,
        "variant_variance": var_vd,
        "per_model_delta": delta,
        "claim_check": {
            "model_dominant_baseline": base_vd.get("dominant") == "model",
            "model_dominant_variant": var_vd.get("dominant") == "model",
        },
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="bioforge compare-harness",
                                 description="Stage 3: harness optimization comparison (paper 4.3)")
    ap.add_argument("--baseline", required=True, help="MatrixReport JSON of baseline harness")
    ap.add_argument("--variant", required=True, help="MatrixReport JSON of variant harness")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rep = compare(load_matrix(a.baseline), load_matrix(a.variant))
    text = json.dumps(rep, indent=2, ensure_ascii=False) + "\n"
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
