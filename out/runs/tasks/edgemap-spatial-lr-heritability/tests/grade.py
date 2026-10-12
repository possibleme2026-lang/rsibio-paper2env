#!/usr/bin/env python3
"""Grader for the EdgeMap spatial-annotation task.

Regenerates all inputs from the protected seed, recomputes the complete
reference pipeline, then scores the agent's two result tables with partial
credit per constraint.  Only numbers are written to reward.json.
"""
import json
import os
import sys
import tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gold_gen  # noqa: E402

OUT = "/app/output"
LOGS = "/logs/verifier"
WEIGHTS = {"format": 0.10, "edge_z": 0.30, "conditional_z": 0.15,
           "tissue_pattern": 0.15, "gene_z": 0.15, "gene_rank": 0.15}
TOL = 0.05


def read_tsv(path):
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]


def emit(score, comp, detail):
    os.makedirs(LOGS, exist_ok=True)
    score = float(max(0.0, min(1.0, score)))
    with open(os.path.join(LOGS, "reward.json"), "w") as fh:
        json.dump({"score": round(score, 4), "reward": round(score, 4)}, fh)
    with open(os.path.join(LOGS, "report.json"), "w") as fh:
        json.dump({"components": comp, "detail": detail}, fh, indent=2, default=str)
    print("component breakdown: " + json.dumps(comp))
    print("detail: " + json.dumps(detail, default=str))
    print("FINAL SCORE %.4f" % score)


def main():
    comp = {k: 0.0 for k in WEIGHTS}
    detail = {}
    try:
        with tempfile.TemporaryDirectory() as td:
            gold_gen.main(td)
            gold = gold_gen.compute_gold(td)
    except Exception as exc:  # pragma: no cover
        detail["gold_error"] = repr(exc)
        emit(0.0, comp, detail)
        return

    tset = gold_gen.TISSUES
    traits = [t for t, _ in gold_gen.TRAITS]

    edge_rows, gene_rows = None, None
    try:
        edge_rows = read_tsv(os.path.join(OUT, "edge_z_scores.tsv"))
        cols = ["trait", "tissue", "edge_tau", "intrinsic_tau",
                "edge_z", "intrinsic_z", "delta_z"]
        keys = [(r["trait"], r["tissue"]) for r in edge_rows]
        want = [(t, ti) for t in traits for ti in tset]
        ok = (len(edge_rows) == len(want) and set(keys) == set(want)
              and keys == sorted(keys)
              and all(c in edge_rows[0] for c in cols)
              and all(np.isfinite(float(r[c])) for r in edge_rows for c in cols[2:]))
        comp["format"] += 0.5 if ok else 0.0
        detail["edge_format_ok"] = bool(ok)
    except Exception as exc:
        detail["edge_file_error"] = repr(exc)
        edge_rows = None

    try:
        gene_rows = read_tsv(os.path.join(OUT, "prioritized_genes.tsv"))
        cols = ["gene", "edge_score", "z_g", "prioritized"]
        ok = (len(gene_rows) == len(gold["gene"])
              and all(c in gene_rows[0] for c in cols)
              and set(r["gene"] for r in gene_rows) == set(gold["gene"])
              and all(r["prioritized"] in ("0", "1") for r in gene_rows)
              and all(np.isfinite(float(r["z_g"])) for r in gene_rows))
        comp["format"] += 0.5 if ok else 0.0
        detail["gene_format_ok"] = bool(ok)
    except Exception as exc:
        detail["gene_file_error"] = repr(exc)
        gene_rows = None

    if edge_rows:
        errs = []
        for r in edge_rows:
            k = (r["trait"], r["tissue"])
            if k in gold["edge"]:
                errs.append(abs(float(r["edge_z"]) - gold["edge"][k]["edge_z"]))
        comp["edge_z"] = sum(1 for e in errs if e <= TOL) / max(1, len(gold["edge"]))
        detail["edge_z_max_abs_err"] = max(errs) if errs else None

        errs = []
        for r in edge_rows:
            k = (r["trait"], r["tissue"])
            if k in gold["edge"]:
                g = gold["edge"][k]
                e1 = abs(float(r["intrinsic_z"]) - g["intrinsic_z"])
                e2 = abs(float(r["delta_z"]) - g["delta_z"])
                errs.append(max(e1, e2))
        comp["conditional_z"] = sum(1 for e in errs if e <= TOL) / max(1, len(gold["edge"]))
        detail["conditional_max_abs_err"] = max(errs) if errs else None

        cur = {(r["trait"], r["tissue"]): float(r["edge_z"]) for r in edge_rows}
        hit = 0
        pattern_agent = {}
        for tr, mt in gold_gen.TRAITS:
            others = [t for t in tset if t != mt and (tr, t) in cur]
            if (tr, mt) not in cur or not others:
                continue
            a = cur[(tr, mt)] > float(np.mean([cur[(tr, t)] for t in others]))
            pattern_agent[tr] = bool(a)
            hit += int(bool(a) == bool(gold["pattern"][tr]))
        comp["tissue_pattern"] = hit / len(gold_gen.TRAITS)
        detail["pattern_agent"] = pattern_agent
        detail["pattern_gold"] = gold["pattern"]

    if gene_rows:
        gmap = {r["gene"]: r for r in gene_rows}
        ok_z = 0
        for g, gv in gold["gene"].items():
            r = gmap.get(g)
            if r is None:
                continue
            if (abs(float(r["z_g"]) - gv["z_g"]) <= TOL
                    and abs(float(r["edge_score"]) - gv["edge_score"]) <= 1e-4):
                ok_z += 1
        comp["gene_z"] = ok_z / max(1, len(gold["gene"]))
        detail["gene_rows_matching"] = ok_z

        pred = set(g for g, r in gmap.items() if r["prioritized"] == "1")
        truth = set(gold["top10"])
        tp = len(pred & truth)
        prec = tp / len(pred) if pred else 0.0
        rec = tp / len(truth) if truth else 0.0
        comp["gene_rank"] = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        detail["n_prioritized_agent"] = len(pred)
        detail["n_prioritized_gold"] = len(truth)
        detail["n_true_positives"] = tp

    detail["gold_top10"] = gold["top10"]
    score = sum(WEIGHTS[k] * comp[k] for k in WEIGHTS)
    emit(score, comp, detail)


if __name__ == "__main__":
    main()
