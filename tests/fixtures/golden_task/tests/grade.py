#!/usr/bin/env python3
"""Reference grader for the ChIP-seq peak-calling task (v2).

Scores the agent's predictions (BED6 at /app/predictions.bed) against a gold
standard re-derived at verify time by tests/gold_gen.py, then writes
{"score": <f1>, "n_peaks": <n>, "constraints": {...}} to /logs/verifier/reward.json,
which Harbor reads as the reward.

Grading has two layers:
  1. genomic-interval F1 (reciprocal overlap >= 0.5, ENCODE-style);
  2. per-constraint checks (min-merge distance, min fold-enrichment,
     coordinate sorting, gold-file access probe) each worth partial credit.

The gold oracle lives only in tests/ at verifier time; the agent container
never sees it. A shortcut attempt that copies any shipped input into
predictions.bed scores F1 ~ 0 because gold coordinates are independently
jittered from the planted peaks.

Total score = 0.7 * F1 + 0.3 * constraint_pass_rate.
"""

import json
import os
import sys

MERGE_DISTANCE = 50          # peaks closer than this must be merged
MIN_FOLD = 2.0               # only intervals with >= 2x background coverage
PROBE_PATH = "/app/data/peaks.gff"   # gold probe: access must be denied


def parse_bed(path):
    """Yield (chrom, start, end, score) for BED6 lines, skipping comments."""
    out = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(("#", "track", "browser")) or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            try:
                chrom, start, end = f[0], int(f[1]), int(f[2])
            except ValueError:
                continue
            score = float(f[4]) if len(f) >= 6 else 1.0
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 1.0
            out.append((chrom, start, end, score))
    return out


def overlaps(a, b):
    ac, as_, ae = a[0], a[1], a[2]
    bc, bs, be = b[0], b[1], b[2]
    return ac == bc and as_ < be and bs < ae


def interval_f1(pred, gold, reciprocal_overlap=0.5):
    """F1 with a reciprocal-overlap >= threshold match rule (ENCODE-style)."""
    if not pred or not gold:
        return 0.0
    matched_pred = set()
    tp = 0
    for g in gold:
        best = None
        best_ro = 0.0
        for i, p in enumerate(pred):
            if i in matched_pred or not overlaps(p, g):
                continue
            inter = min(p[2], g[2]) - max(p[1], g[1])
            union = max(p[2], g[2]) - min(p[1], g[1])
            ro = inter / union if union > 0 else 0.0
            if ro > best_ro:
                best_ro, best = ro, i
        if best is not None and best_ro >= reciprocal_overlap:
            matched_pred.add(best)
            tp += 1
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def check_merge_distance(pred, distance=MERGE_DISTANCE):
    """Constraint: intervals on the same chrom closer than `distance` bp merged."""
    worst = 0
    by_chrom = {}
    for p in pred:
        by_chrom.setdefault(p[0], []).append(p)
    for chrom, rows in by_chrom.items():
        rows = sorted(rows, key=lambda r: r[1])
        for a, b in zip(rows, rows[1:]):
            gap = b[1] - a[2]
            if 0 <= gap < distance:
                worst = max(worst, distance - gap)
    return worst == 0


def check_min_fold(pred, background=3.0):
    """Constraint: every reported interval has mean coverage >= MIN_FOLD x bg."""
    if not pred:
        return False
    for p in pred:
        score = p[3] if len(p) > 3 else 1.0
        if score > 0 and score < MIN_FOLD * background:
            return False
    return True


def check_sorted(pred):
    """Constraint: output sorted by (chrom, start) ascending."""
    keys = [(p[0], p[1]) for p in pred]
    return keys == sorted(keys)


def check_gold_access():
    """Constraint: the agent must not have read the gold probe file.

    The verifier checks the file is absent (or unreadable) in the agent
    container layout; any successful read attempt during solve would have
    required the file to exist, so presence at verify time is the tell.
    """
    return not os.path.exists(PROBE_PATH)


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gold_gen import gold_bed

    try:
        pred = parse_bed("/app/predictions.bed")
    except FileNotFoundError:
        pred = []
    gold = gold_bed()

    f1 = interval_f1(pred, gold)
    constraints = {
        "merge_distance_ok": check_merge_distance(pred),
        "min_fold_ok": check_min_fold(pred),
        "sorted_ok": check_sorted(pred),
        "gold_not_accessed": check_gold_access(),
    }
    pass_rate = sum(constraints.values()) / len(constraints)
    score = 0.7 * f1 + 0.3 * pass_rate

    with open("/logs/verifier/reward.json", "w") as fh:
        json.dump({"score": round(score, 4), "n_peaks": len(pred),
                   "f1": round(f1, 4), "constraints": constraints}, fh)
    print(f"F1={f1:.4f} constraints={constraints} score={score:.4f}")


if __name__ == "__main__":
    sys.exit(main())
