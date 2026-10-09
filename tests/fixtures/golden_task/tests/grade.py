#!/usr/bin/env python3
"""Reference grader for the ChIP-seq peak-calling task.

Scores the agent's predictions (BED6 at /app/predictions.bed) against a hidden
gold standard (/app/data/peaks.gff) by genomic-interval F1, then writes
{"score": <f1>} to /app/rewards.json which Harbor reads as the reward.

The gold standard lives outside the agent's reachable paths at grade time;
this file runs in the verifier phase.
"""

import json
import sys


def parse_bed(path):
    """Yield (chrom, start, end) for BED6 lines, skipping comments/tracks."""
    out = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(("#", "track", "browser")) or not line.strip():
                continue
            f = line.split("\t")
            if len(f) < 3:
                continue
            try:
                out.append((f[0], int(f[1]), int(f[2])))
            except ValueError:
                continue
    return out


def overlaps(a, b):
    ac, as_, ae = a
    bc, bs, be = b
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


def main():
    try:
        pred = parse_bed("/app/predictions.bed")
    except FileNotFoundError:
        pred = []
    gold = parse_bed("/app/data/peaks.bed")  # gold shipped as BED for the grader
    f1 = interval_f1(pred, gold)
    with open("/app/rewards.json", "w") as fh:
        json.dump({"score": round(f1, 4)}, fh)
    print(f"F1={f1:.4f}")


if __name__ == "__main__":
    sys.exit(main())
