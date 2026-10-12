#!/usr/bin/env python3
"""Grader for the PG-LLM sequence-only variant-effect baseline task.

1. regenerate a pristine copy of the benchmark inputs from the protected seed;
2. recompute the reference variant scores with an independent implementation of
   the published scoring rule;
3. compare /app/output/predictions.csv and /app/output/report.json with it;
4. score six independent constraints and write their mean as the reward.
"""
import csv
import json
import math
import os
import subprocess
import sys

GOLD_DIR = "/tmp/gold_inputs"
AGENT_DIR = "/app/output"
REWARD_PATH = "/logs/verifier/reward.json"
TOL = 1e-6
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)
EXPECTED_HEADER = ["assay_id", "variant_id", "predicted_score"]


# ---------------------------------------------------------------------------
# low level helpers
# ---------------------------------------------------------------------------
def read_fasta(path):
    records = []
    name = None
    buf = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(buf)))
                name = line[1:]
                buf = []
            else:
                buf.append(line)
    if name is not None:
        records.append((name, "".join(buf)))
    return records


def blosum_lookup():
    from Bio.Align import substitution_matrices

    mat = substitution_matrices.load("BLOSUM62")

    def score(a, b):
        try:
            return int(mat[a, b])
        except Exception:
            pass
        alphabet = mat.alphabet
        return int(mat[alphabet.index(a)][alphabet.index(b)])

    return score


def spearman(x, y):
    from scipy import stats

    if len(x) < 3:
        return 0.0
    rho = float(stats.spearmanr(x, y)[0])
    if not math.isfinite(rho):
        return 0.0
    return rho


def load_report():
    path = os.path.join(AGENT_DIR, "report.json")
    if not os.path.exists(path):
        return {}, "report.json missing"
    try:
        with open(path) as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return {}, "report.json is not an object"
        return data, None
    except Exception as exc:  # noqa: BLE001
        return {}, str(exc)


# ---------------------------------------------------------------------------
# reference implementation of MSA-CONS-BLOSUM
# ---------------------------------------------------------------------------
def reference_scores(gold_dir):
    blosum = blosum_lookup()
    manifest = list(csv.DictReader(open(os.path.join(gold_dir, "assay_manifest.csv"))))
    ref = {}
    variants_by_assay = {}
    split_of = {}

    for row in manifest:
        assay = row["assay_id"]
        adir = os.path.join(gold_dir, "assays", assay)
        wt = read_fasta(os.path.join(adir, "wt.fasta"))[0][1]
        msa = [seq for _, seq in read_fasta(os.path.join(adir, "msa.fasta"))]
        length = len(wt)

        counts = [dict() for _ in range(length)]
        for seq in msa:
            if len(seq) != length:
                raise ValueError("unaligned MSA record for %s" % assay)
            for p, ch in enumerate(seq):
                if ch in AA_SET:
                    counts[p][ch] = counts[p].get(ch, 0) + 1

        freq = []
        for p in range(length):
            total = sum(counts[p].values())
            freq.append({a: (counts[p].get(a, 0) + 1.0) / (total + 20.0) for a in AA})

        rows = list(csv.DictReader(open(os.path.join(adir, "variants.csv"))))
        variants_by_assay[assay] = rows
        split_of[assay] = row["split"]

        scored = {}
        for variant in rows:
            vid = variant["variant_id"]
            muts = variant["mutations"]
            if muts == "WT":
                scored[vid] = 0.0
                continue
            total = 0.0
            for single in muts.split(":"):
                w = single[0]
                p = int(single[1:-1]) - 1
                a = single[-1]
                if wt[p] != w:
                    raise ValueError("WT residue mismatch: %s %s" % (assay, single))
                total += math.log2(freq[p][a] / freq[p][w])
                total += 0.10 * (blosum(w, a) - blosum(w, w))
            scored[vid] = total
        ref[assay] = scored

    return ref, variants_by_assay, split_of


def gold_top5(ref, variants_by_assay, assay):
    order = sorted(ref[assay].items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in order[:5]]


# ---------------------------------------------------------------------------
# agent output
# ---------------------------------------------------------------------------
def load_agent_predictions():
    path = os.path.join(AGENT_DIR, "predictions.csv")
    preds = {}
    header_ok = False
    if not os.path.exists(path):
        return preds, header_ok, "predictions.csv missing"
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return preds, header_ok, "predictions.csv empty"
        header = [h.strip() for h in header]
        header_ok = header == EXPECTED_HEADER
        for row in reader:
            if len(row) < 3:
                continue
            key = (row[0].strip(), row[1].strip())
            try:
                value = float(row[2].strip())
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            preds.setdefault(key, []).append(value)
    return preds, header_ok, None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(REWARD_PATH), exist_ok=True)

    if not os.path.exists(os.path.join(GOLD_DIR, "assay_manifest.csv")):
        subprocess.run([sys.executable, "/tests/gold_gen.py", GOLD_DIR], check=True)

    ref, variants_by_assay, split_of = reference_scores(GOLD_DIR)
    preds, header_ok, load_err = load_agent_predictions()
    report, report_err = load_report()
    reported = report.get("per_assay", {}) if isinstance(report, dict) else {}
    if not isinstance(reported, dict):
        reported = {}

    expected = [(a, r["variant_id"]) for a in ref for r in variants_by_assay[a]]
    checks = {}

    # ---- constraint 1: format, coverage, uniqueness ----------------------
    good = 0
    duplicates = 0
    for key in expected:
        vals = preds.get(key)
        if vals is None:
            continue
        if len(vals) > 1:
            duplicates += 1
            continue
        good += 1
    coverage = good / float(len(expected)) if expected else 0.0
    factor = 1.0
    if not header_ok:
        factor *= 0.5
    if duplicates:
        factor *= 0.5
    checks["c1_output_format_and_coverage"] = round(coverage * factor, 6)

    # ---- constraint 2: score fidelity ------------------------------------
    matched = 0
    for key in expected:
        vals = preds.get(key)
        if not vals or len(vals) != 1:
            continue
        if abs(vals[0] - ref[key[0]][key[1]]) <= TOL:
            matched += 1
    checks["c2_score_fidelity"] = round(matched / float(len(expected)), 6) if expected else 0.0

    # ---- constraint 3: wild-type controls --------------------------------
    wt_keys = [(a, r["variant_id"]) for a in ref for r in variants_by_assay[a]
               if r["mutations"] == "WT"]
    wt_ok = 0
    for key in wt_keys:
        vals = preds.get(key)
        if vals and len(vals) == 1 and abs(vals[0]) <= 1e-9:
            wt_ok += 1
    checks["c3_wt_control_zero"] = round(wt_ok / float(len(wt_keys)), 6) if wt_keys else 0.0

    # ---- constraint 4: top-5 prioritisation ------------------------------
    assays = list(ref.keys())
    top5_ok = 0
    complete_predictions = {}
    for assay in assays:
        values = {}
        complete = True
        for row in variants_by_assay[assay]:
            vals = preds.get((assay, row["variant_id"]))
            if not vals or len(vals) != 1:
                complete = False
                break
            values[row["variant_id"]] = vals[0]
        if not complete:
            continue
        complete_predictions[assay] = values
        order = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))
        if [k for k, _ in order[:5]] == gold_top5(ref, variants_by_assay, assay):
            top5_ok += 1
    checks["c4_top5_prioritisation"] = round(top5_ok / float(len(assays)), 6) if assays else 0.0

    # ---- constraint 5: per-assay Spearman --------------------------------
    recomputed_rho = {}
    rho_ok = 0
    for assay in assays:
        values = complete_predictions.get(assay)
        if values is None:
            continue
        pred_vec = []
        meas_vec = []
        for row in variants_by_assay[assay]:
            pred_vec.append(values[row["variant_id"]])
            meas_vec.append(float(row["dms_score"]))
        rho = spearman(pred_vec, meas_vec)
        recomputed_rho[assay] = rho
        entry = reported.get(assay)
        if not isinstance(entry, dict):
            continue
        try:
            reported_rho = float(entry.get("spearman_rho"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(reported_rho) and abs(reported_rho - rho) <= 1e-4:
            rho_ok += 1
    checks["c5_per_assay_spearman"] = round(rho_ok / float(len(assays)), 6) if assays else 0.0

    # ---- constraint 6: split aggregation ---------------------------------
    def effective_rho(assay):
        if assay in recomputed_rho:
            return recomputed_rho[assay]
        entry = reported.get(assay)
        if isinstance(entry, dict):
            try:
                value = float(entry.get("spearman_rho"))
                if math.isfinite(value):
                    return value
            except (TypeError, ValueError):
                pass
        return None

    subs = []
    for split in ("proteinGym", "heldout"):
        members = [a for a in assays if split_of[a] == split]
        values = [effective_rho(a) for a in members]
        if not members or any(v is None for v in values):
            subs.append(0.0)
            continue
        mean = sum(values) / len(values)
        try:
            got = float(report.get("split_mean_rho", {})[split])
        except (KeyError, TypeError, ValueError):
            subs.append(0.0)
            continue
        subs.append(1.0 if abs(got - mean) <= 1e-4 else 0.0)

    try:
        counts = report.get("n_assays", {})
        counts_ok = all(
            int(counts[split]) == len([a for a in assays if split_of[a] == split])
            for split in ("proteinGym", "heldout")
        )
    except (KeyError, TypeError, ValueError):
        counts_ok = False
    subs.append(1.0 if counts_ok else 0.0)

    try:
        nvar_ok = all(
            isinstance(reported.get(a), dict)
            and int(reported[a].get("n_variants")) == len(variants_by_assay[a])
            for a in assays
        )
    except (TypeError, ValueError):
        nvar_ok = False
    subs.append(1.0 if nvar_ok else 0.0)

    checks["c6_split_aggregation"] = round(sum(subs) / len(subs), 6)

    score = sum(checks.values()) / float(len(checks))
    score = max(0.0, min(1.0, score))

    with open(REWARD_PATH, "w") as fh:
        json.dump({"score": round(score, 6), "reward": round(score, 6)}, fh)

    print(json.dumps({
        "checks": checks,
        "score": round(score, 6),
        "n_expected_rows": len(expected),
        "n_assays": len(assays),
        "predictions_error": load_err,
        "report_error": report_err,
    }, indent=2))


if __name__ == "__main__":
    main()
