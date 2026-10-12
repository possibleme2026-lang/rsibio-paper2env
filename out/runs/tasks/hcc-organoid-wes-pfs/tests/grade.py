#!/usr/bin/env python3
"""Deep grader for the HCC organoid cohort benchmark task.

The grader never trusts the files inside the container: it rebuilds the whole
dataset from the protected seed in tests/gold_gen.py, recomputes the reference
analysis, verifies that /app/data was not tampered with, and then scores every
constraint of the task independently with weighted partial credit.
"""
from __future__ import annotations

import json
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import gold_gen as gg

OUT_DIR = "/app/output"
DATA_DIR = "/app/data"
VERIFIER_DIR = "/logs/verifier"

TAB = chr(9)
NL = chr(10)
CR = chr(13)

PAIR_HEADER = ["patient_id", "shared", "tumor_only", "organoid_only",
               "jaccard", "vaf_spearman", "median_abs_vaf_diff", "fidelity_pass"]
SHARED_HEADER = ["patient_id", "chrom", "pos", "ref", "alt", "gene",
                 "tumor_vaf", "organoid_vaf"]

TRUE_TOKENS = ("true", "1", "yes")
FALSE_TOKENS = ("false", "0", "no")


def _as_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _as_int(value):
    try:
        return int(float(value))
    except Exception:
        return None


def _near(value, target, rel=0.0, absol=0.0):
    got = _as_float(value)
    if got is None or target is None:
        return False
    if absol and abs(got - target) <= absol:
        return True
    if rel and target != 0 and abs(got - target) / abs(target) <= rel:
        return True
    return False


def _read_tsv(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()
    except Exception:
        return None
    lines = [ln for ln in raw.replace(CR, "").split(NL) if ln.strip() != ""]
    if not lines:
        return None
    header = [h.strip() for h in lines[0].split(TAB)]
    rows = []
    for line in lines[1:]:
        rows.append(dict(zip(header, line.split(TAB))))
    return header, rows


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _finish(score, report):
    os.makedirs(VERIFIER_DIR, exist_ok=True)
    score = max(0.0, min(1.0, float(score)))
    with open(os.path.join(VERIFIER_DIR, "reward.json"), "w", encoding="utf-8") as handle:
        json.dump({"score": round(score, 4), "reward": round(score, 4)}, handle)
    with open(os.path.join(VERIFIER_DIR, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


def main():
    checks = []

    def add(name, weight, fraction, detail):
        fraction = max(0.0, min(1.0, float(fraction)))
        checks.append({"name": name, "weight": weight,
                       "fraction": round(fraction, 4), "detail": detail})

    files = gg.build_dataset(gg.SEED)
    gold = gg.compute_gold(files)

    mismatch = []
    for rel, content in files.items():
        path = os.path.join(DATA_DIR, rel)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                disk = handle.read()
        except Exception:
            mismatch.append(rel)
            continue
        if disk != content:
            mismatch.append(rel)
    if mismatch:
        _finish(0.0, {"error": "input integrity violation under /app/data",
                      "files": sorted(mismatch)})
        return

    summary = _load_json(os.path.join(OUT_DIR, "summary.json"))
    if not isinstance(summary, dict):
        summary = {}

    subs = [
        _as_int(summary.get("n_patients")) == gold["n_patients"],
        _as_int(summary.get("n_organoid_established")) == gold["n_established"],
        _near(summary.get("establishment_rate_pct"), gold["establishment_rate_pct"], absol=0.15),
    ]
    add("cohort_summary", 1.0, sum(subs) / 3.0,
        {"subchecks": subs, "gold_n_patients": gold["n_patients"],
         "gold_n_established": gold["n_established"],
         "gold_rate_pct": gold["establishment_rate_pct"]})

    morph = summary.get("morphology_counts")
    if not isinstance(morph, dict):
        morph = {}
    subs = [_as_int(morph.get("cystic")) == gold["morphology_counts"]["cystic"],
            _as_int(morph.get("solid")) == gold["morphology_counts"]["solid"]]
    add("morphology_counts", 1.0, sum(subs) / 2.0,
        {"subchecks": subs, "gold": gold["morphology_counts"]})

    counts = summary.get("filtered_variant_counts")
    if not isinstance(counts, dict):
        counts = {}
    keys = sorted(gold["filtered_variant_counts"])
    ok = sum(1 for k in keys if _as_int(counts.get(k)) == gold["filtered_variant_counts"][k])
    add("variant_filtering", 2.0, ok / float(len(keys)),
        {"n_correct": ok, "n_total": len(keys),
         "gold": gold["filtered_variant_counts"],
         "reported": {k: counts.get(k) for k in keys}})

    pair_tbl = _read_tsv(os.path.join(OUT_DIR, "pair_concordance.tsv"))
    by_patient = {}
    pair_header_ok = False
    pair_row_count = 0
    if pair_tbl:
        header, pair_rows = pair_tbl
        pair_row_count = len(pair_rows)
        pair_header_ok = [h.lower() for h in header] == [h.lower() for h in PAIR_HEADER]
        for row in pair_rows:
            pid = (row.get("patient_id") or "").strip()
            if pid:
                by_patient[pid] = row

    gold_pairs = gold["pairs"]
    pids = sorted(gold_pairs)

    ok = 0
    per_pair = {}
    for pid in pids:
        expected = gold_pairs[pid]
        row = by_patient.get(pid)
        good = False
        if row is not None:
            good = (_as_int(row.get("shared")) == expected["shared"]
                    and _as_int(row.get("tumor_only")) == expected["tumor_only"]
                    and _as_int(row.get("organoid_only")) == expected["organoid_only"]
                    and _near(row.get("jaccard"), expected["jaccard"], absol=0.005))
        per_pair[pid] = good
        ok += 1 if good else 0
    add("pair_intersections", 2.0, ok / float(len(pids)),
        {"per_pair": per_pair, "rows_found": pair_row_count,
         "header_ok": pair_header_ok,
         "gold": {p: {"shared": gold_pairs[p]["shared"],
                      "tumor_only": gold_pairs[p]["tumor_only"],
                      "organoid_only": gold_pairs[p]["organoid_only"],
                      "jaccard": round(gold_pairs[p]["jaccard"], 6)} for p in pids}})

    ok = 0
    per_pair = {}
    for pid in pids:
        expected = gold_pairs[pid]
        row = by_patient.get(pid)
        good = False
        if row is not None:
            good = (_near(row.get("vaf_spearman"), expected["vaf_spearman"], absol=0.02)
                    and _near(row.get("median_abs_vaf_diff"),
                              expected["median_abs_vaf_diff"], absol=0.005))
        per_pair[pid] = good
        ok += 1 if good else 0
    add("vaf_concordance", 1.5, ok / float(len(pids)),
        {"per_pair": per_pair,
         "gold": {p: {"vaf_spearman": round(gold_pairs[p]["vaf_spearman"], 6),
                      "median_abs_vaf_diff": round(gold_pairs[p]["median_abs_vaf_diff"], 6)}
                  for p in pids}})

    ok = 0
    per_pair = {}
    for pid in pids:
        want_true = bool(gold_pairs[pid]["fidelity_pass"])
        row = by_patient.get(pid)
        token = ""
        if row is not None:
            token = str(row.get("fidelity_pass", "")).strip().lower()
        good = token in TRUE_TOKENS if want_true else token in FALSE_TOKENS
        per_pair[pid] = good
        ok += 1 if good else 0
    n_pass_ok = _as_int(summary.get("n_fidelity_pass")) == gold["n_fidelity_pass"]
    add("fidelity_classification", 1.0, (ok + (1 if n_pass_ok else 0)) / float(len(pids) + 1),
        {"per_pair": per_pair, "n_fidelity_pass_ok": n_pass_ok,
         "gold_n_fidelity_pass": gold["n_fidelity_pass"],
         "gold_pass": {p: gold_pairs[p]["fidelity_pass"] for p in pids}})

    hr = _as_float(summary.get("cox_hr_solid_vs_cystic"))
    good = hr is not None and hr > 1.0 and _near(hr, gold["cox_hr"], rel=0.20)
    add("cox_hazard_ratio", 1.5, 1.0 if good else 0.0,
        {"reported": hr, "gold": gold["cox_hr"]})

    pval = _as_float(summary.get("logrank_p"))
    good = (pval is not None and 0.0 < pval < 0.05
            and _near(pval, gold["logrank_p"], rel=0.25))
    add("logrank_p_value", 1.5, 1.0 if good else 0.0,
        {"reported": pval, "gold": gold["logrank_p"]})

    medians = summary.get("median_pfs_months")
    if not isinstance(medians, dict):
        medians = {}
    subs = [_near(medians.get("cystic"), gold["median_pfs"]["cystic"], absol=0.3),
            _near(medians.get("solid"), gold["median_pfs"]["solid"], absol=0.3)]
    add("kaplan_meier_medians", 1.0, sum(subs) / 2.0,
        {"subchecks": subs, "gold": gold["median_pfs"]})

    pvalues = summary.get("establishment_p_values")
    if not isinstance(pvalues, dict):
        pvalues = {}
    factors = ["age_gt70", "hbv", "bcat_positive"]
    subs = [_near(pvalues.get(f), gold["fisher_p"][f], rel=0.15, absol=0.003)
            for f in factors]
    add("establishment_associations", 1.0, sum(subs) / 3.0,
        {"subchecks": dict(zip(factors, subs)), "gold": gold["fisher_p"]})

    shared_tbl = _read_tsv(os.path.join(OUT_DIR, "shared_variants.tsv"))
    subs = [False, False, False, False]
    stats_detail = {"rows_found": 0, "gold_rows": gold["shared_loci_count"]}
    if shared_tbl:
        header, rows = shared_tbl
        stats_detail["rows_found"] = len(rows)
        subs[0] = [h.lower() for h in header] == [h.lower() for h in SHARED_HEADER]
        subs[1] = len(rows) == gold["shared_loci_count"]
        got = set()
        ordered = True
        prev = None
        for row in rows:
            pid = (row.get("patient_id") or "").strip()
            chrom = (row.get("chrom") or "").strip()
            pos = _as_int(row.get("pos"))
            ref = (row.get("ref") or "").strip()
            alt = (row.get("alt") or "").strip()
            if pos is None:
                ordered = False
                continue
            key = (pid, chrom, pos, ref, alt)
            got.add(key)
            cur = (pid, gg.CHROM_INDEX.get(chrom, 99), pos)
            if prev is not None and cur < prev:
                ordered = False
            prev = cur
        subs[2] = got == gold["shared_loci_set"]
        subs[3] = ordered and len(rows) > 0
        stats_detail["loci_matched"] = len(got & gold["shared_loci_set"])
    add("shared_variant_catalogue", 1.5, sum(subs) / 4.0,
        dict(stats_detail, subchecks=subs))

    total_weight = sum(c["weight"] for c in checks)
    score = sum(c["weight"] * c["fraction"] for c in checks) / total_weight

    report = {
        "score": round(score, 4),
        "total_weight": total_weight,
        "checks": checks,
        "gold_overview": {
            "n_patients": gold["n_patients"],
            "n_established": gold["n_established"],
            "morphology_counts": gold["morphology_counts"],
            "n_fidelity_pass": gold["n_fidelity_pass"],
            "cox_hr": gold["cox_hr"],
            "logrank_p": gold["logrank_p"],
            "median_pfs": gold["median_pfs"],
            "fisher_p": gold["fisher_p"],
            "mean_jaccard": gold["mean_jaccard"],
            "mean_vaf_spearman": gold["mean_vaf_spearman"],
        },
    }
    _finish(score, report)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # never let the verifier crash without a reward file
        import traceback
        _finish(0.0, {"error": str(exc), "traceback": traceback.format_exc()})
