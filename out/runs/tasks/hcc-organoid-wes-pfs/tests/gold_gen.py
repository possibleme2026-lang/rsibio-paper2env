#!/usr/bin/env python3
"""Hidden data generator + reference analysis for the HCC organoid task.

The generator block below is byte-identical to environment/data/gen_inputs.py so
that the grader can rebuild the exact same inputs from the protected seed.  The
analysis block is the ground truth used by tests/grade.py; it never ships inside
the agent image.
"""
from __future__ import annotations

import math
import random

import numpy as np
from scipy import stats

SEED = 20260802
TAB = chr(9)
NL = chr(10)

CHROMS = [str(i) for i in range(1, 23)] + ["X"]
CHROM_INDEX = {c: i for i, c in enumerate(CHROMS)}

GENES = [
    "TP53", "CTNNB1", "TERT", "AXIN1", "ARID1A", "ARID2", "ALB", "APOB",
    "KEAP1", "NFE2L2", "TSC1", "TSC2", "PTEN", "PIK3CA", "RB1", "ACVR2A",
    "BAP1", "KMT2D", "SETD2", "NF1", "CDKN2A", "MYC", "CCND1", "FAT4",
    "SMARCA4", "JAK1", "IL6ST", "GNAS", "RPS6KA3", "ABCA13", "MUC16",
    "RYR2", "OBSCN", "SACS", "SYNE1", "CSMD3", "LRP1B", "PCLO", "DNAH5",
]

CONSEQUENCES = [
    "missense_variant", "stop_gained", "synonymous_variant",
    "frameshift_variant", "splice_region_variant", "splice_acceptor_variant",
    "inframe_deletion", "5_prime_UTR_variant", "intron_variant",
    "non_coding_transcript_exon_variant",
]

VARIANT_COLUMNS = ["chrom", "pos", "ref", "alt", "gene", "consequence",
                   "depth", "alt_depth", "vaf", "qual"]

CLINICAL_COLUMNS = [
    "patient_id", "age", "sex", "etiology", "tumor_size_cm", "differentiation",
    "bcat_positive", "afp_ng_ml", "organoid_established", "organoid_morphology",
    "pfs_months", "pfs_event",
]

CLINICAL_ROWS = [
    ("HCC-01", 72, "M", "HBV", 5.4, "moderate", 1, 320.5, 1, "cystic", 8.1, 1),
    ("HCC-02", 58, "M", "HCV", 4.2, "moderate", 0, 45.2, 0, "NA", 33.4, 1),
    ("HCC-03", 76, "F", "HBV", 6.8, "poor", 1, 980.0, 1, "cystic", 13.4, 1),
    ("HCC-04", 64, "M", "alcohol", 7.5, "poor", 0, 210.3, 1, "solid", 4.2, 1),
    ("HCC-05", 61, "F", "HBV", 3.9, "moderate", 1, 150.7, 0, "NA", 26.7, 0),
    ("HCC-06", 55, "M", "NBNC", 5.1, "well", 0, 12.4, 0, "NA", 41.2, 0),
    ("HCC-07", 73, "M", "HBV", 8.2, "moderate", 1, 760.5, 1, "cystic", 22.5, 1),
    ("HCC-08", 68, "F", "NBNC", 4.6, "moderate", 1, 55.9, 1, "cystic", 30.2, 1),
    ("HCC-09", 59, "M", "HCV", 6.3, "poor", 0, 88.1, 0, "NA", 51.8, 0),
    ("HCC-10", 71, "M", "HBV", 9.1, "poor", 0, 1200.0, 1, "cystic", 38.9, 1),
    ("HCC-11", 63, "F", "alcohol", 3.4, "moderate", 0, 30.6, 0, "NA", 18.9, 1),
    ("HCC-12", 66, "M", "HBV", 7.8, "poor", 1, 430.2, 1, "solid", 9.7, 1),
    ("HCC-13", 52, "M", "NBNC", 5.7, "moderate", 0, 22.8, 0, "NA", 47.5, 1),
    ("HCC-14", 78, "F", "HBV", 6.1, "moderate", 1, 615.4, 1, "cystic", 57.3, 0),
    ("HCC-15", 60, "M", "HCV", 4.8, "well", 1, 95.3, 0, "NA", 36.1, 1),
    ("HCC-16", 72, "F", "HBV", 8.6, "poor", 0, 505.2, 0, "NA", 21.4, 1),
    ("HCC-17", 69, "M", "HBV", 10.2, "poor", 0, 1540.0, 1, "solid", 16.3, 1),
    ("HCC-18", 65, "M", "alcohol", 5.9, "moderate", 0, 78.6, 0, "NA", 29.3, 0),
    ("HCC-19", 74, "F", "HCV", 4.4, "moderate", 1, 260.1, 1, "cystic", 44.1, 0),
    ("HCC-20", 62, "M", "HBV", 6.7, "moderate", 0, 340.8, 0, "NA", 12.7, 1),
    ("HCC-21", 77, "M", "HBV", 7.2, "poor", 1, 890.3, 1, "cystic", 50.7, 0),
    ("HCC-22", 54, "F", "NBNC", 3.7, "well", 0, 8.9, 0, "NA", 39.6, 0),
    ("HCC-23", 70, "M", "HBV", 9.4, "poor", 0, 720.9, 1, "solid", 24.8, 1),
    ("HCC-24", 56, "M", "HCV", 5.3, "moderate", 0, 61.7, 0, "NA", 16.2, 1),
    ("HCC-25", 76, "F", "alcohol", 4.1, "moderate", 1, 110.5, 0, "NA", 43.8, 0),
    ("HCC-26", 75, "M", "HBV", 11.0, "poor", 1, 2100.0, 1, "solid", 35.6, 0),
    ("HCC-27", 51, "M", "NBNC", 6.9, "moderate", 0, 18.3, 0, "NA", 25.1, 1),
]

PAIR_SPECS = [
    {"id": "HCC-01", "core": 320, "share": 0.92, "t_priv": 22, "o_priv": 20, "sd": 0.040, "n_germ": 16},
    {"id": "HCC-04", "core": 300, "share": 0.90, "t_priv": 26, "o_priv": 24, "sd": 0.045, "n_germ": 20},
    {"id": "HCC-08", "core": 280, "share": 0.93, "t_priv": 18, "o_priv": 22, "sd": 0.035, "n_germ": 12},
    {"id": "HCC-12", "core": 340, "share": 0.89, "t_priv": 30, "o_priv": 28, "sd": 0.050, "n_germ": 22},
    {"id": "HCC-17", "core": 360, "share": 0.62, "t_priv": 70, "o_priv": 85, "sd": 0.130, "n_germ": 18},
    {"id": "HCC-23", "core": 330, "share": 0.58, "t_priv": 80, "o_priv": 90, "sd": 0.150, "n_germ": 14},
]


def _locus(rng, used):
    while True:
        chrom = rng.choice(CHROMS)
        pos = rng.randint(200000, 240000000)
        ref = rng.choice("ACGT")
        alt = rng.choice([b for b in "ACGT" if b != ref])
        key = (chrom, pos, ref, alt)
        if key not in used:
            used.add(key)
            return key


def _reflect(value, lo=0.115, hi=0.85):
    while value < lo or value > hi:
        if value < lo:
            value = lo + (lo - value)
        else:
            value = hi - (value - hi)
    return value


def _fail_mode(rng, p=0.07):
    u = rng.random()
    if u < p / 2.0:
        return "low_vaf"
    if u < p:
        return "low_depth"
    return None


def _make_row(rng, locus, gene, cons, vaf_hint=None, fail=None):
    chrom, pos, ref, alt = locus
    if fail == "low_vaf":
        vaf = round(rng.uniform(0.02, 0.08), 4)
        depth = rng.randint(30, 180)
    elif fail == "low_depth":
        vaf = round(rng.uniform(0.12, 0.70), 4)
        depth = rng.randint(8, 18)
    else:
        if vaf_hint is None:
            vaf = round(rng.uniform(0.12, 0.72), 4)
        else:
            vaf = round(_reflect(vaf_hint), 4)
        depth = rng.randint(30, 180)
    alt_depth = max(1, int(round(vaf * depth)))
    vaf = round(alt_depth / float(depth), 4)
    qual = round(rng.uniform(60.0, 220.0), 1)
    return {"chrom": chrom, "pos": pos, "ref": ref, "alt": alt, "gene": gene,
            "consequence": cons, "depth": depth, "alt_depth": alt_depth,
            "vaf": vaf, "qual": qual}


def _build_snps(rng, n=600):
    used = set()
    snps = []
    while len(snps) < n:
        snps.append(_locus(rng, used))
    return snps


def _build_pair(rng, spec, snp_pool):
    used = set()
    t_rows = []
    o_rows = []

    germ = rng.sample(snp_pool, spec["n_germ"])
    for loc in germ:
        used.add(loc)
        gene = rng.choice(GENES)
        cons = rng.choice(CONSEQUENCES)
        gv = rng.uniform(0.42, 0.58)
        t_rows.append(_make_row(rng, loc, gene, cons, vaf_hint=gv))
        o_rows.append(_make_row(rng, loc, gene, cons, vaf_hint=gv))

    for _ in range(spec["core"]):
        loc = _locus(rng, used)
        gene = rng.choice(GENES)
        cons = rng.choice(CONSEQUENCES)
        base = rng.uniform(0.14, 0.68)
        t_rows.append(_make_row(rng, loc, gene, cons, vaf_hint=base, fail=_fail_mode(rng)))
        if rng.random() < spec["share"]:
            shifted = base + rng.normalvariate(0.0, spec["sd"])
            o_rows.append(_make_row(rng, loc, gene, cons, vaf_hint=shifted, fail=_fail_mode(rng)))

    for _ in range(spec["t_priv"]):
        loc = _locus(rng, used)
        gene = rng.choice(GENES)
        cons = rng.choice(CONSEQUENCES)
        t_rows.append(_make_row(rng, loc, gene, cons, fail=_fail_mode(rng)))

    for _ in range(spec["o_priv"]):
        loc = _locus(rng, used)
        gene = rng.choice(GENES)
        cons = rng.choice(CONSEQUENCES)
        o_rows.append(_make_row(rng, loc, gene, cons, fail=_fail_mode(rng)))

    n_sub_t = int(round(0.05 * (spec["core"] + spec["t_priv"])))
    n_sub_o = int(round(0.05 * (spec["core"] + spec["o_priv"])))
    for _ in range(n_sub_t):
        loc = _locus(rng, used)
        gene = rng.choice(GENES)
        cons = rng.choice(CONSEQUENCES)
        mode = "low_vaf" if rng.random() < 0.5 else "low_depth"
        t_rows.append(_make_row(rng, loc, gene, cons, fail=mode))
    for _ in range(n_sub_o):
        loc = _locus(rng, used)
        gene = rng.choice(GENES)
        cons = rng.choice(CONSEQUENCES)
        mode = "low_vaf" if rng.random() < 0.5 else "low_depth"
        o_rows.append(_make_row(rng, loc, gene, cons, fail=mode))

    return t_rows, o_rows


def _render_clinical():
    lines = [TAB.join(CLINICAL_COLUMNS)]
    for row in CLINICAL_ROWS:
        lines.append(TAB.join(str(x) for x in row))
    return NL.join(lines) + NL


def _render_snps(snps):
    lines = [TAB.join(["chrom", "pos", "ref", "alt", "rsid"])]
    ordered = sorted(snps, key=lambda s: (CHROM_INDEX[s[0]], s[1]))
    for i, (chrom, pos, ref, alt) in enumerate(ordered, start=1):
        lines.append(TAB.join([chrom, str(pos), ref, alt, "rs%d" % (1000000 + i)]))
    return NL.join(lines) + NL


def _render_variants(rows):
    ordered = sorted(rows, key=lambda r: (CHROM_INDEX[r["chrom"]], r["pos"]))
    lines = [TAB.join(VARIANT_COLUMNS)]
    for r in ordered:
        lines.append(TAB.join([r["chrom"], str(r["pos"]), r["ref"], r["alt"],
                               r["gene"], r["consequence"], str(r["depth"]),
                               str(r["alt_depth"]), "%.4f" % r["vaf"],
                               "%.1f" % r["qual"]]))
    return NL.join(lines) + NL


def build_dataset(seed=SEED):
    rng = random.Random(seed)
    files = {}
    files["clinical.tsv"] = _render_clinical()
    snps = _build_snps(rng)
    files["common_snps.tsv"] = _render_snps(snps)
    for spec in PAIR_SPECS:
        t_rows, o_rows = _build_pair(rng, spec, snps)
        files["variants/%s_tumor.tsv" % spec["id"]] = _render_variants(t_rows)
        files["variants/%s_organoid.tsv" % spec["id"]] = _render_variants(o_rows)
    return files


def parse_clinical(text):
    lines = [ln for ln in text.strip(NL).split(NL) if ln.strip()]
    header = lines[0].split(TAB)
    return [dict(zip(header, ln.split(TAB))) for ln in lines[1:]]


def parse_variants(text):
    lines = [ln for ln in text.strip(NL).split(NL) if ln.strip()]
    header = lines[0].split(TAB)
    out = []
    for ln in lines[1:]:
        rec = dict(zip(header, ln.split(TAB)))
        out.append({"chrom": rec["chrom"], "pos": int(rec["pos"]),
                    "ref": rec["ref"], "alt": rec["alt"],
                    "gene": rec["gene"], "consequence": rec["consequence"],
                    "depth": int(rec["depth"]), "alt_depth": int(rec["alt_depth"]),
                    "vaf": float(rec["vaf"]), "qual": float(rec["qual"])})
    return out


def parse_snp_keys(text):
    keys = set()
    lines = [ln for ln in text.strip(NL).split(NL) if ln.strip()]
    for ln in lines[1:]:
        f = ln.split(TAB)
        if len(f) >= 4:
            keys.add((f[0], int(f[1]), f[2], f[3]))
    return keys


def filter_rows(rows, snp_keys):
    kept = []
    for r in rows:
        key = (r["chrom"], r["pos"], r["ref"], r["alt"])
        if key in snp_keys:
            continue
        if r["depth"] < 20:
            continue
        if r["vaf"] < 0.10:
            continue
        kept.append(r)
    return kept


def _fisher(clinical, predicate):
    a = sum(1 for r in clinical if int(r["organoid_established"]) == 1 and predicate(r))
    b = sum(1 for r in clinical if int(r["organoid_established"]) == 1 and not predicate(r))
    c = sum(1 for r in clinical if int(r["organoid_established"]) == 0 and predicate(r))
    d = sum(1 for r in clinical if int(r["organoid_established"]) == 0 and not predicate(r))
    _, pvalue = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return float(pvalue)


def _cox_nr(times, events, x):
    beta = 0.0
    info = 0.0
    for _ in range(200):
        grad = 0.0
        info = 0.0
        for i in range(len(times)):
            if not events[i]:
                continue
            s0 = s1 = s2 = 0.0
            for j in range(len(times)):
                if times[j] >= times[i]:
                    w = math.exp(beta * x[j])
                    s0 += w
                    s1 += w * x[j]
                    s2 += w * x[j] * x[j]
            grad += x[i] - s1 / s0
            info += s2 / s0 - (s1 / s0) ** 2
        if info <= 1e-12:
            break
        step = grad / info
        beta += step
        if abs(step) < 1e-10:
            break
    return math.exp(beta)


def _logrank_nr(times, events, groups):
    event_times = sorted(set(times[i] for i in range(len(times)) if events[i]))
    obs = 0.0
    exp = 0.0
    var = 0.0
    for t in event_times:
        n1 = sum(1 for i in range(len(times)) if groups[i] == 1 and times[i] >= t)
        n0 = sum(1 for i in range(len(times)) if groups[i] == 0 and times[i] >= t)
        n = n1 + n0
        d = sum(1 for i in range(len(times)) if events[i] and times[i] == t)
        d1 = sum(1 for i in range(len(times)) if events[i] and times[i] == t and groups[i] == 1)
        obs += d1
        exp += d * n1 / float(n)
        if n > 1:
            var += n1 * n0 * d * (n - d) / float(n * n * (n - 1))
    if var <= 0:
        return 1.0, obs, exp
    chi2 = (obs - exp) ** 2 / var
    return float(stats.chi2.sf(chi2, 1)), obs, exp


def _km_median(times, events, group, groups):
    data = sorted([(times[i], events[i]) for i in range(len(times)) if groups[i] == group])
    if not data:
        return float("inf")
    n = len(data)
    at_risk = n
    surv = 1.0
    i = 0
    while i < n:
        t = data[i][0]
        k = 0
        d = 0
        while i < n and data[i][0] == t:
            k += 1
            d += data[i][1]
            i += 1
        if d > 0:
            surv *= (at_risk - d) / float(at_risk)
            if surv <= 0.5:
                return float(t)
        at_risk -= k
    return float("inf")


def _survival(times, events, groups):
    hr = None
    p_lr = None
    try:
        import pandas as pd
        from lifelines import CoxPHFitter
        from lifelines.statistics import logrank_test
        frame = pd.DataFrame({"T": times, "E": events, "x": groups})
        cph = CoxPHFitter()
        cph.fit(frame, duration_col="T", event_col="E")
        hr = float(cph.hazard_ratios_["x"])
        solid = frame[frame.x == 1]
        cystic = frame[frame.x == 0]
        p_lr = float(logrank_test(solid["T"], cystic["T"], solid["E"], cystic["E"]).p_value)
    except Exception:
        hr = None
        p_lr = None
    if hr is None or p_lr is None or math.isnan(hr) or math.isnan(p_lr):
        hr = _cox_nr(times, events, groups)
        p_lr = _logrank_nr(times, events, groups)[0]
    return hr, p_lr


def compute_gold(files):
    clinical = parse_clinical(files["clinical.tsv"])
    snp_keys = parse_snp_keys(files["common_snps.tsv"])
    gold = {}

    gold["n_patients"] = len(clinical)
    established = [r for r in clinical if int(r["organoid_established"]) == 1]
    gold["n_established"] = len(established)
    gold["establishment_rate_pct"] = round(100.0 * len(established) / len(clinical), 4)

    morph = {"cystic": 0, "solid": 0}
    for rec in established:
        name = rec["organoid_morphology"]
        if name in morph:
            morph[name] += 1
    gold["morphology_counts"] = morph

    gold["fisher_p"] = {
        "age_gt70": _fisher(clinical, lambda r: int(r["age"]) > 70),
        "hbv": _fisher(clinical, lambda r: r["etiology"] == "HBV"),
        "bcat_positive": _fisher(clinical, lambda r: int(r["bcat_positive"]) == 1),
    }

    times = [float(r["pfs_months"]) for r in established]
    events = [int(r["pfs_event"]) for r in established]
    groups = [1 if r["organoid_morphology"] == "solid" else 0 for r in established]
    hr, p_lr = _survival(times, events, groups)
    gold["cox_hr"] = float(hr)
    gold["logrank_p"] = float(p_lr)
    gold["median_pfs"] = {"solid": _km_median(times, events, 1, groups),
                          "cystic": _km_median(times, events, 0, groups)}

    counts = {}
    pair_stats = {}
    shared_loci = []
    for spec in PAIR_SPECS:
        pid = spec["id"]
        t_rows = parse_variants(files["variants/%s_tumor.tsv" % pid])
        o_rows = parse_variants(files["variants/%s_organoid.tsv" % pid])
        t_keep = filter_rows(t_rows, snp_keys)
        o_keep = filter_rows(o_rows, snp_keys)
        counts["%s_tumor" % pid] = len(t_keep)
        counts["%s_organoid" % pid] = len(o_keep)

        t_map = {(r["chrom"], r["pos"], r["ref"], r["alt"]): r for r in t_keep}
        o_map = {(r["chrom"], r["pos"], r["ref"], r["alt"]): r for r in o_keep}
        shared = sorted(set(t_map) & set(o_map),
                        key=lambda k: (CHROM_INDEX.get(k[0], 99), k[1]))
        n_shared = len(shared)
        n_t_only = len(t_map) - n_shared
        n_o_only = len(o_map) - n_shared
        union = len(t_map) + len(o_map) - n_shared
        jaccard = (n_shared / float(union)) if union else 0.0

        tv = [t_map[k]["vaf"] for k in shared]
        ov = [o_map[k]["vaf"] for k in shared]
        if len(shared) >= 3:
            rho = float(stats.spearmanr(tv, ov)[0])
            if math.isnan(rho):
                rho = 0.0
        else:
            rho = 0.0
        mad = float(np.median(np.abs(np.array(tv) - np.array(ov)))) if shared else 0.0

        pair_stats[pid] = {
            "shared": n_shared,
            "tumor_only": n_t_only,
            "organoid_only": n_o_only,
            "jaccard": jaccard,
            "vaf_spearman": rho,
            "median_abs_vaf_diff": mad,
            "fidelity_pass": bool(jaccard >= 0.60 and rho >= 0.70),
        }
        for key in shared:
            shared_loci.append((pid,) + key)

    gold["filtered_variant_counts"] = counts
    gold["pairs"] = pair_stats
    gold["shared_loci"] = sorted(shared_loci,
                                 key=lambda r: (r[0], CHROM_INDEX.get(r[1], 99), r[2]))
    gold["shared_loci_set"] = set(shared_loci)
    gold["shared_loci_count"] = len(shared_loci)
    gold["mean_jaccard"] = sum(p["jaccard"] for p in pair_stats.values()) / len(pair_stats)
    gold["mean_vaf_spearman"] = sum(p["vaf_spearman"] for p in pair_stats.values()) / len(pair_stats)
    gold["n_fidelity_pass"] = sum(1 for p in pair_stats.values() if p["fidelity_pass"])
    return gold


if __name__ == "__main__":
    import json
    print(json.dumps(compute_gold(build_dataset(SEED)), indent=2, default=str))
