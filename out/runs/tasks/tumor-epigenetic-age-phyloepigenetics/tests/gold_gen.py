#!/usr/bin/env python3
"""Protected reference implementation: recomputes the gold answers from /app/data."""
from __future__ import annotations

import csv
import json
import os

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist

DATA = "/app/data"
OUT = "/tmp/gold.json"
MISSING = {"", "NA", "NaN", "nan", "None"}


def read_beta():
    with open(os.path.join(DATA, "beta_matrix.tsv")) as fh:
        rdr = csv.reader(fh, delimiter="\t")
        header = next(rdr)
        probes = header[1:]
        samples, rows = [], []
        for row in rdr:
            samples.append(row[0])
            rows.append([np.nan if v in MISSING else float(v) for v in row[1:]])
    return samples, probes, np.array(rows, dtype=float)


def main():
    samples, probes, B = read_beta()
    pidx = {p: i for i, p in enumerate(probes)}
    sidx = {s: i for i, s in enumerate(samples)}
    with open(os.path.join(DATA, "sample_metadata.csv")) as fh:
        meta = {r["sample_id"]: r for r in csv.DictReader(fh)}
    with open(os.path.join(DATA, "probe_annotation.csv")) as fh:
        ann = {r["probe_id"]: r for r in csv.DictReader(fh)}
    with open(os.path.join(DATA, "clock_coefficients.csv")) as fh:
        clock = list(csv.DictReader(fh))
    with open(os.path.join(DATA, "clock_model.json")) as fh:
        model = json.load(fh)

    Bimp = np.where(np.isnan(B), np.nanmean(B, axis=0)[None, :], B)

    # ---- Part 1: epigenetic clock ----
    xs = np.full(len(samples), float(model["intercept"]))
    for r in clock:
        v = Bimp[:, pidx[r["probe_id"]]]
        if r["transform"] == "one_minus_beta":
            v = 1.0 - v
        xs = xs + float(r["coefficient"]) * v
    age = np.where(xs >= 0.0, 21.0 * np.exp(xs) - 1.0, 20.0 + 20.0 * xs)

    chrono = np.array([float(meta[s]["chronological_age_years"]) for s in samples])
    is_norm = np.array([meta[s]["tissue"] == "normal" for s in samples])
    normal_mae = float(np.mean(np.abs(age[is_norm] - chrono[is_norm])))

    normal_of = {meta[s]["patient_id"]: s for s in samples if meta[s]["tissue"] == "normal"}
    cts = sorted({meta[s]["cancer_type"] for s in samples})

    raw_p, deltas = [], {}
    for ct in cts:
        d = []
        for s in samples:
            if meta[s]["cancer_type"] == ct and meta[s]["tissue"] == "tumor":
                d.append(age[sidx[s]] - age[sidx[normal_of[meta[s]["patient_id"]]]])
        d = np.array(d)
        deltas[ct] = d
        raw_p.append(float(stats.wilcoxon(d, alternative="two-sided")[1]))
    raw_p = np.array(raw_p)
    m = raw_p.size
    order = np.argsort(raw_p)
    ranked = raw_p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qvals = np.empty(m)
    qvals[order] = np.clip(ranked, 0.0, 1.0)

    clock_out = {"normal_mae_years": round(normal_mae, 3),
                 "n_normal_samples": int(is_norm.sum()),
                 "per_cancer_type": {}}
    for i, ct in enumerate(cts):
        med = float(np.median(deltas[ct]))
        q = float(qvals[i])
        if q < 0.05 and med > 0:
            call = "accelerated"
        elif q < 0.05 and med < 0:
            call = "decelerated"
        else:
            call = "ns"
        clock_out["per_cancer_type"][ct] = {
            "n_pairs": int(deltas[ct].size),
            "median_delta_years": round(med, 3),
            "wilcoxon_p": float(raw_p[i]),
            "bh_q": q,
            "call": call,
        }

    # ---- Part 2: phyloepigenetics ----
    miss_frac = np.isnan(B).mean(axis=0)
    keep = np.where(miss_frac <= 0.02)[0]
    chrom = np.array([ann[probes[j]]["chromosome"] for j in range(len(probes))])
    chrx_keep = chrom[keep] == "chrX"

    phylo_ct = {}
    tree_nodes = {}
    tot_pairs = tot_chrx_pairs = ev_pairs = ev_chrx = 0

    for ct in cts:
        trows = [sidx[s] for s in samples
                 if meta[s]["cancer_type"] == ct and meta[s]["tissue"] == "tumor"]
        n_tum = len(trows)
        sub = Bimp[np.ix_(trows, keep)]
        dd = pdist(sub, metric="correlation")
        Z = linkage(dd, method="average")
        coph = float(cophenet(Z, dd)[0])

        counts = np.zeros(keep.size, dtype=int)
        for r in trows:
            nrm = sidx[normal_of[meta[samples[r]]["patient_id"]]]
            e = np.abs(Bimp[r, keep] - Bimp[nrm, keep]) >= 0.30
            counts += e.astype(int)
            ev_pairs += int(e.sum())
            ev_chrx += int(e[chrx_keep].sum())
            tot_pairs += int(keep.size)
            tot_chrx_pairs += int(chrx_keep.sum())

        n_ev = int(np.sum(counts >= 1))
        n_tr = int(np.sum(counts >= n_tum))
        n_pr = int(np.sum(counts == 1))
        phylo_ct[ct] = {
            "n_tumor_samples": n_tum,
            "n_event_probes": n_ev,
            "n_trunk_probes": n_tr,
            "n_private_probes": n_pr,
            "n_shared_probes": n_ev - n_tr - n_pr,
            "private_fraction": round(n_pr / n_ev, 4) if n_ev else 0.0,
            "cophenetic_correlation": round(coph, 6),
        }

        nodes = {i: frozenset([samples[r]]) for i, r in enumerate(trows)}
        heights = {}
        for k in range(Z.shape[0]):
            a, b = int(Z[k, 0]), int(Z[k, 1])
            s = nodes[a] | nodes[b]
            nodes[n_tum + k] = s
            heights[s] = float(Z[k, 2])
        tree_nodes[ct] = [[sorted(list(s)), h] for s, h in heights.items()]

    p_x = float(stats.hypergeom.sf(ev_chrx - 1, tot_pairs, tot_chrx_pairs, ev_pairs))
    x_out = {
        "n_event_occurrences": ev_pairs,
        "n_chrx_event_occurrences": ev_chrx,
        "event_chrx_fraction": round(ev_chrx / ev_pairs, 4) if ev_pairs else 0.0,
        "background_chrx_fraction": round(tot_chrx_pairs / tot_pairs, 4),
        "hypergeom_p": p_x,
    }

    gold = {"clock": clock_out,
            "phylo": {"per_cancer_type": phylo_ct, "x_enrichment": x_out},
            "trees": tree_nodes}
    with open(OUT, "w") as fh:
        json.dump(gold, fh)
    print("gold written:", OUT)


if __name__ == "__main__":
    main()
