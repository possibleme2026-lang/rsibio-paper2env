#!/bin/bash
set -euo pipefail
mkdir -p /app/output/trees
cat > /app/solve_task.py <<'PYEOF'
import csv, json, os
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist

D, O = "/app/data", "/app/output"
os.makedirs(O + "/trees", exist_ok=True)

with open(D + "/beta_matrix.tsv") as fh:
    rdr = csv.reader(fh, delimiter="\t")
    hdr = next(rdr)
    probes = hdr[1:]
    samples, rows = [], []
    for row in rdr:
        samples.append(row[0])
        rows.append([np.nan if v in ("NA", "", "nan") else float(v) for v in row[1:]])
B = np.array(rows, float)
pidx = {p: i for i, p in enumerate(probes)}
sidx = {s: i for i, s in enumerate(samples)}
meta = {r["sample_id"]: r for r in csv.DictReader(open(D + "/sample_metadata.csv"))}
ann = {r["probe_id"]: r for r in csv.DictReader(open(D + "/probe_annotation.csv"))}
clock = list(csv.DictReader(open(D + "/clock_coefficients.csv")))
model = json.load(open(D + "/clock_model.json"))

Bimp = np.where(np.isnan(B), np.nanmean(B, axis=0)[None, :], B)

# ---- Part 1 -------------------------------------------------------------
xs = np.full(len(samples), float(model["intercept"]))
for r in clock:
    v = Bimp[:, pidx[r["probe_id"]]]
    if r["transform"] == "one_minus_beta":
        v = 1.0 - v
    xs = xs + float(r["coefficient"]) * v
age = np.where(xs >= 0.0, 21.0 * np.exp(xs) - 1.0, 20.0 + 20.0 * xs)

chrono = np.array([float(meta[s]["chronological_age_years"]) for s in samples])
isn = np.array([meta[s]["tissue"] == "normal" for s in samples])
normal_mae = float(np.mean(np.abs(age[isn] - chrono[isn])))
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
             "n_normal_samples": int(isn.sum()),
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
        "n_pairs": int(deltas[ct].size), "median_delta_years": round(med, 3),
        "wilcoxon_p": float(raw_p[i]), "bh_q": q, "call": call}

# ---- Part 2 -------------------------------------------------------------
miss_frac = np.isnan(B).mean(axis=0)
keep = np.where(miss_frac <= 0.02)[0]
chrom = np.array([ann[probes[j]]["chromosome"] for j in range(len(probes))])
chrx_keep = chrom[keep] == "chrX"

def fmt(v):
    return "%.6f" % v

def newick(n, Z, labels):
    nodes = {i: (labels[i], 0.0) for i in range(n)}
    kids = {}
    for k in range(Z.shape[0]):
        kids[n + k] = (int(Z[k, 0]), int(Z[k, 1]))
        nodes[n + k] = ("", float(Z[k, 2]))
    def rec(idx, ph, root=False):
        nm, h = nodes[idx]
        if idx < n:
            return nm + ":" + fmt(ph - h)
        a, b = kids[idx]
        body = "(" + rec(a, h) + "," + rec(b, h) + ")"
        return body if root else body + ":" + fmt(ph - h)
    r = 2 * n - 2
    return rec(r, nodes[r][1], True) + ";"

phylo_ct = {}
tot_pairs = tot_chrx = ev_pairs = ev_chrx = 0
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
        tot_chrx += int(chrx_keep.sum())
    n_ev = int(np.sum(counts >= 1))
    n_tr = int(np.sum(counts >= n_tum))
    n_pr = int(np.sum(counts == 1))
    phylo_ct[ct] = {"n_tumor_samples": n_tum, "n_event_probes": n_ev,
                    "n_trunk_probes": n_tr, "n_private_probes": n_pr,
                    "n_shared_probes": n_ev - n_tr - n_pr,
                    "private_fraction": round(n_pr / n_ev, 4) if n_ev else 0.0,
                    "cophenetic_correlation": round(coph, 6)}
    with open(O + "/trees/" + ct + ".nwk", "w") as fh:
        fh.write(newick(n_tum, Z, [samples[r] for r in trows]) + "\n")

p_x = float(stats.hypergeom.sf(ev_chrx - 1, tot_pairs, tot_chrx, ev_pairs))
x_out = {"n_event_occurrences": ev_pairs, "n_chrx_event_occurrences": ev_chrx,
         "event_chrx_fraction": round(ev_chrx / ev_pairs, 4) if ev_pairs else 0.0,
         "background_chrx_fraction": round(tot_chrx / tot_pairs, 4),
         "hypergeom_p": p_x}

with open(O + "/results.json", "w") as fh:
    json.dump({"clock": clock_out,
               "phylo": {"per_cancer_type": phylo_ct, "x_enrichment": x_out}}, fh, indent=2)
print("done")
PYEOF
python /app/solve_task.py
