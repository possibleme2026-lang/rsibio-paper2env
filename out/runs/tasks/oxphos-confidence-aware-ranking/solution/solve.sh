#!/bin/bash
set -euo pipefail
mkdir -p /app/results
cat > /app/pipeline.py <<'PY'
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA = '/app/data'
OUT = '/app/results'
COMPLEXES = ['Complex I', 'Complex II', 'Complex III', 'Complex IV', 'Complex V']

bulk = pd.read_csv(os.path.join(DATA, 'expression', 'bulk_timecourse.tsv'), sep='\t', index_col=0)
genes = list(bulk.index)
gidx = {g: i for i, g in enumerate(genes)}
Z = bulk.to_numpy(dtype=float)
Z = (Z - Z.mean(axis=1, keepdims=True)) / (Z.std(axis=1, keepdims=True) + 1e-9)

hc = pd.read_csv(os.path.join(DATA, 'annotations', 'high_confidence_oxphos.tsv'), sep='\t')
lc = pd.read_csv(os.path.join(DATA, 'annotations', 'low_confidence_oxphos.tsv'), sep='\t')
neg = pd.read_csv(os.path.join(DATA, 'annotations', 'other_process_annotations.tsv'), sep='\t')
uni = pd.read_csv(os.path.join(DATA, 'candidates', 'gene_universe.tsv'), sep='\t')

cand_ids = [g for g in uni['gene_id'] if g in gidx]
hc_ids = list(hc['gene_id'])
lc_ids = list(lc['gene_id'])
neg_ids = list(neg['gene_id'])
cand_i = np.array([gidx[g] for g in cand_ids])
hc_i = np.array([gidx[g] for g in hc_ids])
lc_i = np.array([gidx[g] for g in lc_ids])
neg_i = np.array([gidx[g] for g in neg_ids])
hc_cx = list(hc['complex'])
lc_cx = list(lc['complex'])


def fit_model(pos_idx, neg_idx, pos_w=None):
    X = np.vstack([Z[pos_idx], Z[neg_idx]])
    y = np.r_[np.ones(len(pos_idx)), np.zeros(len(neg_idx))]
    w = np.ones(len(y))
    if pos_w is not None:
        w[:len(pos_idx)] = pos_w
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.5, max_iter=5000)
    clf.fit(sc.transform(X), y, sample_weight=w)
    return clf, sc


def predict(model, idx):
    clf, sc = model
    return clf.predict_proba(sc.transform(Z[idx]))[:, 1]


def auroc(scores, labels):
    labels = np.asarray(labels).astype(bool)
    r = rankdata(np.asarray(scores, dtype=float))
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ---- round 1: high-confidence annotations only
m1 = fit_model(hc_i, neg_i)
p1 = predict(m1, lc_i)
thr = float(np.quantile(p1, 0.7))
keep = p1 >= thr
if int(keep.sum()) < 5:
    keep = np.zeros_like(p1, dtype=bool)
    keep[np.argsort(-p1)[:5]] = True
pseudo_i = lc_i[keep]
pseudo_cx = [c for c, k in zip(lc_cx, keep) if k]

# ---- round 2: informed bagging over the five complexes
bag = np.zeros((len(COMPLEXES), len(cand_i)))
per_module = {}
for k, cx in enumerate(COMPLEXES):
    tr_pos = np.array([gidx[g] for g, c in zip(hc_ids, hc_cx) if c != cx])
    tr_ps = np.array([gidx[g] for g, c in zip(lc_ids, pseudo_cx) if c != cx])
    if tr_ps.size == 0:
        tr_ps = np.array([], dtype=int)
    held = np.array([gidx[g] for g, c in zip(hc_ids, hc_cx) if c == cx])
    pos = np.concatenate([tr_pos, tr_ps]) if tr_ps.size else tr_pos
    wpos = np.r_[np.ones(len(tr_pos)), 0.5 * np.ones(len(tr_ps))] if tr_ps.size else None
    model = fit_model(pos, neg_i, wpos)
    bag[k] = predict(model, cand_i)
    held_scores = predict(model, np.concatenate([held, neg_i]))
    held_lab = np.r_[np.ones(len(held)), np.zeros(len(neg_i))]
    per_module[cx] = auroc(held_scores, held_lab)

score_sup = bag.mean(axis=0)
n_support = (bag > 0.5).sum(axis=0).astype(int)


def atlas_corr(path):
    d = np.load(path, allow_pickle=True)
    ids = [str(x) for x in d['genes']]
    col = np.array([gidx[g] for g in ids])
    X = d['X'].astype(np.float64)
    Y = np.empty_like(X)
    Y[:, col] = X
    Y = Y - Y.mean(axis=0, keepdims=True)
    Y = Y / (Y.std(axis=0) + 1e-9)
    C = (Y.T @ Y) / Y.shape[0]
    return C[:, hc_i].mean(axis=1)


co_all = 0.5 * (atlas_corr(os.path.join(DATA, 'expression', 'sc_embryonic.npz')) +
                atlas_corr(os.path.join(DATA, 'expression', 'sc_adult.npz')))
co = co_all[cand_i]

final = 0.5 * rankdata(score_sup) / len(score_sup) + 0.5 * rankdata(co) / len(co)
order = np.argsort(-final, kind='mergesort')

lines = ['gene_id\tscore_supervised\tscore_coexpression\tn_modules_supporting\tfinal_score\trank']
for r, i in enumerate(order, start=1):
    lines.append('%s\t%.6f\t%.6f\t%d\t%.6f\t%d' % (
        cand_ids[i], score_sup[i], co[i], int(n_support[i]), final[i], r))
with open(os.path.join(OUT, 'candidate_scores.tsv'), 'w') as fh:
    fh.write('\n'.join(lines) + '\n')

short = ['gene_id\trank\tfinal_score']
for r, i in enumerate(order[:25], start=1):
    short.append('%s\t%d\t%.6f' % (cand_ids[i], r, final[i]))
with open(os.path.join(OUT, 'shortlist.tsv'), 'w') as fh:
    fh.write('\n'.join(short) + '\n')

report = {
    'seed': 0,
    'n_round1_positives': int(len(hc_i)),
    'n_round2_added': int(len(pseudo_i)),
    'n_module_models': len(COMPLEXES),
    'per_module_auroc': {k: float(v) for k, v in per_module.items()},
    'overall_cv_auroc': float(np.nanmean(list(per_module.values()))),
    'n_coexp_reference_genes': int(len(hc_i)),
    'method_notes': 'round-1 logistic model on high-confidence positives; round-2 consensus pseudo-labels; leave-one-complex-out bagging; mean Pearson co-expression against the high-confidence set in both single-cell atlases'
}
with open(os.path.join(OUT, 'report.json'), 'w') as fh:
    json.dump(report, fh, indent=2)
print('wrote', len(cand_ids), 'candidate scores')
PY
python /app/pipeline.py
