#!/usr/bin/env python3
"""Verifier for the DBayesCM-vMF hyperspherical clustering task."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, roc_auc_score

GOLD_PATH = Path('/tmp/gold.npz')
OUT_DIR = Path('/app/output')
REWARD_DIR = Path('/logs/verifier')
N_SAMPLES = 240
N_TAXA = 55
N_HOST = 80

WEIGHTS = {'c1': 0.10, 'c2': 0.25, 'c3': 0.10, 'c4': 0.10,
           'c5': 0.15, 'c6': 0.15, 'c7': 0.15}


def clip01(x):
    return float(min(1.0, max(0.0, x)))


def linear(x, lo, hi):
    if not np.isfinite(x):
        return 0.0
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return clip01((x - lo) / (hi - lo))


def entropy_from_probs(pp):
    pp = np.clip(np.asarray(pp, dtype=float), 1e-12, 1.0)
    return -(pp * np.log(pp)).sum(axis=1)


def write_reward(score, report):
    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    payload = {'score': round(float(score), 4), 'reward': round(float(score), 4),
               'details': report}
    (REWARD_DIR / 'reward.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def main():
    gold = np.load(GOLD_PATH, allow_pickle=True)
    labels_true = np.asarray(gold['labels'])
    boundary = np.asarray(gold['boundary']).astype(bool)
    feat_signal = np.asarray(gold['feature_signal']).astype(bool)
    cooc_mask = np.asarray(gold['cooc_mask']).astype(bool)
    k_true = int(gold['k_true'])
    feat_names = [str(x) for x in gold['feature_names']]
    taxa_names = [str(x) for x in gold['taxa_names']]
    host_names = [str(x) for x in gold['host_names']]
    keep = ~boundary

    report = {}
    try:
        res = json.loads((OUT_DIR / 'results.json').read_text())
        sel = pd.read_csv(OUT_DIR / 'selected_features.csv')
        co = pd.read_csv(OUT_DIR / 'cooccurrence.csv')
    except Exception as exc:
        report['load_error'] = str(exc)
        write_reward(0.0, report)
        return

    labels_vmf = np.asarray(res.get('cluster_labels_vmf', []), dtype=float)
    labels_gmm = np.asarray(res.get('cluster_labels_gmm', []), dtype=float)
    k_vmf = res.get('k_vmf')
    k_gmm = res.get('k_gmm')
    pp = np.asarray(res.get('posterior_probs', []), dtype=float)
    ent = np.asarray(res.get('posterior_entropy', []), dtype=float)

    # ---------------- constraint 1 : schema ----------------
    sub = {}
    sub['k_vmf_present'] = isinstance(k_vmf, int) and k_vmf >= 1
    sub['labels_vmf_len'] = labels_vmf.size == N_SAMPLES
    sub['labels_vmf_integer'] = bool(labels_vmf.size and np.all(labels_vmf == np.round(labels_vmf))
                                     and labels_vmf.min() >= 0)
    sub['k_vmf_matches_unique'] = bool(sub['k_vmf_present'] and sub['labels_vmf_len']
                                       and len(np.unique(labels_vmf)) == int(k_vmf))
    sub['pp_shape'] = bool(pp.ndim == 2 and pp.shape[0] == N_SAMPLES
                           and sub['k_vmf_present'] and pp.shape[1] == int(k_vmf))
    sub['pp_rows_sum'] = bool(sub['pp_shape'] and np.allclose(pp.sum(axis=1), 1.0, atol=1e-6))
    sub['pp_range'] = bool(pp.size and pp.min() >= -1e-9 and pp.max() <= 1 + 1e-9)
    sub['entropy_len'] = ent.size == N_SAMPLES
    sub['entropy_consistent'] = bool(sub['pp_shape'] and sub['entropy_len']
                                     and np.max(np.abs(entropy_from_probs(pp) - ent)) < 1e-3)
    if sub['entropy_len'] and sub['k_vmf_present']:
        sub['entropy_range'] = bool(ent.min() >= -1e-9
                                    and ent.max() <= np.log(max(int(k_vmf), 2)) + 1e-3)
    else:
        sub['entropy_range'] = False
    sub['sel_schema'] = ({'feature_id', 'pip'} <= set(sel.columns) and len(sel) == len(feat_names))
    sub['sel_covers'] = bool(sub['sel_schema']
                             and set(sel['feature_id'].astype(str)) == set(feat_names))
    sub['sel_range'] = bool(sub['sel_schema']
                            and pd.to_numeric(sel['pip'], errors='coerce').between(0, 1).all())
    sub['co_schema'] = {'taxon_id', 'host_feature', 'probability'} <= set(co.columns)
    sub['co_covers'] = bool(sub['co_schema'] and len(co) == N_TAXA * N_HOST)
    sub['co_range'] = bool(sub['co_schema']
                           and pd.to_numeric(co['probability'], errors='coerce').between(0, 1).all())
    c1 = float(np.mean([bool(v) for v in sub.values()]))
    report['c1_schema'] = {'score': round(c1, 4), 'checks': {k: bool(v) for k, v in sub.items()}}

    # ---------------- constraint 2 : community recovery ----------------
    ari_vmf = float('nan')
    if sub['labels_vmf_len']:
        ari_vmf = float(adjusted_rand_score(labels_true[keep], labels_vmf[keep]))
    c2 = linear(ari_vmf, 0.40, 0.85)
    report['c2_ari_vmf'] = {'score': round(c2, 4), 'ari_vmf': None if not np.isfinite(ari_vmf) else round(ari_vmf, 4)}

    # ---------------- constraint 3 : number of communities ----------------
    c3 = 0.0
    if sub['k_vmf_present'] and sub['k_vmf_matches_unique']:
        delta = abs(int(k_vmf) - k_true)
        c3 = 1.0 if delta == 0 else (0.5 if delta == 1 else 0.0)
    report['c3_k'] = {'score': round(c3, 4), 'k_vmf': k_vmf, 'k_true': k_true}

    # ---------------- constraint 4 : geometry contrast ----------------
    gmm_ok = bool(labels_gmm.size == N_SAMPLES and isinstance(k_gmm, int) and k_gmm >= 1
                  and np.all(labels_gmm == np.round(labels_gmm))
                  and len(np.unique(labels_gmm)) == int(k_gmm))
    ari_gmm = float('nan')
    if gmm_ok:
        ari_gmm = float(adjusted_rand_score(labels_true[keep], labels_gmm[keep]))
    c4 = 0.5 if gmm_ok else 0.0
    if gmm_ok and np.isfinite(ari_vmf) and ari_vmf >= ari_gmm - 0.01:
        c4 += 0.5
    report['c4_geometry'] = {'score': round(c4, 4), 'k_gmm': k_gmm,
                             'ari_gmm': None if not np.isfinite(ari_gmm) else round(ari_gmm, 4),
                             'ari_vmf': None if not np.isfinite(ari_vmf) else round(ari_vmf, 4)}

    # ---------------- constraint 5 : feature selection ----------------
    c5 = 0.0
    f1 = 0.0
    if sub['sel_schema'] and sub['sel_range']:
        pip = pd.to_numeric(sel['pip'], errors='coerce')
        chosen = set(sel.loc[pip >= 0.5, 'feature_id'].astype(str))
        truth = set(np.asarray(feat_names)[feat_signal])
        tp = len(chosen & truth)
        fp = len(chosen - truth)
        fn = len(truth - chosen)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        c5 = linear(f1, 0.30, 0.75)
    report['c5_features'] = {'score': round(c5, 4), 'f1': round(f1, 4),
                             'n_true': int(feat_signal.sum())}

    # ---------------- constraint 6 : co-occurrence ----------------
    c6 = 0.0
    auc = float('nan')
    if sub['co_schema']:
        mat = np.zeros((N_TAXA, N_HOST))
        tidx = {n: i for i, n in enumerate(taxa_names)}
        hidx = {n: i for i, n in enumerate(host_names)}
        vals = pd.to_numeric(co['probability'], errors='coerce').fillna(0.0).values
        for t, h, v in zip(co['taxon_id'].astype(str), co['host_feature'].astype(str), vals):
            if t in tidx and h in hidx:
                mat[tidx[t], hidx[h]] = float(v)
        if mat.std() > 0:
            auc = float(roc_auc_score(cooc_mask.ravel().astype(int), mat.ravel()))
            c6 = linear(auc, 0.55, 0.80)
    report['c6_cooccurrence'] = {'score': round(c6, 4),
                                 'auroc': None if not np.isfinite(auc) else round(auc, 4),
                                 'n_positive_pairs': int(cooc_mask.sum())}

    # ---------------- constraint 7 : uncertainty calibration ----------------
    c7 = 0.0
    rec_unc = 0.0
    if sub['entropy_len'] and np.isfinite(ent).all() and ent.std() > 1e-6:
        n_top = int(round(0.30 * N_SAMPLES))
        order = np.argsort(-ent)[:n_top]
        flagged = set(np.flatnonzero(boundary).tolist())
        rec_unc = len(set(order.tolist()) & flagged) / max(len(flagged), 1)
        c7 = linear(rec_unc, 0.0, 0.70)
    report['c7_uncertainty'] = {'score': round(c7, 4), 'recall': round(rec_unc, 4),
                                'n_boundary': int(boundary.sum())}

    score = (WEIGHTS['c1'] * c1 + WEIGHTS['c2'] * c2 + WEIGHTS['c3'] * c3
             + WEIGHTS['c4'] * c4 + WEIGHTS['c5'] * c5 + WEIGHTS['c6'] * c6
             + WEIGHTS['c7'] * c7)
    report['weights'] = WEIGHTS
    report['final_score'] = round(float(score), 4)
    write_reward(score, report)


if __name__ == '__main__':
    main()
