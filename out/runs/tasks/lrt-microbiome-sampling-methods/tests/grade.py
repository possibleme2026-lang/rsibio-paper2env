#!/usr/bin/env python3
"""Grader for bioforge/lrt-microbiome-sampling-methods.

The reference pipeline is re-implemented here from the raw tables in
/app/data (the same tables the agent sees) and each of the seven constraints
stated in the instruction is checked independently.  The score is the
fraction of constraints passed.
"""
import json
import os
import traceback

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

DATA = '/app/data'
OUT = '/app/output'
SEED = 42
N_PERM = 999
INDVAL_MIN = 40.0
P_MAX = 0.05

CHECKS = [
    'decontamination_and_filtered_table',
    'pcoa_pc1_variance',
    'permanova_sample_type',
    'pairwise_permanova',
    'indicator_species',
    'cancer_status',
    'report_schema',
]
results = {}
notes = {}


def record(name, ok, msg):
    results[name] = bool(ok)
    notes[name] = msg
    print(('[PASS] ' if ok else '[FAIL] ') + name + ' :: ' + str(msg), flush=True)


def get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# --------------------------------------------------------------------------
# reference implementation
# --------------------------------------------------------------------------
def bray_curtis(X):
    from scipy.spatial.distance import pdist, squareform
    return squareform(pdist(np.asarray(X, dtype=float), metric='braycurtis'))


def pcoa_pc1(d):
    try:
        from skbio.stats.distance import DistanceMatrix
        from skbio.stats.ordination import pcoa
        return float(pcoa(DistanceMatrix(d)).proportion_explained.iloc[0])
    except Exception:
        A = -0.5 * np.asarray(d, dtype=float) ** 2
        G = A - A.mean(0)[None, :] - A.mean(1)[:, None] + A.mean()
        ev = np.linalg.eigvalsh(G)[::-1]
        ev = ev[ev > 1e-9]
        return float(ev[0] / ev.sum())


def permanova_test(d, groups, permutations=N_PERM):
    groups = np.asarray(groups)
    try:
        from skbio.stats.distance import DistanceMatrix, permanova
        np.random.seed(SEED)
        res = permanova(DistanceMatrix(d), groups, permutations=permutations)
        if isinstance(res, tuple):
            stat, p = res[0], float(res[1])
        else:
            stat, p = res, float(res['p-value'])
        return float(stat['test statistic']), p, int(stat['sample size'])
    except Exception:
        d = np.asarray(d, dtype=float)
        n = d.shape[0]
        labels = np.unique(groups)
        a = len(labels)
        iu = np.triu_indices(n, 1)

        def ssw(lab):
            tot = 0.0
            for g in labels:
                idx = np.where(lab == g)[0]
                if len(idx) > 1:
                    sub = d[np.ix_(idx, idx)]
                    tot += sub[np.triu_indices(len(idx), 1)].sum() / len(idx)
            return tot

        sst = d[iu].sum() / n
        w = ssw(groups)
        F = ((sst - w) / (a - 1)) / (w / (n - a))
        rng = np.random.default_rng(SEED)
        cnt = 0
        for _ in range(permutations):
            wp = ssw(rng.permutation(groups))
            Fp = ((sst - wp) / (a - 1)) / (wp / (n - a))
            if Fp >= F:
                cnt += 1
        return float(F), (cnt + 1) / (permutations + 1), n


def r2_from_f(F, a, n):
    return F * (a - 1) / (F * (a - 1) + (n - a))


def indval(rel, groups, permutations=N_PERM, seed=SEED):
    """rel: samples x ASVs.  Returns (types, IndVal[m,k], p[m,k])."""
    X = np.asarray(rel, dtype=float)
    groups = np.asarray(groups)
    types = sorted(set(groups))
    m = X.shape[1]
    means = np.array([X[groups == t].mean(axis=0) for t in types])
    tot = means.sum(axis=0)
    tot[tot <= 0] = 1.0
    obs = np.zeros((m, len(types)))
    for k, t in enumerate(types):
        b = (X[groups == t] > 0).mean(axis=0)
        obs[:, k] = 100.0 * means[k] / tot * b
    rng = np.random.default_rng(seed)
    ge = np.zeros((m, len(types)))
    for _ in range(permutations):
        perm = rng.permutation(groups)
        pm = np.array([X[perm == t].mean(axis=0) for t in types])
        pt = pm.sum(axis=0)
        pt[pt <= 0] = 1.0
        mx = np.zeros(m)
        for k, t in enumerate(types):
            pb = (X[perm == t] > 0).mean(axis=0)
            np.maximum(mx, 100.0 * pm[k] / pt * pb, out=mx)
        for k in range(len(types)):
            ge[:, k] += mx >= obs[:, k]
    return types, obs, ge / permutations


def build_gold():
    asv = pd.read_csv(os.path.join(DATA, 'asv_table.tsv'), sep='\t', index_col=0)
    meta = pd.read_csv(os.path.join(DATA, 'metadata.tsv'), sep='\t').set_index('sample_id')
    tax = pd.read_csv(os.path.join(DATA, 'taxonomy.tsv'), sep='\t', index_col=0)
    ctrl = [s for s in asv.columns if meta.loc[s, 'sample_type'] == 'control']
    part = [s for s in asv.columns if meta.loc[s, 'sample_type'] != 'control']
    prev_c = (asv[ctrl] >= 1).mean(axis=1)
    prev_p = (asv[part] >= 1).mean(axis=1)
    contaminant = (prev_c >= 0.5) & (prev_p < 0.5)
    keep = (~contaminant) & (asv[part].sum(axis=1) >= 10)
    sub = asv.loc[keep, part].sort_index()
    rel = sub / sub.sum(axis=0)
    Xs = rel.T
    d = bray_curtis(Xs.values)
    pc1 = pcoa_pc1(d)
    stype = meta.loc[part, 'sample_type'].values
    F, p, n = permanova_test(d, stype)
    R2 = r2_from_f(F, len(set(stype)), n)
    idx = {s: i for i, s in enumerate(part)}
    pair_res = {}
    raw, keys = [], []
    for x, y in (('oral', 'bal'), ('oral', 'brush'), ('bal', 'brush')):
        ids = [s for s in part if meta.loc[s, 'sample_type'] in (x, y)]
        sel = np.array([idx[s] for s in ids])
        sub_d = d[np.ix_(sel, sel)]
        Fp, pp, np_ = permanova_test(sub_d, meta.loc[ids, 'sample_type'].values)
        key = x + '_vs_' + y
        pair_res[key] = {'p_value': float(pp), 'r2': float(r2_from_f(Fp, 2, np_))}
        raw.append(pp)
        keys.append(key)
    adj = multipletests(raw, method='fdr_bh')[1]
    for k, key in enumerate(keys):
        pair_res[key]['p_adjusted'] = float(adj[k])
    types, obs, pv = indval(Xs.values, stype)
    ti = {t: i for i, t in enumerate(types)}
    sets = {}
    for t in ('oral', 'bal', 'brush'):
        col = ti[t]
        mask = (obs[:, col] >= INDVAL_MIN) & (pv[:, col] < P_MAX)
        sets[t] = sorted(np.array(sub.index)[mask].tolist())
    famcol = 'Family' if 'Family' in tax.columns else 'family'
    sph = [a for a in sets['brush']
           if str(tax[famcol].get(a, '')).strip().lower() == 'sphingomonadaceae']
    bal_ids = [s for s in part if meta.loc[s, 'sample_type'] == 'bal']
    grp = (meta.loc[bal_ids, 'cancer_status'] == 'cancer').values
    relbal = rel[bal_ids]
    pvals = []
    for a in relbal.index:
        v = relbal.loc[a].values
        try:
            pvals.append(mannwhitneyu(v[grp], v[~grp], alternative='two-sided',
                                      method='asymptotic').pvalue)
        except Exception:
            pvals.append(1.0)
    pvals = np.nan_to_num(np.array(pvals, dtype=float), nan=1.0)
    n_sig = int((multipletests(pvals, method='fdr_bh')[1] < 0.05).sum())
    Fb, pb, nb = permanova_test(d, (meta.loc[part, 'cancer_status'] == 'cancer').values)
    return {
        'raw': asv,
        'meta': meta,
        'tax': tax,
        'part': part,
        'keep': set(sub.index),
        'n_retained': int(len(sub)),
        'n_contaminants': int(contaminant.sum()),
        'pc1': pc1,
        'permanova': {'F': F, 'p': p, 'R2': R2},
        'pairwise': pair_res,
        'indicator': {
            'sets': sets,
            'overlaps': {
                'oral_bal': len(set(sets['oral']) & set(sets['bal'])),
                'oral_brush': len(set(sets['oral']) & set(sets['brush'])),
                'bal_brush': len(set(sets['bal']) & set(sets['brush'])),
            },
            'sphingo': sph[0] if sph else None,
        },
        'cancer': {'n_sig': n_sig, 'permanova_p': float(pb)},
    }


# --------------------------------------------------------------------------
# constraint checks
# --------------------------------------------------------------------------
def check_filtered(gold):
    path = os.path.join(OUT, 'filtered_asv_table.tsv')
    if not os.path.exists(path):
        return False, 'missing /app/output/filtered_asv_table.tsv'
    tab = pd.read_csv(path, sep='\t', index_col=0)
    if list(tab.columns) != list(gold['part']):
        return False, 'sample columns differ from the 99 participant samples (or wrong order)'
    ids = list(tab.index)
    if len(ids) == 0:
        return False, 'filtered table is empty'
    if ids != sorted(ids):
        return False, 'rows are not sorted by ASV id ascending'
    agent = set(ids)
    keep = gold['keep']
    false_pos = sorted(agent - keep)
    covered = len(agent & keep) / max(1, len(keep))
    shared = [a for a in ids if a in keep]
    try:
        same = bool((tab.loc[shared, gold['part']].astype(int).values ==
                     gold['raw'].loc[shared, gold['part']].values).all())
    except Exception:
        same = False
    ok = (len(false_pos) <= 2) and covered >= 0.95 and same
    return ok, ('n_agent=%d n_gold=%d false_positives=%d coverage=%.3f counts_match_input=%s'
                % (len(agent), len(keep), len(false_pos), covered, same))


def check_pc1(gold, report):
    v = get(report, 'beta_diversity', 'pcoa_pc1_variance_explained')
    if v is None:
        return False, 'missing beta_diversity.pcoa_pc1_variance_explained'
    v = float(v)
    delta = abs(v - gold['pc1'])
    return delta <= 0.03, 'agent=%.4f gold=%.4f delta=%.4f (tol 0.03)' % (v, gold['pc1'], delta)


def check_permanova_type(gold, report):
    node = get(report, 'beta_diversity', 'permanova_sample_type') or {}
    r2 = node.get('r2')
    p = node.get('p_value')
    if r2 is None or p is None:
        return False, 'missing r2/p_value for sample_type PERMANOVA'
    r2, p = float(r2), float(p)
    delta = abs(r2 - gold['permanova']['R2'])
    ok = delta <= 0.03 and p <= 0.01
    return ok, ('agent R2=%.4f p=%.4f ; gold R2=%.4f p=%.4f ; delta=%.4f (tol 0.03) ; need p<=0.01'
                % (r2, p, gold['permanova']['R2'], gold['permanova']['p'], delta))


def check_pairwise(gold, report):
    pw = get(report, 'beta_diversity', 'pairwise_permanova') or {}
    need = ['oral_vs_bal', 'oral_vs_brush', 'bal_vs_brush']
    vals = {}
    for k in need:
        v = get(pw, k, 'p_adjusted')
        if v is None:
            return False, 'missing p_adjusted for ' + k
        vals[k] = float(v)
    ok = (vals['oral_vs_bal'] >= 0.05 and vals['oral_vs_brush'] < 0.05
          and vals['bal_vs_brush'] < 0.05)
    return ok, 'adjusted p oral_vs_bal=%.4f oral_vs_brush=%.4f bal_vs_brush=%.4f' % (
        vals['oral_vs_bal'], vals['oral_vs_brush'], vals['bal_vs_brush'])


def check_indicators(gold, report):
    node = get(report, 'indicator_analysis') or {}
    lists = {}
    for t in ('oral', 'bal', 'brush'):
        v = node.get(t)
        if not isinstance(v, list) or not v:
            return False, 'missing or empty indicator list for ' + t
        lists[t] = set(str(x) for x in v)
    overlaps = node.get('overlaps') or {}
    computed = (len(lists['oral'] & lists['bal']),
                len(lists['oral'] & lists['brush']),
                len(lists['bal'] & lists['brush']))
    reported = (overlaps.get('oral_bal'), overlaps.get('oral_brush'), overlaps.get('bal_brush'))
    consistent = tuple(reported) == computed
    sph = node.get('brush_specific_sphingomonadaceae_asv')
    gold_sph = gold['indicator']['sphingo']
    gsize = len(gold['indicator']['sets']['brush'])
    size_ok = 0.5 * gsize <= len(lists['brush']) <= 1.5 * gsize
    pattern_ok = True
    go = gold['indicator']['overlaps']
    if go['oral_bal'] >= 1 and go['oral_brush'] == 0 and go['bal_brush'] == 0:
        pattern_ok = (computed[0] >= 1 and computed[1] == 0 and computed[2] == 0)
    ok = bool(consistent and sph is not None and gold_sph is not None
              and str(sph) == str(gold_sph) and str(sph) in lists['brush']
              and size_ok and pattern_ok)
    return ok, ('sphingo agent=%s gold=%s ; brush_set_size=%d gold=%d ; computed_overlaps=%s reported=%s'
                % (sph, gold_sph, len(lists['brush']), gsize, computed, reported))


def check_cancer(gold, report):
    node = get(report, 'cancer_status') or {}
    n = node.get('n_asvs_fdr_significant')
    p = node.get('permanova_p_value')
    if n is None or p is None:
        return False, 'missing cancer_status fields'
    n, p = int(n), float(p)
    gn = gold['cancer']['n_sig']
    ok = (n == gn)
    if gn == 0:
        ok = ok and (p >= 0.05)
    return ok, 'n_fdr_significant agent=%d gold=%d ; cancer PERMANOVA p agent=%.4f gold=%.4f' % (
        n, gn, p, gold['cancer']['permanova_p'])


def check_schema(gold, report):
    if report is None:
        return False, 'report.json missing or not valid JSON'
    req = [('n_asvs_retained',), ('n_contaminants_removed',),
           ('beta_diversity', 'pcoa_pc1_variance_explained'),
           ('beta_diversity', 'permanova_sample_type', 'pseudo_f'),
           ('beta_diversity', 'permanova_sample_type', 'r2'),
           ('beta_diversity', 'permanova_sample_type', 'p_value'),
           ('beta_diversity', 'pairwise_permanova', 'oral_vs_bal', 'p_adjusted'),
           ('beta_diversity', 'pairwise_permanova', 'oral_vs_brush', 'p_adjusted'),
           ('beta_diversity', 'pairwise_permanova', 'bal_vs_brush', 'p_adjusted'),
           ('indicator_analysis', 'oral'), ('indicator_analysis', 'bal'),
           ('indicator_analysis', 'brush'), ('indicator_analysis', 'overlaps'),
           ('indicator_analysis', 'brush_specific_sphingomonadaceae_asv'),
           ('cancer_status', 'permanova_p_value'),
           ('cancer_status', 'n_asvs_fdr_significant')]
    missing = ['.'.join(k) for k in req if get(report, *k) is None]
    if missing:
        return False, 'missing keys: ' + ', '.join(missing)
    n_ret = int(get(report, 'n_asvs_retained'))
    if abs(n_ret - gold['n_retained']) > 2:
        return False, 'n_asvs_retained=%d but %d ASVs pass the stated rule' % (n_ret, gold['n_retained'])
    nc = int(get(report, 'n_contaminants_removed'))
    if abs(nc - gold['n_contaminants']) > 3:
        return False, 'n_contaminants_removed=%d but %d ASVs satisfy the contaminant rule' % (
            nc, gold['n_contaminants'])
    floats = [get(report, 'beta_diversity', 'pcoa_pc1_variance_explained'),
              get(report, 'beta_diversity', 'permanova_sample_type', 'r2'),
              get(report, 'beta_diversity', 'permanova_sample_type', 'p_value')]
    bad = [v for v in floats if abs(float(v) - round(float(v), 4)) > 1e-9]
    if bad:
        return False, 'values not rounded to 4 decimals: %s' % (bad,)
    for t in ('oral', 'bal', 'brush'):
        v = get(report, 'indicator_analysis', t)
        if list(v) != sorted(list(v)):
            return False, 'indicator list for %s is not sorted by ASV id' % t
    return True, 'schema ok (n_retained=%d, n_contaminants=%d)' % (n_ret, nc)


def main():
    gold = None
    try:
        gold = build_gold()
        print('reference pipeline built: %d ASVs retained, %d contaminants, PC1=%.4f R2=%.4f'
              % (gold['n_retained'], gold['n_contaminants'], gold['pc1'], gold['permanova']['R2']),
              flush=True)
    except Exception:
        traceback.print_exc()

    report = None
    rp = os.path.join(OUT, 'report.json')
    if os.path.exists(rp):
        try:
            with open(rp) as fh:
                report = json.load(fh)
        except Exception:
            traceback.print_exc()

    checkers = [
        ('decontamination_and_filtered_table', lambda: check_filtered(gold)),
        ('pcoa_pc1_variance', lambda: check_pc1(gold, report)),
        ('permanova_sample_type', lambda: check_permanova_type(gold, report)),
        ('pairwise_permanova', lambda: check_pairwise(gold, report)),
        ('indicator_species', lambda: check_indicators(gold, report)),
        ('cancer_status', lambda: check_cancer(gold, report)),
        ('report_schema', lambda: check_schema(gold, report)),
    ]
    for name, fn in checkers:
        if gold is None:
            record(name, False, 'reference pipeline could not be built')
            continue
        try:
            ok, msg = fn()
        except Exception:
            traceback.print_exc()
            ok, msg = False, 'checker raised an exception'
        record(name, ok, msg)

    passed = sum(1 for c in CHECKS if results.get(c))
    score = passed / float(len(CHECKS))
    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as fh:
        json.dump({'score': round(score, 4), 'reward': round(score, 4),
                   'constraints': {c: bool(results.get(c)) for c in CHECKS},
                   'messages': notes}, fh, indent=2)
    print(json.dumps({'score': round(score, 4), 'reward': round(score, 4),
                      'passed': passed, 'total': len(CHECKS)}), flush=True)


if __name__ == '__main__':
    main()
