#!/usr/bin/env python3
"""Deep grader for the KMT2D/MLL4 CRC immune-microenvironment task.

The grader recomputes the reference statistics directly from the input data
(never from a stored answer file) and compares the agent's /app/results.json
against them constraint by constraint. Score = fraction of constraints passed.
"""
import json
import os
import sys
import subprocess

import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = '/app/results.json'
REWARD_PATH = '/logs/verifier/reward.json'
DATA_CANDIDATES = ['/app/data', os.path.join(HERE, 'data')]
LOF_CLASSES = {'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'Splice_Site'}
CHECKPOINTS = ['CD274', 'CTLA4', 'CD8A']
SIGNATURES = {
    'T_effector': ['GZMA', 'GZMB', 'PRF1', 'IFNG', 'CXCL9', 'CXCL10'],
    'IFN_gamma': ['IDO1', 'STAT1', 'HLA-DRA', 'CXCL11', 'IFNG'],
}
CELL_TYPES = {
    'CD8_T_cell': ['CD8A', 'CD8B', 'GZMK'],
    'NK_cell': ['NKG7', 'KLRD1', 'GNLY'],
    'Macrophage': ['CD68', 'CD163', 'CSF1R'],
    'Treg': ['FOXP3', 'IL2RA', 'IKZF2'],
}


def mwu(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    delta = float(a.mean() - b.mean())
    p = float(stats.mannwhitneyu(a, b, alternative='two-sided').pvalue)
    return delta, p


def analyse(data_dir):
    expr = pd.read_csv(os.path.join(data_dir, 'expression_matrix.tsv'), sep='\t', index_col=0)
    mut = pd.read_csv(os.path.join(data_dir, 'kmt2d_mutations.tsv'), sep='\t')
    clin = pd.read_csv(os.path.join(data_dir, 'clinical.tsv'), sep='\t')
    drug = pd.read_csv(os.path.join(data_dir, 'cell_line_drug_screen.tsv'), sep='\t')

    lof_s = set(mut.loc[mut.variant_classification.isin(LOF_CLASSES), 'sample_id'])
    samples = list(expr.columns)
    lof = [s for s in samples if s in lof_s]
    wt = [s for s in samples if s not in lof_s]
    expr = expr.astype(float)

    def gene_cmp(g):
        return mwu(expr.loc[g, lof].values, expr.loc[g, wt].values)

    z = expr.sub(expr.mean(axis=1), axis=0).div(expr.std(axis=1, ddof=0), axis=0)

    def sig_cmp(genes):
        score = z.loc[genes].mean(axis=0)
        return mwu(score[lof].values, score[wt].values)

    res = {'n_lof': len(lof), 'n_wt': len(wt),
           'immune_markers': {}, 'signatures': {}, 'deconvolution': {}}
    for g in CHECKPOINTS:
        d, p = gene_cmp(g)
        res['immune_markers'][g] = {'delta': d, 'p_value': p}
    for name, genes in SIGNATURES.items():
        d, p = sig_cmp(genes)
        res['signatures'][name] = {'delta': d, 'p_value': p}
    for name, genes in CELL_TYPES.items():
        d, p = sig_cmp(genes)
        res['deconvolution'][name] = {'delta': d, 'p_value': p}

    ici = clin[clin.ici_treated == 'yes'].copy()
    ici['grp'] = np.where(ici.sample_id.isin(lof_s), 'LOF', 'WT')
    ici['resp'] = ici.response.isin(['CR', 'PR'])
    a = int(((ici.grp == 'LOF') & ici.resp).sum())
    b = int(((ici.grp == 'LOF') & ~ici.resp).sum())
    c = int(((ici.grp == 'WT') & ici.resp).sum())
    d2 = int(((ici.grp == 'WT') & ~ici.resp).sum())
    odds, p_fish = stats.fisher_exact([[a, b], [c, d2]])
    res['ici_response'] = {
        'odds_ratio': float(odds),
        'p_value': float(p_fish),
        'response_rate_lof': a / (a + b),
        'response_rate_wt': c / (c + d2),
    }

    t1 = ici.loc[ici.grp == 'LOF', 'os_months']
    e1 = ici.loc[ici.grp == 'LOF', 'os_event']
    t2 = ici.loc[ici.grp == 'WT', 'os_months']
    e2 = ici.loc[ici.grp == 'WT', 'os_event']
    lr = logrank_test(t1, t2, event_observed_A=e1, event_observed_B=e2)
    dd = pd.DataFrame({'T': ici.os_months.values,
                       'E': ici.os_event.values,
                       'lof': (ici.grp == 'LOF').astype(int).values})
    cph = CoxPHFitter().fit(dd, 'T', 'E')
    res['survival'] = {'logrank_p': float(lr.p_value),
                       'hazard_ratio': float(np.exp(cph.params_['lof']))}

    a_auc = drug.loc[drug.kmt2d_status == 'LOF', 'cisplatin_auc'].values
    b_auc = drug.loc[drug.kmt2d_status == 'WT', 'cisplatin_auc'].values
    d_auc, p_auc = mwu(a_auc, b_auc)
    res['drug_sensitivity'] = {'delta': d_auc, 'p_value': p_auc}
    return res


def find_data():
    for cand in DATA_CANDIDATES:
        if os.path.exists(os.path.join(cand, 'expression_matrix.tsv')):
            return cand
    tmp = '/tmp/regen_data'
    os.makedirs(tmp, exist_ok=True)
    subprocess.run([sys.executable, os.path.join(HERE, 'gold_gen.py'), tmp], check=True)
    return tmp


def num(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    if isinstance(cur, bool):
        return None
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def write_reward(score, detail):
    os.makedirs('/logs/verifier', exist_ok=True)
    score = max(0.0, min(1.0, float(score)))
    with open(REWARD_PATH, 'w') as fh:
        json.dump({'score': round(score, 4), 'reward': round(score, 4)}, fh)
    print(json.dumps(detail, indent=2, default=str))


def main():
    if not os.path.exists(RESULT_PATH):
        write_reward(0.0, {'error': 'results.json not found'})
        return
    try:
        with open(RESULT_PATH) as fh:
            agent = json.load(fh)
    except Exception as exc:
        write_reward(0.0, {'error': 'unparseable results.json: %s' % exc})
        return

    gold = analyse(find_data())
    checks = []

    def add(name, ok, detail):
        checks.append({'name': name, 'passed': bool(ok), 'detail': detail})

    # ---------- C1: group sizes ----------
    a_lof = num(agent, 'n_lof')
    a_wt = num(agent, 'n_wt')
    add('group_sizes',
        a_lof == float(gold['n_lof']) and a_wt == float(gold['n_wt']),
        {'agent': [a_lof, a_wt], 'gold': [gold['n_lof'], gold['n_wt']]})

    # ---------- C2: immune marker deltas ----------
    ok = True
    det = {}
    for g in CHECKPOINTS:
        av = num(agent, 'immune_markers', g, 'delta')
        gv = gold['immune_markers'][g]['delta']
        good = av is not None and abs(av - gv) <= 0.03
        ok = ok and good
        det[g] = {'agent': av, 'gold': round(gv, 4), 'tol': 0.03}
    add('immune_marker_deltas', ok, det)

    # ---------- C3: immune marker direction & significance ----------
    ok = True
    det = {}
    for g in CHECKPOINTS:
        av = num(agent, 'immune_markers', g, 'delta')
        ap = num(agent, 'immune_markers', g, 'p_value')
        gv = gold['immune_markers'][g]['delta']
        good = (av is not None and ap is not None and av > 0 and ap < 0.05
                and abs(av - gv) <= 0.20)
        ok = ok and good
        det[g] = {'delta': av, 'p': ap}
    add('immune_marker_direction', ok, det)

    # ---------- C4: signatures ----------
    ok = True
    det = {}
    for name in SIGNATURES:
        av = num(agent, 'signatures', name, 'delta')
        ap = num(agent, 'signatures', name, 'p_value')
        gv = gold['signatures'][name]['delta']
        good = (av is not None and ap is not None and abs(av - gv) <= 0.03
                and av > 0 and ap < 0.05)
        ok = ok and good
        det[name] = {'agent': av, 'gold': round(gv, 4), 'p': ap}
    add('signature_effects', ok, det)

    # ---------- C5: deconvolution effects ----------
    ok = True
    det = {}
    for name in CELL_TYPES:
        av = num(agent, 'deconvolution', name, 'delta')
        gv = gold['deconvolution'][name]['delta']
        if name == 'Treg':
            good = av is not None and abs(av - gv) <= 0.03 and av < 0
        else:
            good = av is not None and abs(av - gv) <= 0.03 and av > 0
        ok = ok and good
        det[name] = {'agent': av, 'gold': round(gv, 4)}
    add('deconvolution_effects', ok, det)

    # ---------- C6: deconvolution significance ----------
    ok = True
    det = {}
    for name in CELL_TYPES:
        av = num(agent, 'deconvolution', name, 'delta')
        ap = num(agent, 'deconvolution', name, 'p_value')
        gv = gold['deconvolution'][name]['delta']
        good = (av is not None and ap is not None and ap < 0.05
                and abs(av - gv) <= 0.20)
        ok = ok and good
        det[name] = {'delta': av, 'p': ap}
    add('deconvolution_significance', ok, det)

    # ---------- C7: ICI response ----------
    a_or = num(agent, 'ici_response', 'odds_ratio')
    a_pr = num(agent, 'ici_response', 'p_value')
    a_rl = num(agent, 'ici_response', 'response_rate_lof')
    a_rw = num(agent, 'ici_response', 'response_rate_wt')
    g_or = gold['ici_response']['odds_ratio']
    g_rl = gold['ici_response']['response_rate_lof']
    g_rw = gold['ici_response']['response_rate_wt']
    ok = (a_or is not None and a_pr is not None and a_rl is not None and a_rw is not None
          and abs(a_or - g_or) <= 0.15 and abs(a_rl - g_rl) <= 0.02
          and abs(a_rw - g_rw) <= 0.02 and a_or > 1 and a_pr < 0.05 and a_rl > a_rw)
    add('ici_response', ok,
        {'odds_ratio': a_or, 'gold_or': round(g_or, 4),
         'rate_lof': a_rl, 'gold_rate_lof': round(g_rl, 4),
         'rate_wt': a_rw, 'gold_rate_wt': round(g_rw, 4), 'p': a_pr})

    # ---------- C8: survival ----------
    a_hr = num(agent, 'survival', 'hazard_ratio')
    a_lp = num(agent, 'survival', 'logrank_p')
    g_hr = gold['survival']['hazard_ratio']
    ok = (a_hr is not None and a_lp is not None and abs(a_hr - g_hr) <= 0.10
          and a_hr < 1 and a_lp < 0.05)
    add('survival', ok,
        {'hazard_ratio': a_hr, 'gold_hr': round(g_hr, 4), 'logrank_p': a_lp})

    # ---------- C9: cisplatin sensitivity ----------
    a_d = num(agent, 'drug_sensitivity', 'delta')
    a_dp = num(agent, 'drug_sensitivity', 'p_value')
    g_d = gold['drug_sensitivity']['delta']
    ok = (a_d is not None and a_dp is not None and abs(a_d - g_d) <= 0.05
          and a_d < 0 and a_dp < 0.05)
    add('drug_sensitivity', ok,
        {'delta': a_d, 'gold_delta': round(g_d, 4), 'p': a_dp})

    # ---------- C10: schema ----------
    schema_ok = True
    schema_detail = {}
    for key in ['n_lof', 'n_wt']:
        v = num(agent, key)
        schema_ok = schema_ok and v is not None
        schema_detail[key] = v
    for g in CHECKPOINTS:
        for sub in ['delta', 'p_value']:
            v = num(agent, 'immune_markers', g, sub)
            schema_ok = schema_ok and v is not None
    for name in SIGNATURES:
        for sub in ['delta', 'p_value']:
            v = num(agent, 'signatures', name, sub)
            schema_ok = schema_ok and v is not None
    for name in CELL_TYPES:
        for sub in ['delta', 'p_value']:
            v = num(agent, 'deconvolution', name, sub)
            schema_ok = schema_ok and v is not None
    for sub in ['odds_ratio', 'p_value', 'response_rate_lof', 'response_rate_wt']:
        v = num(agent, 'ici_response', sub)
        schema_ok = schema_ok and v is not None
    for sub in ['logrank_p', 'hazard_ratio']:
        v = num(agent, 'survival', sub)
        schema_ok = schema_ok and v is not None
    for sub in ['delta', 'p_value']:
        v = num(agent, 'drug_sensitivity', sub)
        schema_ok = schema_ok and v is not None
    add('schema', schema_ok, schema_detail)

    passed = sum(1 for c in checks if c['passed'])
    total = len(checks)
    score = passed / total
    write_reward(score, {'passed': passed, 'total': total, 'score': round(score, 4),
                         'checks': checks})


if __name__ == '__main__':
    main()
