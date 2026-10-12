#!/usr/bin/env python3
"""Verifier for the circulating-miRNA COVID-19 signature task.

The full reference analysis is recomputed from the raw qPCR data. Raw inputs are
expected in /app/data (they are generated at image build time by the seeded
simulator); if that directory is absent the identical simulator shipped in
tests/gold_gen.py regenerates them into /tmp/gold_data.

Nine independently checked constraints, each worth 1/9 of the reward:
  C1 plate QC / row selection      C6 significant signature
  C2 spike-in normalisation        C7 ROC AUC
  C3 group means / log2fc          C8 Youden operating points
  C4 Welch t / p-values            C9 target network
  C5 BH-FDR q-values
"""
import json
import math
import os
import sys
import traceback

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.multitest import multipletests

DATA_DIR = '/app/data'
OUT_DIR = '/app/output'
REWARD_PATH = '/logs/verifier/reward.json'
REPORT_PATH = '/logs/verifier/report.json'

SPIKES = {'cel-miR-39-3p', 'UniSp6'}
NON_DETECT_CT = 38.0
MIN_DETECT_FRAC = 0.70
FDR_CUT = 0.05
LOG2FC_CUT = math.log2(1.5)
CONF_CUT = 0.75

TOL_DCT = 0.02
TOL_MEAN = 0.05
TOL_LOG2FC = 0.10
TOL_PV = 0.15
TOL_QV = 0.01
TOL_AUC = 0.02
TOL_SE = 0.05


def ensure_data():
    if os.path.exists(os.path.join(DATA_DIR, 'plate_cq.csv')):
        return DATA_DIR
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gold_gen  # identical copy of the seeded input simulator
    tmp = '/tmp/gold_data'
    if not os.path.exists(os.path.join(tmp, 'plate_cq.csv')):
        os.makedirs(tmp, exist_ok=True)
        gold_gen.main(tmp)
    return tmp


def build_reference(data_dir):
    cq = pd.read_csv(os.path.join(data_dir, 'plate_cq.csv'))
    meta = pd.read_csv(os.path.join(data_dir, 'sample_metadata.csv'))
    cq['sample_type'] = cq['sample_type'].astype(str).str.strip().str.upper()
    cq = cq[cq['sample_type'] == 'SAMPLE'].copy()
    ct = pd.to_numeric(cq['cq'], errors='coerce')
    cq['ct'] = np.where(ct > NON_DETECT_CT, np.nan, ct)

    spk = cq[cq['mirna'].isin(SPIKES)]
    spk = spk.groupby('sample_id')['ct'].agg(['mean', 'count'])
    spk = spk[spk['count'] >= 2]['mean']

    mir = cq[~cq['mirna'].isin(SPIKES)]
    agg = mir.groupby(['sample_id', 'mirna'])['ct'].agg(['mean', 'count']).reset_index()
    agg = agg[agg['count'] >= 1]
    agg = agg.merge(spk.rename('spike_mean'), left_on='sample_id', right_index=True, how='inner')
    agg['dct'] = agg['mean'] - agg['spike_mean']

    n_samples = int(spk.shape[0])
    detected = agg.groupby('mirna')['dct'].count()
    retained = sorted(detected[detected >= MIN_DETECT_FRAC * n_samples].index)

    dct = agg[agg['mirna'].isin(retained)][['sample_id', 'mirna', 'dct']]
    dct = dct.merge(meta[['sample_id', 'group']], on='sample_id', how='inner')
    dct = dct.sort_values(['sample_id', 'mirna']).reset_index(drop=True)

    rows = []
    for m in retained:
        s = dct[dct['mirna'] == m]
        con = s[s['group'] == 'Control']['dct'].values
        cov = s[s['group'] == 'COVID-19']['dct'].values
        t, p = stats.ttest_ind(cov, con, equal_var=False)
        rows.append({'mirna': m, 'mean_dct_control': float(con.mean()),
                     'mean_dct_covid': float(cov.mean()),
                     'log2fc': float(con.mean() - cov.mean()),
                     't_statistic': float(t), 'p_value': float(p),
                     'n_control': int(con.size), 'n_covid': int(cov.size)})
    de = pd.DataFrame(rows)
    de['q_value'] = multipletests(de['p_value'].values, method='fdr_bh')[1]
    de['significant'] = (de['q_value'] < FDR_CUT) & (de['log2fc'] > LOG2FC_CUT)

    roc_rows = []
    for m in retained:
        s = dct[dct['mirna'] == m]
        y = (s['group'] == 'COVID-19').astype(int).values
        score = -s['dct'].values
        fpr, tpr, thr = roc_curve(y, score)
        j = tpr - fpr
        i = int(np.argmax(j))
        roc_rows.append({'mirna': m, 'auc': float(roc_auc_score(y, score)),
                         'youden_cutoff_score': float(thr[i]),
                         'sensitivity': float(tpr[i]),
                         'specificity': float(1.0 - fpr[i])})
    roc = pd.DataFrame(roc_rows)

    sig = sorted(de.loc[de['significant'], 'mirna'])
    pairs = pd.read_csv(os.path.join(data_dir, 'mirna_target_pairs.tsv'), sep='\t')
    pairs['confidence'] = pd.to_numeric(pairs['confidence'], errors='coerce')
    sub = pairs[pairs['mirna'].isin(sig) & (pairs['confidence'] >= CONF_CUT)]
    edges = sub.drop_duplicates(subset=['mirna', 'gene'])
    deg = edges.groupby('gene')['mirna'].nunique()
    panel = set()
    with open(os.path.join(data_dir, 'immune_gene_panel.txt')) as fh:
        for line in fh:
            if line.strip():
                panel.add(line.strip())
    net = {
        'n_edges': int(len(edges)),
        'n_target_genes': int(len(deg)),
        'gene_degrees': {g: int(d) for g, d in deg.items()},
        'top_hub_genes': [g for g, _ in sorted(deg.items(), key=lambda kv: (-kv[1], kv[0]))[:5]],
        'immune_overlap': sorted(set(deg.index) & panel),
    }
    return {'dct': dct, 'de': de, 'roc': roc, 'net': net, 'retained': retained, 'sig': sig}


def _num(x):
    try:
        v = float(x)
        return v
    except Exception:
        return float('nan')


def load_tsv(path, cols):
    if not os.path.exists(path):
        return None, 'file %s is missing' % path
    try:
        df = pd.read_csv(path, sep='\t')
    except Exception as exc:
        return None, 'unreadable (%s)' % exc
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None, 'missing columns %s' % missing
    return df, 'ok'


def load_json(path):
    if not os.path.exists(path):
        return None, 'file %s is missing' % path
    try:
        with open(path) as fh:
            return json.load(fh), 'ok'
    except Exception as exc:
        return None, 'unreadable (%s)' % exc


def main():
    checks = []

    def add(name, passed, detail):
        checks.append({'name': name, 'passed': bool(passed), 'detail': detail})

    ref = build_reference(ensure_data())
    ref_dct = ref['dct']
    ref_de = {r['mirna']: r for r in ref['de'].to_dict('records')}
    ref_roc = {r['mirna']: r for r in ref['roc'].to_dict('records')}
    ref_net = ref['net']
    ref_sig = set(ref['sig'])
    ref_keys = set(zip(ref_dct['sample_id'].astype(str), ref_dct['mirna'].astype(str)))
    ref_map = {(str(a), str(b)): float(v) for a, b, v in
               zip(ref_dct['sample_id'], ref_dct['mirna'], ref_dct['dct'])}

    # ---------------- C1 + C2 : normalised dCt table ----------------
    tbl, msg = load_tsv(os.path.join(OUT_DIR, 'normalized_dct.tsv'),
                        ['sample_id', 'mirna', 'dct'])
    if tbl is None:
        add('C1_plate_qc_row_selection', False, msg)
        add('C2_spikein_normalisation', False, msg)
    else:
        got_keys = set(zip(tbl['sample_id'].astype(str), tbl['mirna'].astype(str)))
        extra = len(got_keys - ref_keys)
        missing = len(ref_keys - got_keys)
        dup = int(len(tbl) - len(got_keys))
        ok1 = (got_keys == ref_keys) and dup == 0
        add('C1_plate_qc_row_selection', ok1,
            'expected_rows=%d got_rows=%d duplicate_keys=%d extra=%d missing=%d'
            % (len(ref_keys), len(tbl), dup, extra, missing))
        got_map = {}
        for a, b, v in zip(tbl['sample_id'].astype(str), tbl['mirna'].astype(str), tbl['dct']):
            got_map[(a, b)] = _num(v)
        common = ref_keys & set(got_map.keys())
        within = sum(1 for k in common if abs(got_map[k] - ref_map[k]) <= TOL_DCT)
        frac_cov = len(common) / max(1, len(ref_keys))
        frac_ok = within / max(1, len(common)) if common else 0.0
        worst = 0.0
        for k in common:
            worst = max(worst, abs(got_map[k] - ref_map[k]))
        ok2 = (frac_cov >= 0.95) and (frac_ok >= 0.98)
        add('C2_spikein_normalisation', ok2,
            'coverage=%.3f within_tol=%.3f max_abs_dev=%.4f' % (frac_cov, frac_ok, worst))

    # ---------------- C3 / C4 / C5 : differential expression ----------------
    de_tbl, msg = load_tsv(os.path.join(OUT_DIR, 'de_results.tsv'),
                           ['mirna', 'mean_dct_control', 'mean_dct_covid', 'log2fc',
                            't_statistic', 'p_value', 'q_value', 'significant'])
    if de_tbl is None:
        add('C3_group_means_log2fc', False, msg)
        add('C4_welch_pvalues', False, msg)
        add('C5_bh_fdr_qvalues', False, msg)
    else:
        got = {str(r['mirna']): r for r in de_tbl.to_dict('records')}
        key_ok = set(got.keys()) == set(ref_de.keys())
        bad_m = []
        for m, r in ref_de.items():
            if m not in got:
                bad_m.append(m)
                continue
            g = got[m]
            if abs(_num(g['mean_dct_control']) - r['mean_dct_control']) > TOL_MEAN or \
               abs(_num(g['mean_dct_covid']) - r['mean_dct_covid']) > TOL_MEAN or \
               abs(_num(g['log2fc']) - r['log2fc']) > TOL_LOG2FC:
                bad_m.append(m)
        add('C3_group_means_log2fc', key_ok and not bad_m,
            'mirna_set_ok=%s offenders=%s' % (key_ok, bad_m[:6]))

        bad_p = []
        for m, r in ref_de.items():
            if m not in got:
                bad_p.append(m)
                continue
            g = got[m]
            t_ref, p_ref = r['t_statistic'], r['p_value']
            t_got, p_got = _num(g['t_statistic']), _num(g['p_value'])
            tol_t = 0.10 * abs(t_ref)
            tol_p = max(0.01, TOL_PV * abs(p_ref))
            if not (abs(abs(t_got) - abs(t_ref)) <= tol_t and abs(p_got - p_ref) <= tol_p):
                bad_p.append(m)
        add('C4_welch_pvalues', not bad_p, 'offenders=%s' % bad_p[:6])

        bad_q = []
        for m, r in ref_de.items():
            if m not in got:
                bad_q.append(m)
                continue
            q_ref = float(r['q_value'])
            q_got = _num(got[m]['q_value'])
            tol_q = max(TOL_QV, 0.20 * q_ref)
            if abs(q_got - q_ref) > tol_q:
                bad_q.append(m)
        add('C5_bh_fdr_qvalues', key_ok and not bad_q, 'offenders=%s' % bad_q[:6])

    # ---------------- C6 : signature ----------------
    sig_doc, msg = load_json(os.path.join(OUT_DIR, 'signature.json'))
    if sig_doc is None:
        add('C6_signature', False, msg)
    else:
        lst = sig_doc.get('signature')
        if not isinstance(lst, list):
            add('C6_signature', False, 'signature key missing or not a list')
        else:
            got_sig = set(str(x) for x in lst)
            required = {'miR-30c', 'miR-126'} & ref_sig
            forbidden = {'miR-155'} - ref_sig
            ok = (got_sig == ref_sig) and required.issubset(got_sig)
            ok = ok and not (forbidden & got_sig)
            n_ok = (sig_doc.get('n_signature') == len(got_sig))
            add('C6_signature', ok and n_ok,
                'ref=%s got=%s n_field_ok=%s' % (sorted(ref_sig), sorted(got_sig), n_ok))

    # ---------------- C7 / C8 : ROC ----------------
    roc_tbl, msg = load_tsv(os.path.join(OUT_DIR, 'roc_results.tsv'),
                            ['mirna', 'auc', 'youden_cutoff_score', 'sensitivity', 'specificity'])
    if roc_tbl is None:
        add('C7_roc_auc', False, msg)
        add('C8_youden_points', False, msg)
    else:
        got_roc = {str(r['mirna']): r for r in roc_tbl.to_dict('records')}
        key_ok = set(got_roc.keys()) == set(ref_roc.keys())
        bad_auc = []
        for m, r in ref_roc.items():
            if m not in got_roc or abs(_num(got_roc[m]['auc']) - r['auc']) > TOL_AUC:
                bad_auc.append(m)
        ranked = sorted(ref_roc.items(), key=lambda kv: (-kv[1]['auc'], kv[0]))
        top2 = {ranked[0][0], ranked[1][0]} if len(ranked) > 1 else {ranked[0][0]}
        best_ok = sig_doc is not None and str(sig_doc.get('best_mirna')) in top2
        add('C7_roc_auc', key_ok and not bad_auc and best_ok,
            'mirna_set_ok=%s offenders=%s best_mirna_ok=%s' % (key_ok, bad_auc[:6], best_ok))

        bad_se = []
        for m, r in ref_roc.items():
            if m not in got_roc:
                bad_se.append(m)
                continue
            g = got_roc[m]
            if abs(_num(g['sensitivity']) - r['sensitivity']) > TOL_SE or \
               abs(_num(g['specificity']) - r['specificity']) > TOL_SE:
                bad_se.append(m)
        add('C8_youden_points', key_ok and not bad_se, 'offenders=%s' % bad_se[:6])

    # ---------------- C9 : target network ----------------
    net_doc, msg = load_json(os.path.join(OUT_DIR, 'network.json'))
    if net_doc is None:
        add('C9_target_network', False, msg)
    else:
        reasons = []
        if int(net_doc.get('n_edges', -1)) != ref_net['n_edges']:
            reasons.append('n_edges %s != %s' % (net_doc.get('n_edges'), ref_net['n_edges']))
        if int(net_doc.get('n_target_genes', -1)) != ref_net['n_target_genes']:
            reasons.append('n_target_genes %s != %s' %
                           (net_doc.get('n_target_genes'), ref_net['n_target_genes']))
        gd = net_doc.get('gene_degrees')
        gd_ref = ref_net['gene_degrees']
        if not isinstance(gd, dict) or {k: int(v) for k, v in gd.items()} != gd_ref:
            reasons.append('gene_degrees mismatch (%s entries vs %s)' %
                           (len(gd) if isinstance(gd, dict) else 'n/a', len(gd_ref)))
        hubs = net_doc.get('top_hub_genes')
        if not isinstance(hubs, list) or set(str(x) for x in hubs) != set(ref_net['top_hub_genes']):
            reasons.append('top_hub_genes %s != %s' % (hubs, ref_net['top_hub_genes']))
        ov = net_doc.get('immune_overlap')
        if not isinstance(ov, list) or set(str(x) for x in ov) != set(ref_net['immune_overlap']):
            reasons.append('immune_overlap %s != %s' % (ov, ref_net['immune_overlap']))
        add('C9_target_network', not reasons, '; '.join(reasons) if reasons else 'all network fields match')

    total = len(checks)
    passed = sum(1 for c in checks if c['passed'])
    score = round(passed / float(total), 4)

    report = {
        'score': score,
        'passed': passed,
        'total': total,
        'reference_signature': sorted(ref_sig),
        'reference_best_auc': sorted(ref_roc.items(), key=lambda kv: (-kv[1]['auc'], kv[0]))[0][0]
        if ref_roc else None,
        'reference_network': {'n_edges': ref_net['n_edges'],
                              'n_target_genes': ref_net['n_target_genes'],
                              'top_hub_genes': ref_net['top_hub_genes'],
                              'immune_overlap': ref_net['immune_overlap']},
        'checks': checks,
    }
    return report


if __name__ == '__main__':
    os.makedirs('/logs/verifier', exist_ok=True)
    try:
        report = main()
        score = float(report.get('score', 0.0))
    except Exception:
        traceback.print_exc()
        report = {'score': 0.0, 'error': traceback.format_exc()[-2000:], 'checks': []}
        score = 0.0

    print('==== circulating-miRNA task verification ====')
    for c in report.get('checks', []):
        print('[%s] %-28s %s' % ('PASS' if c['passed'] else 'FAIL', c['name'], c['detail']))
    print('passed %s / %s  -> score %.4f'
          % (report.get('passed', 0), report.get('total', 9), score))

    with open(REWARD_PATH, 'w') as fh:
        json.dump({'score': score, 'reward': score}, fh)
    try:
        with open(REPORT_PATH, 'w') as fh:
            json.dump(report, fh, indent=2, default=str)
    except Exception:
        pass
