#!/usr/bin/env python3
'''Deep grader for the Hippo/TEAD kidney transcriptomics task.

The grader regenerates the input data from the protected seed (gold_gen.py),
re-runs the full PyDESeq2 pipeline specified in the task instructions, and
compares the agent artefacts in /app/output against the independently
recomputed values.  Ten constraints are scored separately and the mean is
written to /logs/verifier/reward.json as {"score": x, "reward": x}.
'''

import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_gen import generate

GOLD_DIR = '/tmp/gold_inputs'
OUT_DIR = '/app/output'
LOG_DIR = '/logs/verifier'
ALPHA = 0.05
LFC_CUT = 1.0
PERM_SEED = 42
N_PERM = 1000
CONTRASTS = ['TEADi_low', 'TEADi_high', 'TEADi_recovery', 'YAPTAZ_KD']
PANEL = 'ALDOSTERONE_SIGNALING'
NEED_COLS = ['gene', 'baseMean', 'log2FoldChange', 'lfcSE', 'stat', 'pvalue', 'padj']
REQUIRED_KEYS = ['n_samples', 'n_genes_retained', 'de_counts', 'concordant_high_vs_KD',
                 'aldosterone_panel_log2fc', 'aldosterone_signature_score',
                 'aldosterone_permutation_p', 'dose_monotonic_fraction',
                 'recovery_persistent_fraction', 'set_mean_log2fc_ranking',
                 'top20_high_vs_vehicle']


def clip01(x):
    return float(max(0.0, min(1.0, x)))


def is_de(df):
    padj = df['padj']
    return padj.notna() & (padj <= ALPHA) & (df['log2FoldChange'].abs() >= LFC_CUT)


def load_gmt(path):
    sets = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            sets[parts[0]] = [g for g in parts[2:] if g]
    return sets


def run_deseq(counts, meta):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    dds = DeseqDataSet(counts=counts, metadata=meta, design='~condition')
    dds.deseq2()
    out = {}
    for grp in CONTRASTS:
        st = DeseqStats(dds, contrast=['condition', grp, 'vehicle'], independent_filter=False)
        st.summary()
        out[grp] = st.results_df.copy()
    return out


def compute_gold():
    generate(GOLD_DIR)
    raw = pd.read_csv(os.path.join(GOLD_DIR, 'kidney_counts.csv'), index_col=0)
    meta = pd.read_csv(os.path.join(GOLD_DIR, 'sample_metadata.csv'), index_col='sample_id')
    gmt = load_gmt(os.path.join(GOLD_DIR, 'gene_sets.gmt'))

    keep = (raw >= 10).sum(axis=1) >= 6
    counts = raw.loc[keep]
    meta = meta.loc[counts.columns]
    res = run_deseq(counts.T, meta)

    de = {g: set(res[g].index[is_de(res[g]).to_numpy()]) for g in CONTRASTS}
    hi = res['TEADi_high']
    kd = res['YAPTAZ_KD']
    lfc_hi = hi['log2FoldChange']
    concordant = sorted([g for g in (de['TEADi_high'] & de['YAPTAZ_KD'])
                         if np.sign(hi.loc[g, 'log2FoldChange']) == np.sign(kd.loc[g, 'log2FoldChange'])])

    panel_genes = [g for g in gmt[PANEL] if g in hi.index]
    panel_lfc = {g: float(lfc_hi[g]) for g in panel_genes}
    score = float(np.mean([panel_lfc[g] for g in panel_genes])) if panel_genes else float('nan')

    rng = np.random.default_rng(PERM_SEED)
    universe = lfc_hi.to_numpy()
    k = len(panel_genes)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = abs(float(np.mean(rng.choice(universe, size=k, replace=False))))
    p_emp = (1.0 + float(np.sum(null >= abs(score)))) / (1.0 + N_PERM)

    lfc_lo = res['TEADi_low']['log2FoldChange']
    mono = [g for g in panel_genes if (lfc_hi[g] < lfc_lo[g]) and (lfc_hi[g] < 0.0)]
    mono_frac = len(mono) / float(len(panel_genes)) if panel_genes else float('nan')

    rec = res['TEADi_recovery']
    hi_de = sorted(de['TEADi_high'])
    persistent = [g for g in hi_de
                  if pd.notna(rec.loc[g, 'padj']) and rec.loc[g, 'padj'] <= ALPHA
                  and abs(rec.loc[g, 'log2FoldChange']) >= LFC_CUT
                  and np.sign(rec.loc[g, 'log2FoldChange']) == np.sign(hi.loc[g, 'log2FoldChange'])]
    persist_frac = len(persistent) / float(len(hi_de)) if hi_de else float('nan')

    set_means = {}
    for name, members in gmt.items():
        vals = [float(lfc_hi[g]) for g in members if g in hi.index]
        set_means[name] = float(np.mean(vals)) if vals else float('-inf')
    ranking = [n for n, _ in sorted(set_means.items(), key=lambda kv: -kv[1])]

    tmp = hi[['padj']].dropna().copy()
    tmp['gene'] = tmp.index.astype(str)
tmp.index.name = None
    tmp = tmp.sort_values(['padj', 'gene'], kind='mergesort')
    top20 = [str(g) for g in tmp['gene'].head(20)]

    return {
        'n_samples': int(counts.shape[1]),
        'n_genes_retained': int(counts.shape[0]),
        'de_counts': {g: int(len(de[g])) for g in CONTRASTS},
        'concordant': concordant,
        'panel_lfc': panel_lfc,
        'signature_score': score,
        'perm_p': p_emp,
        'dose_monotonic': mono_frac,
        'recovery_persistent': persist_frac,
        'set_ranking': ranking,
        'top20': top20,
        'set_means': set_means,
    }


def frac_score(value, gold, tol=0.03, scale=0.15):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:
        return 0.0
    d = abs(v - float(gold))
    if d <= tol:
        return 1.0
    return clip01(1.0 - d / scale)


def emit(score, report):
    os.makedirs(LOG_DIR, exist_ok=True)
    score = clip01(score)
    with open(os.path.join(LOG_DIR, 'reward.json'), 'w') as fh:
        json.dump({'score': score, 'reward': score}, fh)
    with open(os.path.join(LOG_DIR, 'report.json'), 'w') as fh:
        json.dump(report, fh, indent=2)
    print('=== constraint breakdown ===')
    for name, info in report.get('constraints', {}).items():
        print('%-30s %.4f  %s' % (name, info['score'], info['detail']))
    for note in report.get('notes', []):
        print('note:', note)
    print('FINAL SCORE: %.4f' % score)


def main():
    report = {'constraints': {}, 'notes': []}
    scores = []

    def add(name, s, detail):
        s = clip01(s)
        report['constraints'][name] = {'score': round(s, 4), 'detail': str(detail)}
        scores.append(s)

    try:
        gold = compute_gold()
    except Exception:
        report['notes'].append('gold computation failed: ' + traceback.format_exc()[-900:])
        emit(0.0, report)
        return

    sub = None
    try:
        with open(os.path.join(OUT_DIR, 'results.json')) as fh:
            sub = json.load(fh)
        if not isinstance(sub, dict):
            sub = None
    except Exception as exc:
        report['notes'].append('results.json unreadable: %s' % exc)

    tsv = None
    try:
        tsv = pd.read_csv(os.path.join(OUT_DIR, 'de_TEADi_high_vs_vehicle.tsv'), sep='\t')
    except Exception as exc:
        report['notes'].append('DE table unreadable: %s' % exc)

    # ---- C1 accounting -------------------------------------------------
    if sub is None:
        add('c01_input_accounting', 0.0, 'no results.json')
    else:
        ok_n = int(sub.get('n_samples', -1)) == gold['n_samples']
        ok_g = int(sub.get('n_genes_retained', -1)) == gold['n_genes_retained']
        add('c01_input_accounting', 0.5 * ok_n + 0.5 * ok_g,
            'n_samples %s (gold %d) ok=%s; n_genes_retained %s (gold %d) ok=%s' % (
                sub.get('n_samples'), gold['n_samples'], ok_n,
                sub.get('n_genes_retained'), gold['n_genes_retained'], ok_g))

    # ---- C2 DE counts per contrast -------------------------------------
    got = (sub or {}).get('de_counts', {}) if isinstance((sub or {}).get('de_counts', {}), dict) else {}
    hits = 0
    detail = []
    for name in CONTRASTS:
        gv = gold['de_counts'][name]
        try:
            v = int(got.get(name, -1))
        except (TypeError, ValueError):
            v = -1
        tol = max(2, int(0.02 * gv))
        ok = abs(v - gv) <= tol
        hits += int(ok)
        detail.append('%s: %d vs %d (tol %d) %s' % (name, v, gv, tol, 'ok' if ok else 'FAIL'))
    add('c02_de_counts', hits / 4.0, '; '.join(detail))

    # ---- C3 concordant gene set ----------------------------------------
    gold_set = set(gold['concordant'])
    block = (sub or {}).get('concordant_high_vs_KD', {})
    rep_list = block.get('genes', []) if isinstance(block, dict) else []
    if not isinstance(rep_list, list):
        rep_list = []
    rep_set = set(str(g) for g in rep_list)
    tp = len(rep_set & gold_set)
    prec = tp / float(len(rep_set)) if rep_set else 0.0
    rec = tp / float(len(gold_set)) if gold_set else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    list_ok = (block.get('n') == len(rep_list)) if isinstance(block, dict) else False
    sorted_ok = rep_list == sorted(rep_list) if rep_list else False
    if list_ok and sorted_ok:
        add('c03_concordant_set', f1, 'F1=%.3f (precision %.3f, recall %.3f, gold n=%d)' % (
            f1, prec, rec, len(gold_set)))
    else:
        add('c03_concordant_set', 0.4 * f1, 'F1=%.3f but n-field/sorted-order violated' % f1)

    # ---- C4 aldosterone panel log2FC -----------------------------------
    panel_rep = (sub or {}).get('aldosterone_panel_log2fc', {})
    if not isinstance(panel_rep, dict):
        panel_rep = {}
    gold_panel = gold['panel_lfc']
    n_gold = max(1, len(gold_panel))
    cov = 0
    acc = 0
    for gene, gv in gold_panel.items():
        if gene in panel_rep:
            cov += 1
            try:
                if abs(float(panel_rep[gene]) - gv) <= 0.10:
                    acc += 1
            except (TypeError, ValueError):
                pass
    add('c04_aldosterone_panel', 0.5 * (cov / float(n_gold)) + 0.5 * (acc / float(n_gold)),
        'coverage %d/%d, per-gene accuracy %d/%d (tol 0.10)' % (cov, n_gold, acc, n_gold))

    # ---- C5 signature score + permutation p ----------------------------
    s5 = []
    try:
        v = float((sub or {}).get('aldosterone_signature_score'))
    except (TypeError, ValueError):
        v = None
    s5.append(frac_score(v, gold['signature_score'], tol=0.05, scale=0.30))
    try:
        p = float((sub or {}).get('aldosterone_permutation_p'))
    except (TypeError, ValueError):
        p = None
    s5.append(frac_score(p, gold['perm_p'], tol=0.005, scale=0.05))
    add('c05_signature_and_perm_p', 0.5 * (s5[0] + s5[1]),
        'score %s (gold %.4f); p %s (gold %.5f)' % (v, gold['signature_score'], p, gold['perm_p']))

    # ---- C6 dose monotonicity ------------------------------------------
    add('c06_dose_monotonic_fraction',
        frac_score((sub or {}).get('dose_monotonic_fraction'), gold['dose_monotonic']),
        'reported %s (gold %.4f)' % ((sub or {}).get('dose_monotonic_fraction'), gold['dose_monotonic']))

    # ---- C7 recovery persistence ---------------------------------------
    add('c07_recovery_persistent_fraction',
        frac_score((sub or {}).get('recovery_persistent_fraction'), gold['recovery_persistent']),
        'reported %s (gold %.4f)' % ((sub or {}).get('recovery_persistent_fraction'),
                                     gold['recovery_persistent']))

    # ---- C8 gene-set ranking -------------------------------------------
    rep_rank = (sub or {}).get('set_mean_log2fc_ranking', [])
    if not isinstance(rep_rank, list):
        rep_rank = []
    gold_rank = gold['set_ranking']
    pos_hits = sum(1 for i, name in enumerate(gold_rank) if i < len(rep_rank) and rep_rank[i] == name)
    add('c08_set_ranking', pos_hits / float(len(gold_rank)),
        'position hits %d/%d; reported %s; gold %s' % (pos_hits, len(gold_rank), rep_rank, gold_rank))

    # ---- C9 top-20 ordering --------------------------------------------
    top = (sub or {}).get('top20_high_vs_vehicle', [])
    if not isinstance(top, list):
        top = []
    gtop = gold['top20']
    top_hits = sum(1 for i, g in enumerate(gtop) if i < len(top) and str(top[i]) == g)
    add('c09_top20_ordering', top_hits / float(len(gtop)),
        'position hits %d/%d' % (top_hits, len(gtop)))

    # ---- C10 output format ---------------------------------------------
    fmt = 0.0
    key_frac = 0.0
    if sub is not None:
        present = sum(1 for k in REQUIRED_KEYS if k in sub)
        key_frac = present / float(len(REQUIRED_KEYS))
    fmt += 0.4 * key_frac
    header_ok = tsv is not None and list(tsv.columns) == NEED_COLS
    rows_ok = header_ok and len(tsv) == gold['n_genes_retained']
    fmt += 0.3 * (1.0 if rows_ok else 0.0)
    order_ok = False
    if header_ok and len(tsv) > 0:
        order_ok = True
        prev = None
        for padj, gene in zip(tsv['padj'].to_numpy(), tsv['gene'].astype(str).to_numpy()):
            key = (float('inf') if padj != padj else float(padj), str(gene))
            if prev is not None and key < prev:
                order_ok = False
                break
            prev = key
    fmt += 0.3 * (1.0 if order_ok else 0.0)
    add('c10_output_format', fmt,
        'json keys %.2f; header+rows %s; sort order %s' % (key_frac, rows_ok, order_ok))

    total = sum(scores) / float(len(scores)) if scores else 0.0
    emit(total, report)


if __name__ == '__main__':
    main()
