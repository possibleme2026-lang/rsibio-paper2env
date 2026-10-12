#!/usr/bin/env python3
import os
import sys
import json
import math
import numpy as np
import pandas as pd

sys.path.insert(0, '/tests')
import gold_gen

LOGS_DIR = '/logs/verifier'
os.makedirs(LOGS_DIR, exist_ok=True)

checks = []

def add_check(name, points, detail=''):
    checks.append({
        'name': name,
        'points': float(points),
        'detail': str(detail)
    })

def finalize():
    total_points = sum(c['points'] for c in checks)
    max_points = len(checks) if len(checks) > 0 else 1
    score = total_points / max_points
    report = {
        'score': float(score),
        'checks': checks,
        'total_points': float(total_points),
        'max_points': float(max_points)
    }
    with open(os.path.join(LOGS_DIR, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(LOGS_DIR, 'reward.json'), 'w') as f:
        json.dump({'score': float(score), 'reward': float(score)}, f)
    print(json.dumps(report, indent=2))
    return score

def main():
    try:
        spots = pd.read_csv('/app/data/spots.csv')
        expr = pd.read_csv('/app/data/expression.csv')
        gene_names = list(expr.columns[1:])
    except Exception as e:
        add_check('input_data_readable', 0.0, str(e))
        finalize()
        return

    ranked_path = '/app/output/ranked_pips.csv'
    panel_path = '/app/output/panel.txt'

    ok_files = os.path.isfile(ranked_path) and os.path.isfile(panel_path)
    add_check('output_files_exist', 1.0 if ok_files else 0.0,
              f'ranked={os.path.isfile(ranked_path)}, panel={os.path.isfile(panel_path)}')
    if not ok_files:
        finalize()
        return

    try:
        ranked = pd.read_csv(ranked_path)
    except Exception as e:
        add_check('ranked_pips_parseable', 0.0, str(e))
        finalize()
        return

    has_cols = list(ranked.columns) == ['gene', 'pip']
    n_rows = len(ranked)
    genes_in_ranked = set(ranked['gene'].astype(str))
    expected_genes = set(gene_names)
    no_dup = ranked['gene'].nunique() == len(ranked)
    all_genes = genes_in_ranked == expected_genes
    format_ok = has_cols and n_rows == len(gene_names) and no_dup and all_genes
    add_check('ranked_format', 1.0 if format_ok else 0.0,
              f'cols={list(ranked.columns)}, rows={n_rows}, unique={no_dup}, all_genes={all_genes}')

    try:
        pips = ranked['pip'].astype(float).values
        range_ok = np.all((pips >= 0.0) & (pips <= 1.0)) and np.all(np.isfinite(pips))
    except Exception:
        range_ok = False
        pips = np.array([])
    add_check('pip_range', 1.0 if range_ok else 0.0,
              'all pips in [0,1]' if range_ok else 'invalid pips')

    sort_ok = False
    if format_ok and len(pips) == len(ranked):
        order = np.lexsort((ranked['gene'].astype(str).values, -pips))
        sort_ok = np.array_equal(order, np.arange(len(ranked)))
    add_check('ranking_sorted', 1.0 if sort_ok else 0.0,
              'sorted by pip desc, gene asc' if sort_ok else 'not sorted')

    panel_ok = False
    panel_genes = []
    try:
        with open(panel_path, 'r') as f:
            panel_genes = [line.strip() for line in f if line.strip()]
        panel_ok = (3 <= len(panel_genes) <= 15)
        panel_ok = panel_ok and len(set(panel_genes)) == len(panel_genes)
        panel_ok = panel_ok and all(g in expected_genes for g in panel_genes)
        if panel_ok and format_ok:
            pip_map = dict(zip(ranked['gene'], pips))
            panel_pips = [pip_map[g] for g in panel_genes]
            for i in range(len(panel_genes)-1):
                g1, g2 = panel_genes[i], panel_genes[i+1]
                p1, p2 = panel_pips[i], panel_pips[i+1]
                if p1 < p2 - 1e-12:
                    panel_ok = False
                    break
                if abs(p1 - p2) < 1e-12 and g1 > g2:
                    panel_ok = False
                    break
    except Exception:
        panel_ok = False
    add_check('panel_format', 1.0 if panel_ok else 0.0,
              f'panel_size={len(panel_genes)}' if panel_ok else 'panel invalid')

    corr_points = 0.0
    corr_val = None
    try:
        X, y, S, gene_names_ref = gold_gen.load_data('/app/data')
        ref_pips, _ = gold_gen.compute_rkmr_pips(X, y, S)
        ref_map = dict(zip(gene_names_ref, ref_pips))
        agent_map = dict(zip(ranked['gene'], pips))
        if set(gene_names_ref) == set(agent_map.keys()):
            agent_vec = np.array([agent_map[g] for g in gene_names_ref])
            ref_vec = np.array([ref_map[g] for g in gene_names_ref])
            from scipy.stats import spearmanr
            corr_val, _ = spearmanr(agent_vec, ref_vec)
            if not np.isfinite(corr_val):
                corr_val = 0.0
            if corr_val >= 0.90:
                corr_points = 1.0
            elif corr_val >= 0.75:
                corr_points = 0.5
            else:
                corr_points = 0.0
    except Exception as e:
        corr_val = None
    add_check('pip_correlation', corr_points,
              f'spearman={corr_val:.4f}' if corr_val is not None else 'error')

    f1_points = 0.0
    f1_val = None
    if panel_ok:
        true_markers = set(gold_gen.TRUE_MARKERS)
        panel_set = set(panel_genes)
        tp = len(panel_set & true_markers)
        precision = tp / len(panel_set) if len(panel_set) > 0 else 0.0
        recall = tp / len(true_markers) if len(true_markers) > 0 else 0.0
        if precision + recall > 0:
            f1_val = 2 * precision * recall / (precision + recall)
        else:
            f1_val = 0.0
        if f1_val >= 0.60:
            f1_points = 1.0
        elif f1_val >= 0.40:
            f1_points = 0.5
        else:
            f1_points = 0.0
    add_check('panel_f1', f1_points,
              f'f1={f1_val:.4f}' if f1_val is not None else 'no panel')

    finalize()

if __name__ == '__main__':
    main()
