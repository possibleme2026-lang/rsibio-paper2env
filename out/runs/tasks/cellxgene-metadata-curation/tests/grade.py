#!/usr/bin/env python3
'''Deep grader: rebuilds hidden gold from a protected seed and scores ten
independently verifiable curation constraints with partial credit.'''
import csv
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gold_gen

OUT = '/app/output'
OBS = os.path.join(OUT, 'obs.tsv')
VAR = os.path.join(OUT, 'var.tsv')
REP = os.path.join(OUT, 'report.json')

OBS_COLS = ['cell_id', 'donor_id', 'tissue_ontology_term_id', 'tissue',
            'cell_type_ontology_term_id', 'cell_type', 'assay_ontology_term_id', 'assay',
            'disease_ontology_term_id', 'disease', 'organism_ontology_term_id', 'organism',
            'sex', 'development_stage_ontology_term_id', 'development_stage',
            'n_genes_detected']
VAR_COLS = ['gene_id', 'gene_name', 'feature_is_filtered']
REPORT_KEYS = ['n_input_rows', 'n_duplicate_rows_removed', 'n_dropped_unresolved_ontology',
               'n_dropped_low_complexity', 'n_retained_cells', 'n_retained_donors',
               'n_genes_input_rows', 'n_genes_output_rows', 'n_genes_filtered']

WEIGHTS = {'output_schema': 0.08, 'cell_set': 0.14, 'ontology_ids': 0.12,
           'ontology_labels': 0.12, 'sex_organism': 0.10, 'disease_rule': 0.10,
           'development_stage': 0.10, 'donor_consistency': 0.08, 'var_table': 0.08,
           'report_counts': 0.08}

ID2LABEL = {name: {t[0]: t[1] for t in table} for name, table in gold_gen.TABLES}
VALID = {name: set(t[0] for t in table) for name, table in gold_gen.TABLES}


def read_tsv_rows(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline='') as fh:
        for row in csv.reader(fh, delimiter='\t'):
            if row:
                rows.append(row)
    return rows


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / float(len(a | b))


def frac(num, den):
    return float(num) / float(den) if den else 0.0


def ids_valid(d):
    ok = d['tissue_ontology_term_id'] in VALID['uberon']
    ok = ok and d['cell_type_ontology_term_id'] in VALID['cl']
    ok = ok and d['assay_ontology_term_id'] in VALID['efo']
    ok = ok and d['organism_ontology_term_id'] in VALID['ncbitaxon']
    dis = d['disease_ontology_term_id']
    ok = ok and (dis in VALID['mondo'] or dis == 'PATO:0000461' or dis == 'unknown')
    dev = d['development_stage_ontology_term_id']
    ok = ok and (dev in VALID['hsapdv'] or dev in VALID['mmusdv'] or dev == 'unknown')
    return bool(ok)


def labels_ok(d):
    if d['tissue'] != ID2LABEL['uberon'].get(d['tissue_ontology_term_id']):
        return False
    if d['cell_type'] != ID2LABEL['cl'].get(d['cell_type_ontology_term_id']):
        return False
    if d['assay'] != ID2LABEL['efo'].get(d['assay_ontology_term_id']):
        return False
    if d['organism'] != ID2LABEL['ncbitaxon'].get(d['organism_ontology_term_id']):
        return False
    dis = d['disease_ontology_term_id']
    exp_dis = ID2LABEL['pato'].get(dis) or ID2LABEL['mondo'].get(dis)
    if dis == 'unknown':
        exp_dis = 'unknown'
    if d['disease'] != exp_dis:
        return False
    dev = d['development_stage_ontology_term_id']
    exp_dev = ID2LABEL['hsapdv'].get(dev) or ID2LABEL['mmusdv'].get(dev)
    if dev == 'unknown':
        exp_dev = 'unknown'
    if d['development_stage'] != exp_dev:
        return False
    return True


def main():
    checks = {}
    root = tempfile.mkdtemp(prefix='gold_')
    try:
        gold = gold_gen.build_gold(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    obs_rows = read_tsv_rows(OBS)
    var_rows = read_tsv_rows(VAR)
    report = None
    if os.path.exists(REP):
        try:
            with open(REP) as fh:
                report = json.load(fh)
        except Exception:
            report = None

    # ---- C1 output schema -------------------------------------------------
    c1 = 0.0
    obs_header_ok = bool(obs_rows) and obs_rows[0] == OBS_COLS
    if obs_header_ok:
        c1 += 0.30
    ag = {}
    if obs_header_ok:
        for r in obs_rows[1:]:
            if len(r) != len(OBS_COLS):
                continue
            d = dict(zip(OBS_COLS, r))
            if d['cell_id'] not in ag:
                ag[d['cell_id']] = d
        ids = [d['cell_id'] for d in ag.values()]
        if ids == sorted(ids) and len(ids) == len(set(ids)):
            c1 += 0.20
    if var_rows and var_rows[0] == VAR_COLS:
        c1 += 0.20
    if isinstance(report, dict) and all(k in report for k in REPORT_KEYS) and all(
            isinstance(report[k], int) and not isinstance(report[k], bool) for k in REPORT_KEYS):
        c1 += 0.30
    checks['output_schema'] = c1

    # ---- C2 retained cell set --------------------------------------------
    checks['cell_set'] = jaccard(ag.keys(), gold['obs_ids'])

    # ---- C3 ontology ids --------------------------------------------------
    if ag:
        checks['ontology_ids'] = frac(sum(1 for d in ag.values() if ids_valid(d)), len(ag))
    else:
        checks['ontology_ids'] = 0.0

    # ---- C4 label / id agreement ------------------------------------------
    if ag:
        checks['ontology_labels'] = frac(sum(1 for d in ag.values() if labels_ok(d)), len(ag))
    else:
        checks['ontology_labels'] = 0.0

    gold_obs = {d['cell_id']: d for d in gold['obs']}
    common = [cid for cid in ag if cid in gold_obs]

    # ---- C5 sex + organism ------------------------------------------------
    hit = sum(1 for cid in common
              if ag[cid]['sex'] == gold_obs[cid]['sex']
              and ag[cid]['organism_ontology_term_id'] == gold_obs[cid]['organism_ontology_term_id'])
    checks['sex_organism'] = frac(hit, len(common))

    # ---- C6 disease rule --------------------------------------------------
    hit = sum(1 for cid in common
              if ag[cid]['disease_ontology_term_id'] == gold_obs[cid]['disease_ontology_term_id']
              and ag[cid]['disease'] == gold_obs[cid]['disease'])
    checks['disease_rule'] = frac(hit, len(common))

    # ---- C7 developmental stage -------------------------------------------
    hit = sum(1 for cid in common
              if ag[cid]['development_stage_ontology_term_id'] == gold_obs[cid]['development_stage_ontology_term_id']
              and ag[cid]['development_stage'] == gold_obs[cid]['development_stage'])
    checks['development_stage'] = frac(hit, len(common))

    # ---- C8 donor consistency ---------------------------------------------
    donor_sigs = defaultdict(set)
    for d in ag.values():
        donor_sigs[d['donor_id']].add((d['organism_ontology_term_id'], d['sex'],
                                       d['development_stage_ontology_term_id'],
                                       d['disease_ontology_term_id']))
    consistency = frac(sum(1 for v in donor_sigs.values() if len(v) == 1), len(donor_sigs))
    checks['donor_consistency'] = 0.5 * consistency + 0.5 * jaccard(donor_sigs.keys(), gold['donor_ids'])

    # ---- C9 var table ------------------------------------------------------
    gold_var = {v['gene_id']: v for v in gold['var']}
    c9 = 0.0
    if var_rows and var_rows[0] == VAR_COLS:
        arows = [dict(zip(VAR_COLS, r)) for r in var_rows[1:] if len(r) == 3]
        gids = [r['gene_id'] for r in arows]
        sub = 0.0
        if len(gids) == len(set(gids)):
            sub += 0.2
        if gids == sorted(gids):
            sub += 0.2
        if not any('.' in g for g in gids):
            sub += 0.2
        if len(gids) == len(gold_var):
            sub += 0.2
        nm = sum(1 for r in arows if gold_var.get(r['gene_id'], {}).get('gene_name') == r['gene_name'])
        if frac(nm, len(gold_var)) > 0.999:
            sub += 0.2
        fl = sum(1 for r in arows
                 if gold_var.get(r['gene_id'], {}).get('feature_is_filtered') == r['feature_is_filtered'])
        c9 = 0.5 * sub + 0.5 * frac(fl, len(gold_var))
    checks['var_table'] = c9

    # ---- C10 report counts -------------------------------------------------
    if isinstance(report, dict):
        matched = sum(1 for k in REPORT_KEYS if report.get(k) == gold['report'][k])
        checks['report_counts'] = frac(matched, len(REPORT_KEYS))
    else:
        checks['report_counts'] = 0.0

    total = sum(WEIGHTS[k] * v for k, v in checks.items())
    total = max(0.0, min(1.0, total))

    breakdown = {'score': round(total, 4),
                 'constraints': {k: {'weight': WEIGHTS[k], 'score': round(v, 4),
                                     'weighted': round(WEIGHTS[k] * v, 4)}
                                 for k, v in checks.items()},
                 'gold_summary': {'retained_cells': len(gold['obs']),
                                  'donors': len(gold['donor_ids']),
                                  'genes_out': len(gold['var']),
                                  'report': gold['report']},
                 'agent_summary': {'obs_rows': len(ag), 'var_rows': (len(var_rows) - 1) if var_rows else 0}}
    print(json.dumps(breakdown, indent=2))

    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/report.json', 'w') as fh:
        json.dump(breakdown, fh, indent=2)
    with open('/logs/verifier/reward.json', 'w') as fh:
        json.dump({'score': round(total, 4), 'reward': round(total, 4)}, fh)


if __name__ == '__main__':
    main()
