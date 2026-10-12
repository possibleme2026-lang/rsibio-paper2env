#!/usr/bin/env python3
import json, os, sys, tempfile, subprocess, shutil
import numpy as np
import pandas as pd

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = '/app/output'
REWARD_FILE = '/logs/verifier/reward.json'
REPORT_FILE = '/logs/verifier/report.json'

def read_tsv(path, required_cols):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} does not exist')
    df = pd.read_csv(path, sep='\t')
    if list(df.columns) != required_cols:
        raise ValueError(f'{path} columns mismatch: expected {required_cols}, got {list(df.columns)}')
    return df

def compute_expected(data_dir):
    psm = pd.read_csv(os.path.join(data_dir, 'psm_table.tsv'), sep='\t')
    feat = pd.read_csv(os.path.join(data_dir, 'feature_table.tsv'), sep='\t')
    map_df = pd.read_csv(os.path.join(data_dir, 'peptide_protein_map.tsv'), sep='\t')
    unique_peps = set(map_df[map_df.mapping_type == 'unique'].peptide_sequence)

    psm_nonmbr = psm[(psm.is_mbr == 0) & (psm.peptide_sequence.isin(unique_peps))]
    spc = psm_nonmbr.groupby(['run', 'protein_id']).size().reset_index(name='spc')

    psm_auc = psm_nonmbr[['run', 'peptide_sequence', 'charge', 'protein_id']].drop_duplicates()
    merged = psm_auc.merge(feat, on=['run', 'peptide_sequence', 'charge'], how='inner')
    merged = merged.drop_duplicates(subset=['run', 'feature_id'])
    auc_nombr = merged.groupby(['run', 'protein_id'])['auc'].sum().reset_index()

    psm_mbr = psm[psm.peptide_sequence.isin(unique_peps)][['run', 'peptide_sequence', 'charge', 'protein_id']].drop_duplicates()
    merged_mbr = psm_mbr.merge(feat, on=['run', 'peptide_sequence', 'charge'], how='inner')
    merged_mbr = merged_mbr.drop_duplicates(subset=['run', 'feature_id'])
    auc_mbr = merged_mbr.groupby(['run', 'protein_id'])['auc'].sum().reset_index()

    mbr_prots = psm[psm.is_mbr == 1].groupby(['run', 'protein_id']).size().reset_index(name='mbr_psm_count')
    nonmbr_pairs = set(psm[psm.is_mbr == 0][['run', 'protein_id']].itertuples(index=False, name=None))
    if len(mbr_prots) > 0:
        mbr_prots['has_nonmbr'] = mbr_prots.apply(lambda r: (r['run'], r['protein_id']) in nonmbr_pairs, axis=1)
        fp = mbr_prots[~mbr_prots.has_nonmbr][['run', 'protein_id', 'mbr_psm_count']]
    else:
        fp = pd.DataFrame(columns=['run', 'protein_id', 'mbr_psm_count'])

    def dynamic_range(df, col):
        vals = df[df[col] > 0][col]
        if len(vals) < 2:
            return 0.0
        return float(np.log10(vals.max() / vals.min()))

    report = {
        'dynamic_range_auc': dynamic_range(auc_nombr, 'auc'),
        'dynamic_range_spc': dynamic_range(spc, 'spc'),
        'quantified_proteins_auc': int(len(auc_nombr)),
        'quantified_proteins_spc': int(len(spc)),
        'mbr_false_positive_proteins': int(len(fp)),
        'mbr_false_positive_psms': int(fp['mbr_psm_count'].sum()) if len(fp) > 0 else 0
    }

    return {
        'spc': spc.sort_values(['run', 'protein_id']).reset_index(drop=True),
        'auc_nombr': auc_nombr.sort_values(['run', 'protein_id']).reset_index(drop=True),
        'auc_mbr': auc_mbr.sort_values(['run', 'protein_id']).reset_index(drop=True),
        'fp': fp.sort_values(['run', 'protein_id']).reset_index(drop=True),
        'report': report
    }

def compare_tables(expected, got, key_cols, value_cols, tol=1e-6):
    exp = expected.copy()
    g = got.copy()
    for c in key_cols:
        if c not in g.columns:
            return 0.0, {'error': f'missing column {c}'}
    for c in key_cols:
        exp[c] = exp[c].astype(str)
        g[c] = g[c].astype(str)
    g = g.drop_duplicates(subset=key_cols, keep='first')
    merged = exp.merge(g, on=key_cols, how='outer', suffixes=('_exp', '_got'), indicator=True)
    correct = 0
    for _, row in merged.iterrows():
        if row['_merge'] != 'both':
            continue
        ok = True
        for v in value_cols:
            ev = row[f'{v}_exp']
            gv = row[f'{v}_got']
            if pd.isna(ev) or pd.isna(gv):
                ok = False
                break
            if isinstance(ev, (int, np.integer)) and isinstance(gv, (int, np.integer)):
                if int(ev) != int(gv):
                    ok = False
                    break
            else:
                if not np.isclose(float(ev), float(gv), rtol=tol, atol=tol):
                    ok = False
                    break
        if ok:
            correct += 1
    precision = correct / len(g) if len(g) > 0 else 0.0
    recall = correct / len(exp) if len(exp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, {'correct': correct, 'expected': len(exp), 'got': len(g), 'precision': precision, 'recall': recall}

def accuracy_score(quant_df, value_col, truth_A, truth_B, meta):
    df = quant_df.rename(columns={value_col: 'value'})
    df = df.merge(meta[['run', 'sample']], on='run', how='inner')
    df = df.groupby(['sample', 'protein_id']).value.mean().reset_index()
    scores = []
    for sample, true_dict in [('A', truth_A), ('B', truth_B)]:
        sub = df[df["sample"] == sample]
        if len(sub) < 10:
            scores.append(0.0)
            continue
        true_vals = [true_dict.get(p, np.nan) for p in sub.protein_id]
        valid = ~np.isnan(true_vals)
        if valid.sum() < 10:
            scores.append(0.0)
            continue
        obs = sub.value.values[valid]
        true_arr = np.array(true_vals)[valid]
        obs_log = np.log10(obs + 1e-9)
        if np.std(obs_log) == 0 or np.std(true_arr) == 0:
            scores.append(0.0)
        else:
            r = np.corrcoef(obs_log, true_arr)[0, 1]
            scores.append(max(0.0, r))
    return float(np.mean(scores))

def main():
    tmpdir = tempfile.mkdtemp()
    try:
        truth_file = os.path.join(tmpdir, 'truth.json')
        subprocess.run(['python', os.path.join(TESTS_DIR, 'gold_gen.py'), '--outdir', tmpdir, '--truth', truth_file], check=True)
        expected = compute_expected(tmpdir)
        with open(truth_file) as f:
            truth = json.load(f)
        meta = pd.read_csv(os.path.join(tmpdir, 'sample_metadata.tsv'), sep='\t')

        constraints = {}
        details = {}

        try:
            spc = read_tsv(os.path.join(OUT_DIR, 'protein_spc.tsv'), ['run', 'protein_id', 'spc'])
            auc_nombr = read_tsv(os.path.join(OUT_DIR, 'protein_auc_nombr.tsv'), ['run', 'protein_id', 'auc'])
            auc_mbr = read_tsv(os.path.join(OUT_DIR, 'protein_auc_mbr.tsv'), ['run', 'protein_id', 'auc'])
            fp = read_tsv(os.path.join(OUT_DIR, 'mbr_false_positive_proteins.tsv'), ['run', 'protein_id', 'mbr_psm_count'])
            with open(os.path.join(OUT_DIR, 'report.json')) as f:
                report = json.load(f)
            for name, df in [('spc', spc), ('auc_nombr', auc_nombr), ('auc_mbr', auc_mbr), ('fp', fp)]:
                if df.duplicated(subset=['run', 'protein_id']).any():
                    raise ValueError(f'{name} has duplicate run/protein_id pairs')
            constraints['format'] = 1.0
        except Exception as e:
            constraints['format'] = 0.0
            details['format_error'] = str(e)
            score = 0.0
            out = {'score': score, 'reward': score}
            os.makedirs(os.path.dirname(REWARD_FILE), exist_ok=True)
            with open(REWARD_FILE, 'w') as f:
                json.dump(out, f)
            with open(REPORT_FILE, 'w') as f:
                json.dump({'constraints': constraints, 'details': details}, f, indent=2)
            print(json.dumps({'constraints': constraints, 'details': details}, indent=2))
            return

        f1, d = compare_tables(expected['spc'], spc, ['run', 'protein_id'], ['spc'], tol=0)
        constraints['spc'] = f1
        details['spc'] = d

        f1, d = compare_tables(expected['auc_nombr'], auc_nombr, ['run', 'protein_id'], ['auc'], tol=1e-6)
        constraints['auc_nombr'] = f1
        details['auc_nombr'] = d

        f1, d = compare_tables(expected['auc_mbr'], auc_mbr, ['run', 'protein_id'], ['auc'], tol=1e-6)
        constraints['auc_mbr'] = f1
        details['auc_mbr'] = d

        f1, d = compare_tables(expected['fp'], fp, ['run', 'protein_id'], ['mbr_psm_count'], tol=0)
        constraints['mbr_false_positive'] = f1
        details['mbr_false_positive'] = d

        report_fields = ['dynamic_range_auc', 'dynamic_range_spc', 'quantified_proteins_auc', 'quantified_proteins_spc', 'mbr_false_positive_proteins', 'mbr_false_positive_psms']
        correct = 0
        report_errors = []
        if isinstance(report, dict):
            for field in report_fields:
                if field not in report:
                    report_errors.append(f'missing {field}')
                    continue
                ev = expected['report'][field]
                gv = report[field]
                if isinstance(ev, float):
                    if not np.isclose(float(gv), ev, rtol=1e-6, atol=1e-6):
                        report_errors.append(f'{field}: got {gv}, expected {ev}')
                    else:
                        correct += 1
                else:
                    if int(gv) != int(ev):
                        report_errors.append(f'{field}: got {gv}, expected {ev}')
                    else:
                        correct += 1
        else:
            report_errors.append('report.json is not a dict')
        constraints['report'] = correct / len(report_fields)
        details['report_errors'] = report_errors

        auc_acc = accuracy_score(auc_nombr, 'auc', truth['true_A'], truth['true_B'], meta)
        spc_acc = accuracy_score(spc, 'spc', truth['true_A'], truth['true_B'], meta)
        acc_score = 0.5 * min(1.0, auc_acc / 0.8) + 0.5 * min(1.0, spc_acc / 0.4)
        constraints['biological_accuracy'] = acc_score
        details['biological_accuracy'] = {'auc_correlation': auc_acc, 'spc_correlation': spc_acc}

        weights = {
            'format': 1.0,
            'spc': 1.0,
            'auc_nombr': 1.0,
            'auc_mbr': 1.0,
            'mbr_false_positive': 1.0,
            'report': 1.0,
            'biological_accuracy': 1.0
        }
        total_weight = sum(weights.values())
        score = sum(constraints.get(k, 0.0) * weights[k] for k in weights) / total_weight
        out = {'score': float(score), 'reward': float(score)}
        os.makedirs(os.path.dirname(REWARD_FILE), exist_ok=True)
        with open(REWARD_FILE, 'w') as f:
            json.dump(out, f)
        report_out = {'constraints': constraints, 'details': details, 'final_score': score}
        os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
        with open(REPORT_FILE, 'w') as f:
            json.dump(report_out, f, indent=2)
        print(json.dumps(report_out, indent=2))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == '__main__':
    main()
