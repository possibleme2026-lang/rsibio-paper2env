#!/usr/bin/env python3
import argparse, os, math, random
import numpy as np
import pandas as pd

def generate(seed, outdir):
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(outdir, exist_ok=True)
    conditions = ['Control', 'BL', 'LPS']
    n_per = 5
    sample_ids = []
    meta = []
    for cond in conditions:
        for rep in range(1, n_per+1):
            sid = f'{cond}_{rep}'
            sample_ids.append(sid)
            meta.append({'sample_id': sid, 'condition': cond, 'sample_type': 'EV',
                         'particle_count_per_uL': np.random.normal(5e8, 5e7),
                         'protein_ug': np.random.normal(20, 3)})
    qc_ids = [f'QC_{i}' for i in range(1,4)]
    blank_ids = [f'Blank_{i}' for i in range(1,4)]
    for sid in qc_ids:
        meta.append({'sample_id': sid, 'condition': 'QC', 'sample_type': 'QC', 'particle_count_per_uL': np.nan, 'protein_ug': np.nan})
    for sid in blank_ids:
        meta.append({'sample_id': sid, 'condition': 'Blank', 'sample_type': 'Blank', 'particle_count_per_uL': np.nan, 'protein_ug': np.nan})
    pd.DataFrame(meta).to_csv(os.path.join(outdir, 'sample_metadata.csv'), index=False)

    n_features = 1200
    feature_ids = [f'LIP{i:04d}' for i in range(1, n_features+1)]
    classes = ['SaturatedFattyAcid', 'Ceramide', 'Phosphatidylcholine', 'Phosphatidylethanolamine',
               'Phosphatidylserine', 'Sphingomyelin', 'Triacylglycerol', 'Diacylglycerol',
               'CholesterolEster', 'FreeCholesterol']
    true_sig = {}
    bl_idx = np.random.choice(n_features, 150, replace=False)
    for i in bl_idx:
        cls = np.random.choice(['SaturatedFattyAcid', 'Ceramide'])
        true_sig[feature_ids[i]] = ('BL_enriched', cls)
    remaining = [i for i in range(n_features) if i not in bl_idx]
    lps_idx = np.random.choice(remaining, 100, replace=False)
    for i in lps_idx:
        cls = np.random.choice([c for c in classes if c not in ['SaturatedFattyAcid', 'Ceramide']])
        true_sig[feature_ids[i]] = ('LPS_enriched', cls)
    remaining = [i for i in remaining if i not in lps_idx]
    shared_idx = np.random.choice(remaining, 30, replace=False)
    for i in shared_idx:
        cls = np.random.choice(classes)
        true_sig[feature_ids[i]] = ('shared', cls)
    for i in range(n_features):
        fid = feature_ids[i]
        if fid not in true_sig:
            cls = np.random.choice(classes)
            true_sig[fid] = ('none', cls)

    data = {'feature_id': feature_ids,
            'mz': np.random.uniform(200, 1000, n_features).round(4),
            'rt': np.random.uniform(1, 15, n_features).round(3),
            'adduct': np.random.choice(['[M+H]+', '[M+Na]+', '[M-H]-'], n_features)}
    all_samples = sample_ids + qc_ids + blank_ids
    for sid in all_samples:
        data[sid] = np.zeros(n_features)
    for i, fid in enumerate(feature_ids):
        baseline = np.random.normal(15, 1.0)
        sample_eff = {sid: np.random.normal(0, 0.2) for sid in all_samples}
        sig, cls = true_sig[fid]
        for sid in all_samples:
            val = baseline + sample_eff[sid]
            cond = sid.split('_')[0] if '_' in sid else sid
            if sid in sample_ids:
                if sig in ('BL_enriched', 'shared') and cond == 'BL':
                    val += np.random.uniform(1.5, 2.5)
                if sig in ('LPS_enriched', 'shared') and cond == 'LPS':
                    val += np.random.uniform(1.5, 2.5)
                if sig == 'BL_enriched' and cond == 'LPS':
                    val += np.random.uniform(-0.2, 0.2)
                if sig == 'LPS_enriched' and cond == 'BL':
                    val += np.random.uniform(-0.2, 0.2)
            elif sid in qc_ids:
                pass
            elif sid in blank_ids:
                if i < 50:
                    val -= np.random.normal(2, 0.5)
                else:
                    val -= np.random.normal(6, 1.0)
            data[sid][i] = val
    for sid in all_samples:
        data[sid] = data[sid] + np.random.normal(0, 0.15, n_features)
    high_cv_idx = np.random.choice(n_features, 100, replace=False)
    for i in high_cv_idx:
        for sid in qc_ids:
            data[sid][i] += np.random.normal(0, 0.8)
    for sid in all_samples:
        miss_idx = np.random.choice(n_features, int(n_features*0.01), replace=False)
        for i in miss_idx:
            data[sid][i] = np.nan
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(outdir, 'lipidomics.csv'), index=False)

    db_rows = []
    for i, fid in enumerate(feature_ids):
        sig, cls = true_sig[fid]
        mz = df.loc[i, 'mz']
        rt = df.loc[i, 'rt']
        adduct = df.loc[i, 'adduct']
        formula = f'C{np.random.randint(20,60)}H{np.random.randint(30,100)}O{np.random.randint(2,10)}'
        db_rows.append({'lipid_id': fid, 'lipid_class': cls, 'lipid_name': f'{cls}_{i+1}',
                        'formula': formula, 'exact_mz': mz + np.random.normal(0, 0.0001),
                        'adduct': adduct, 'rt_min': rt + np.random.normal(0, 0.05)})
    for j in range(300):
        mz = np.random.uniform(200, 1000)
        rt = np.random.uniform(1, 15)
        cls = np.random.choice(classes)
        db_rows.append({'lipid_id': f'DECOY{j:04d}', 'lipid_class': cls, 'lipid_name': f'Decoy_{j}',
                        'formula': 'C0H0O0', 'exact_mz': mz, 'adduct': np.random.choice(['[M+H]+', '[M+Na]+', '[M-H]-']),
                        'rt_min': rt})
    pd.DataFrame(db_rows).to_csv(os.path.join(outdir, 'lipid_reference_db.csv'), index=False)

    act_rows = []
    for sid in sample_ids:
        cond = sid.split('_')[0]
        if cond == 'Control':
            inos = np.random.normal(0.2, 0.05)
            nitrite = np.random.normal(2, 0.5)
            tnf = np.random.normal(10, 2)
        elif cond == 'BL':
            inos = np.random.normal(1.5, 0.2)
            nitrite = np.random.normal(18, 3)
            tnf = np.random.normal(45, 8)
        else:
            inos = np.random.normal(1.8, 0.2)
            nitrite = np.random.normal(22, 4)
            tnf = np.random.normal(55, 10)
        particle = np.random.normal(5e8, 5e7)
        act_rows.append({'sample_id': sid, 'condition': cond, 'particle_count_per_uL': particle,
                         'protein_ug': np.random.normal(20, 3), 'iNOS_band_intensity': max(0, inos),
                         'nitrite_uM': max(0, nitrite), 'TNF_pg_ml': max(0, tnf)})
    pd.DataFrame(act_rows).to_csv(os.path.join(outdir, 'ev_activity.csv'), index=False)

    endo_rows = []
    for sid in sample_ids:
        cond = sid.split('_')[0]
        if cond == 'Control':
            il6 = np.random.normal(50, 10)
            ccl5 = np.random.normal(80, 15)
            icam = np.random.normal(0.3, 0.05)
        elif cond == 'BL':
            il6 = np.random.normal(60, 12)
            ccl5 = np.random.normal(90, 18)
            icam = np.random.normal(0.35, 0.06)
        else:
            il6 = np.random.normal(300, 40)
            ccl5 = np.random.normal(400, 50)
            icam = np.random.normal(1.5, 0.2)
        endo_rows.append({'sample_id': sid, 'condition': cond, 'IL6_pg_ml': max(0, il6),
                          'CCL5_pg_ml': max(0, ccl5), 'ICAM1_od': max(0, icam)})
    pd.DataFrame(endo_rows).to_csv(os.path.join(outdir, 'endothelial_response.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20251211)
    parser.add_argument('--outdir', default='/app/data')
    args = parser.parse_args()
    generate(args.seed, args.outdir)
