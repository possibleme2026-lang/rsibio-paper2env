import os, math, json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

def reference_analysis(data_dir):
    lipid = pd.read_csv(os.path.join(data_dir, 'lipidomics.csv'))
    meta = pd.read_csv(os.path.join(data_dir, 'sample_metadata.csv'))
    db = pd.read_csv(os.path.join(data_dir, 'lipid_reference_db.csv'))
    activity = pd.read_csv(os.path.join(data_dir, 'ev_activity.csv'))
    endo = pd.read_csv(os.path.join(data_dir, 'endothelial_response.csv'))
    ev_samples = meta[meta['sample_type'] == 'EV']['sample_id'].tolist()
    ctrl = [s for s in ev_samples if s.startswith('Control')]
    bl = [s for s in ev_samples if s.startswith('BL')]
    lps = [s for s in ev_samples if s.startswith('LPS')]
    qc = meta[meta['sample_type'] == 'QC']['sample_id'].tolist()
    blank = meta[meta['sample_type'] == 'Blank']['sample_id'].tolist()
    def annotate(row):
        candidates = db[db['adduct'] == row['adduct']].copy()
        if len(candidates) == 0:
            return None
        ppm = (abs(candidates['exact_mz'] - row['mz']) / candidates['exact_mz'] * 1e6)
        candidates = candidates[ppm <= 10]
        if len(candidates) == 0:
            return None
        candidates['ppm'] = ppm.loc[candidates.index]
        candidates['rt_diff'] = abs(candidates['rt_min'] - row['rt'])
        candidates = candidates[candidates['rt_diff'] <= 0.3]
        if len(candidates) == 0:
            return None
        best = candidates.sort_values('ppm').iloc[0]
        return best['lipid_class']
    lipid['lipid_class'] = lipid.apply(annotate, axis=1)
    cond_cols = {'Control': ctrl, 'BL': bl, 'LPS': lps}
    keep_missing = []
    for idx, row in lipid.iterrows():
        ok = False
        for cond, cols in cond_cols.items():
            if row[cols].notna().sum() >= 4:
                ok = True
                break
        keep_missing.append(ok)
    lipid = lipid[keep_missing].copy()
    qc_vals = lipid[qc]
    mean_qc = qc_vals.mean(axis=1)
    sd_qc = qc_vals.std(axis=1)
    cv = sd_qc / mean_qc * 100
    lipid = lipid[(cv <= 30) | (mean_qc == 0)].copy()
    keep_blank = []
    for idx, row in lipid.iterrows():
        bm = row[blank].mean()
        ok = any(row[cols].mean() > 3 * bm for cols in cond_cols.values())
        keep_blank.append(ok)
    lipid = lipid[keep_blank].copy()
    all_ev = ctrl + bl + lps
    for col in all_ev:
        lipid[col] = np.log2(lipid[col] + 1)
    for col in all_ev:
        med = lipid[col].median()
        lipid[col] = lipid[col] - med
    def diff(cols_case, cols_ctrl):
        case_vals = lipid[cols_case]
        ctrl_vals = lipid[cols_ctrl]
        log2fc = case_vals.median(axis=1) - ctrl_vals.median(axis=1)
        tstats, pvals = stats.ttest_ind(case_vals, ctrl_vals, axis=1, equal_var=False, nan_policy='omit')
        pvals = np.nan_to_num(pvals, nan=1.0)
        fdr = multipletests(pvals, method='fdr_bh')[1]
        return log2fc, fdr
    log2fc_bl, fdr_bl = diff(bl, ctrl)
    log2fc_lps, fdr_lps = diff(lps, ctrl)
    lipid['log2FC_BL_vs_Control'] = log2fc_bl
    lipid['fdr_BL'] = fdr_bl
    lipid['log2FC_LPS_vs_Control'] = log2fc_lps
    lipid['fdr_LPS'] = fdr_lps
    sig = []
    for idx, row in lipid.iterrows():
        bl_en = row['log2FC_BL_vs_Control'] >= 1 and row['fdr_BL'] <= 0.05 and row['lipid_class'] in ['SaturatedFattyAcid', 'Ceramide']
        lps_en = row['log2FC_LPS_vs_Control'] >= 1 and row['fdr_LPS'] <= 0.05
        if bl_en and lps_en:
            sig.append('shared')
        elif bl_en:
            sig.append('BL_enriched')
        elif lps_en:
            sig.append('LPS_enriched')
        else:
            sig.append('none')
    lipid['signature_group'] = sig
    ctrl_act = activity[activity['condition'] == 'Control']
    mean_nit = ctrl_act['nitrite_uM'].mean()
    sd_nit = ctrl_act['nitrite_uM'].std()
    mean_inos = ctrl_act['iNOS_band_intensity'].mean()
    sd_inos = ctrl_act['iNOS_band_intensity'].std()
    iNOS_positive = []
    for cond in ['Control', 'BL', 'LPS']:
        sub = activity[activity['condition'] == cond]
        med_nit = sub['nitrite_uM'].median()
        med_inos = sub['iNOS_band_intensity'].median()
        if med_nit > mean_nit + 2*sd_nit and med_inos > mean_inos + 2*sd_inos:
            iNOS_positive.append(cond)
    endo_ctrl = endo[endo['condition'] == 'Control']
    endo_res = {'IL6': [], 'CCL5': [], 'ICAM1': []}
    pvals = []
    pairs = []
    for cyto in ['IL6_pg_ml', 'CCL5_pg_ml', 'ICAM1_od']:
        for cond in ['BL', 'LPS']:
            case = endo[endo['condition'] == cond][cyto]
            ctrl_vals = endo_ctrl[cyto]
            tstat, pval = stats.ttest_ind(case, ctrl_vals, equal_var=False, nan_policy='omit')
            log2fc = np.log2(case.median() + 1) - np.log2(ctrl_vals.median() + 1)
            pvals.append(pval)
            pairs.append((cyto, cond, log2fc))
    fdrs = multipletests(pvals, method='fdr_bh')[1]
    for (cyto, cond, log2fc), fdr in zip(pairs, fdrs):
        if fdr <= 0.05 and log2fc >= 1:
            key = cyto.replace('_pg_ml', '').replace('_od', '')
            endo_res[key].append(cond)
    summary = {
        'iNOS_positive_conditions': iNOS_positive,
        'nitrite_median_by_condition': {cond: float(activity[activity['condition']==cond]['nitrite_uM'].median()) for cond in ['Control','BL','LPS']},
        'endothelial_induction': endo_res,
        'signature_counts': {k: int(v) for k,v in lipid['signature_group'].value_counts().items()}
    }
    ref = {
        'lipid_signature': lipid[['feature_id', 'lipid_class', 'log2FC_BL_vs_Control', 'fdr_BL', 'log2FC_LPS_vs_Control', 'fdr_LPS', 'signature_group']].copy(),
        'summary': summary
    }
    return ref
