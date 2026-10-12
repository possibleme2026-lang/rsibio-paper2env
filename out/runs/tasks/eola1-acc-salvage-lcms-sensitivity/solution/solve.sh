#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python3 - <<'PY'
import json, re
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from pyteomics import mass as pm

D, O = '/app/data', '/app/output'
PROTON = 1.007276466621
ELEC = 0.000548579909
SHIFT = {'[M+H]+': PROTON, '[M+Na]+': 22.9897692809 - ELEC,
         '[M+NH4]+': 14.0030740048 + 4 * 1.00782503207 - ELEC}

# 1. channel annotation -------------------------------------------------
ch = pd.read_csv(f'{D}/lcms_channels.csv')
lib = pd.read_csv(f'{D}/compound_library.csv')
ann, rows = {}, []
for _, c in ch.iterrows():
    hit = None
    for _, l in lib.iterrows():
        base = pm.calculate_mass(formula=l['formula'])
        for a, s in SHIFT.items():
            mz = base + s
            ppm = (c['channel_mz'] - mz) / mz * 1e6
            if abs(ppm) <= 10 and abs(c['rt_min'] - l['expected_rt_min']) <= 0.3:
                if hit is None or abs(ppm) < abs(hit[2]):
                    hit = (l['compound'], a, ppm)
    if hit:
        ann[round(float(c['channel_mz']), 4)] = hit[0]
        rows.append({'channel_mz': c['channel_mz'], 'rt_min': c['rt_min'],
                     'compound': hit[0], 'adduct': hit[1], 'ppm_error': round(hit[2], 2)})
    else:
        rows.append({'channel_mz': c['channel_mz'], 'rt_min': c['rt_min'],
                     'compound': 'unassigned', 'adduct': 'none', 'ppm_error': ''})
pd.DataFrame(rows).to_csv(f'{O}/channel_annotation.csv', index=False)

# 2. calibration --------------------------------------------------------
cal = pd.read_csv(f'{D}/lcms_calibration.csv')
fits = {}
for mz, g in cal.groupby('channel_mz'):
    a, b = np.polyfit(g['conc_nM'], g['area'], 1)
    fits[round(float(mz), 4)] = (a, b)

sm = pd.read_csv(f'{D}/lcms_samples.csv')
meta = pd.read_csv(f'{D}/lcms_sample_metadata.csv')
sm['mz4'] = sm['channel_mz'].round(4)
sm['analyte'] = sm['mz4'].map(ann)
sm = sm.dropna(subset=['analyte'])
# only channels covered by the external calibration can be
# quantified (the file calibrates 5 of the 8 channels by design)
sm = sm[sm['mz4'].isin(fits.keys())]
sm['conc_nM'] = [(r['area'] - fits[r['mz4']][1]) / fits[r['mz4']][0] for _, r in sm.iterrows()]
sm = sm.merge(meta[['sample_id', 'protein_mg', 'extract_volume_mL']], on='sample_id')
sm['pmol_per_mg'] = sm['conc_nM'] * sm['extract_volume_mL'] / sm['protein_mg']

# 3. per cell line ------------------------------------------------------
lines = ['WT', 'EOLA1-KO1', 'EOLA1-KO2', 'KO1+EOLA1']
analytes = sorted(sm['analyte'].unique())
summary_rows = []
means, sds = {}, {}
for line in lines:
    for ana in analytes:
        v = sm[(sm.cell_line == line) & (sm.analyte == ana)]['pmol_per_mg'].values
        means[(line, ana)] = float(np.mean(v))
        sds[(line, ana)] = float(np.std(v, ddof=1))
for ana in analytes:
    pvals = {}
    for line in lines[1:]:
        a = sm[(sm.cell_line == line) & (sm.analyte == ana)]['pmol_per_mg'].values
        w = sm[(sm.cell_line == 'WT') & (sm.analyte == ana)]['pmol_per_mg'].values
        pvals[line] = float(stats.ttest_ind(a, w, equal_var=False).pvalue)
    order = sorted(pvals, key=lambda k: pvals[k])
    adj = {}
    m = len(order)
    prev = 1.0
    for rank in range(m, 0, -1):
        k = order[rank - 1]
        prev = min(prev, pvals[k] * m / rank)
        adj[k] = prev
    for line in lines:
        mean = means[(line, ana)]
        if line == 'WT':
            fc, p, sig = 1.0, 1.0, False
        else:
            fc = mean / means[('WT', ana)]
            p = pvals[line]
            sig = bool(adj[line] < 0.05 and abs(np.log2(fc)) >= 1)
        summary_rows.append({'cell_line': line, 'analyte': ana,
                             'mean_pmol_per_mg': round(mean, 3),
                             'sd_pmol_per_mg': round(sds[(line, ana)], 3),
                             'fold_change_vs_WT': round(fc, 3), 'ttest_p': float(p),
                             'significant': str(sig).lower()})
pd.DataFrame(summary_rows).to_csv(f'{O}/nucleoside_quantification.csv', index=False)

# 4. dose-response plates ----------------------------------------------
def logistic(logx, bottom, top, log_ic50, hill):
    return bottom + (top - bottom) / (1.0 + 10 ** ((log_ic50 - logx) * hill))

def fit_curve(x, y):
    p0 = [5.0, 100.0, float(np.log10(np.median(x))), 1.0]
    popt, _ = curve_fit(logistic, np.log10(x), y, p0=p0, maxfev=20000)
    resid = y - logistic(np.log10(x), *popt)
    r2 = 1.0 - np.sum(resid ** 2) / np.sum((y - np.mean(y)) ** 2)
    return popt, r2

p1 = pd.read_csv(f'{D}/viability_dose_response.csv')
rows = []
for line in lines:
    sub = p1[p1.cell_line == line]
    popt, r2 = fit_curve(sub['dipyridamole_uM'].values, sub['viability_pct'].values)
    rows.append({'dataset': 'dose_response', 'cell_line': line, 'treatment': 'dipyridamole',
                 'ic50_uM': round(10 ** popt[2], 3), 'hill_slope': round(popt[3], 3),
                 'r_squared': round(r2, 4)})
p2 = pd.read_csv(f'{D}/viability_cytidine_rescue.csv')
for treat in ['dipyridamole', 'dipyridamole+cytidine']:
    sub = p2[p2.treatment == treat]
    popt, r2 = fit_curve(sub['dipyridamole_uM'].values, sub['viability_pct'].values)
    rows.append({'dataset': 'cytidine_rescue', 'cell_line': 'EOLA1-KO1', 'treatment': treat,
                 'ic50_uM': round(10 ** popt[2], 3), 'hill_slope': round(popt[3], 3),
                 'r_squared': round(r2, 4)})
ic = pd.DataFrame(rows)
ic.to_csv(f'{O}/ic50_summary.csv', index=False)

# 5. summary ------------------------------------------------------------
def ic_lookup(dataset, line, treat):
    m = ic[(ic.dataset == dataset) & (ic.cell_line == line) & (ic.treatment == treat)]
    return float(m['ic50_uM'].iloc[0])

wt = ic_lookup('dose_response', 'WT', 'dipyridamole')
ko1 = ic_lookup('dose_response', 'EOLA1-KO1', 'dipyridamole')
ko2 = ic_lookup('dose_response', 'EOLA1-KO2', 'dipyridamole')
resc = ic_lookup('dose_response', 'KO1+EOLA1', 'dipyridamole')
dip = ic_lookup('cytidine_rescue', 'EOLA1-KO1', 'dipyridamole')
dipc = ic_lookup('cytidine_rescue', 'EOLA1-KO1', 'dipyridamole+cytidine')
acc_ko1 = means[('EOLA1-KO1', 'N4-acetylcytidine')] / means[('WT', 'N4-acetylcytidine')]
acc_ko2 = means[('EOLA1-KO2', 'N4-acetylcytidine')] / means[('WT', 'N4-acetylcytidine')]
acc_resc = means[('KO1+EOLA1', 'N4-acetylcytidine')] / means[('WT', 'N4-acetylcytidine')]
q = pd.read_csv(f'{O}/nucleoside_quantification.csv')
other = q[(q.analyte != 'N4-acetylcytidine') & (q.cell_line != 'WT')]
other_ok = not bool(other['significant'].map(lambda v: str(v).lower() == 'true').any())
out = {'n_channels_annotated': int(len(ann)),
       'n_channels_unassigned': 8 - int(len(ann)),
       'free_acC_fold_change_KO1_vs_WT': round(acc_ko1, 3),
       'free_acC_fold_change_KO2_vs_WT': round(acc_ko2, 3),
       'free_acC_restored_by_reexpression': bool(acc_resc <= 1.5),
       'other_nucleosides_unchanged': other_ok,
       'ic50_uM_WT': wt, 'ic50_uM_EOLA1_KO1': ko1, 'ic50_uM_EOLA1_KO2': ko2,
       'ic50_uM_KO1_reexpressed': resc,
       'ic50_uM_KO1_dipyridamole': dip,
       'ic50_uM_KO1_dipyridamole_cytidine': dipc,
       'cytidine_rescue_index': round(dipc / dip, 3),
       'cytidine_rescues_viability': bool(dipc / dip >= 2),
       'EOLA1_loss_sensitises_to_dipyridamole': bool(ko1 <= 0.5 * wt and ko2 <= 0.5 * wt)}
with open(f'{O}/summary.json', 'w') as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out, indent=2))
PY
