#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python - <<'PY'
import pandas as pd, numpy as np, json
from scipy.optimize import curve_fit
viab = pd.read_csv('/app/data/viability_matrix.csv')
wb = pd.read_csv('/app/data/western_blot.csv')
mean_viab = viab.groupby(['cell_line','lurbinectedin_nM','prexasertib_nM'])['viability'].mean().reset_index()
hsa_rows = []
for cl in mean_viab['cell_line'].unique():
    sub = mean_viab[mean_viab['cell_line'] == cl]
    L_map = dict(zip(sub[sub['prexasertib_nM'] == 0]['lurbinectedin_nM'], sub[sub['prexasertib_nM'] == 0]['viability']))
    P_map = dict(zip(sub[sub['lurbinectedin_nM'] == 0]['prexasertib_nM'], sub[sub['lurbinectedin_nM'] == 0]['viability']))
    for _, row in sub.iterrows():
        dL, dP = row['lurbinectedin_nM'], row['prexasertib_nM']
        expected = min(L_map[dL], P_map[dP])
        score = expected - row['viability']
        call = 'synergistic' if score > 0.05 else ('antagonistic' if score < -0.05 else 'additive')
        hsa_rows.append([cl, dL, dP, row['viability'], expected, score, call])
hsa = pd.DataFrame(hsa_rows, columns=['cell_line','lurbinectedin_nM','prexasertib_nM','mean_viability','expected_hsa','hsa_score','call'])
hsa.to_csv('/app/output/hsa_scores.csv', index=False)
ic50_rows = []
for cl in mean_viab['cell_line'].unique():
    for drug in ['lurbinectedin','prexasertib']:
        if drug == 'lurbinectedin':
            sub = mean_viab[(mean_viab['cell_line'] == cl) & (mean_viab['prexasertib_nM'] == 0)]
            doses, viabs = sub['lurbinectedin_nM'].values, sub['viability'].values
        else:
            sub = mean_viab[(mean_viab['cell_line'] == cl) & (mean_viab['lurbinectedin_nM'] == 0)]
            doses, viabs = sub['prexasertib_nM'].values, sub['viability'].values
        mask = doses > 0
        x = np.log10(doses[mask])
        y = viabs[mask]
        def model(x, bottom, top, ic50, hill):
            return bottom + (top - bottom) / (1.0 + (10.0 ** (x - np.log10(ic50))) ** hill)
        p0 = [max(0.0, min(y)), min(1.2, max(y)), np.median(doses[mask]), 1.0]
        bounds = ([0.0, 0.5, 0.01, 0.1], [0.5, 1.2, 1000.0, 5.0])
        popt, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=10000)
        y_pred = model(x, *popt)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        ic50_rows.append([cl, drug, popt[2], popt[3], popt[0], popt[1], r2])
ic50 = pd.DataFrame(ic50_rows, columns=['cell_line','drug','ic50_nM','hill','bottom','top','r2'])
ic50.to_csv('/app/output/ic50_summary.csv', index=False)
wb_wide = wb.pivot_table(index=['cell_line','condition','replicate'], columns='protein', values='intensity').reset_index()
for prot in wb_wide.columns:
    if prot in ['cell_line','condition','replicate','beta-actin']:
        continue
    wb_wide[f'norm_{prot}'] = wb_wide[prot] / wb_wide['beta-actin']
mech_rows = []
for cl in wb_wide['cell_line'].unique():
    for prot in ['pChk1_S296','pChk1_S317','pChk1_S345','pChk2_Y68','Chk1','Chk2','Wee1','gammaH2AX','cleaved_PARP']:
        for cond in ['vehicle','L_1nM','L_10nM','P_30nM','P_300nM','L_10nM_P_300nM']:
            sub = wb_wide[(wb_wide['cell_line'] == cl) & (wb_wide['condition'] == cond)]
            median_norm = sub[f'norm_{prot}'].median()
            veh_sub = wb_wide[(wb_wide['cell_line'] == cl) & (wb_wide['condition'] == 'vehicle')]
            veh_median = veh_sub[f'norm_{prot}'].median()
            fold = median_norm / veh_median if veh_median > 0 else 0.0
            mech_rows.append([cl, prot, cond, fold])
mech = pd.DataFrame(mech_rows, columns=['cell_line','protein','condition','median_fold_change'])
mech.to_csv('/app/output/mechanism_summary.csv', index=False)
synergy_counts = hsa[hsa['call'] == 'synergistic'].groupby('cell_line').size()
hsa_synergy_cell_lines = sorted(list(synergy_counts[synergy_counts >= 3].index))
max_hsa_score = hsa.groupby('cell_line')['hsa_score'].max().to_dict()
ic50_nM = {}
for cl in mean_viab['cell_line'].unique():
    ic50_nM[cl] = {}
    for drug in ['lurbinectedin','prexasertib']:
        val = ic50[(ic50['cell_line'] == cl) & (ic50['drug'] == drug)]['ic50_nM'].values[0]
        ic50_nM[cl][drug] = float(val)
mechanism = {}
for cl in mean_viab['cell_line'].unique():
    l10 = mech[(mech['cell_line'] == cl) & (mech['condition'] == 'L_10nM')]
    pchk1_sites = ['pChk1_S296','pChk1_S317','pChk1_S345']
    pchk1_up = sum(l10[l10['protein'] == s]['median_fold_change'].values[0] > 1.5 for s in pchk1_sites)
    pchk2_up = l10[l10['protein'] == 'pChk2_Y68']['median_fold_change'].values[0] > 1.5
    pchk_up = (pchk1_up >= 2) or pchk2_up
    combo = mech[(mech['cell_line'] == cl) & (mech['condition'] == 'L_10nM_P_300nM')]
    suppressed = 0
    for prot in ['pChk1_S296','pChk1_S317','pChk1_S345','pChk2_Y68']:
        combo_val = combo[combo['protein'] == prot]['median_fold_change'].values[0]
        l10_val = l10[l10['protein'] == prot]['median_fold_change'].values[0]
        if combo_val < 0.7 * l10_val:
            suppressed += 1
    combo_supp = suppressed >= 2
    gh2ax = combo[combo['protein'] == 'gammaH2AX']['median_fold_change'].values[0]
    parp = combo[combo['protein'] == 'cleaved_PARP']['median_fold_change'].values[0]
    damage = (gh2ax > 2.0) and (parp > 1.5)
    mechanism[cl] = {'pchk_upregulated_by_lurbinectedin': bool(pchk_up), 'combination_suppresses_pchk': bool(combo_supp), 'dna_damage_apoptosis': bool(damage)}
report = {'hsa_synergy_cell_lines': hsa_synergy_cell_lines, 'max_hsa_score': {k: float(v) for k, v in max_hsa_score.items()}, 'ic50_nM': ic50_nM, 'mechanism': mechanism}
with open('/app/output/report.json','w') as f:
    json.dump(report, f, indent=2)
print('done')
PY
