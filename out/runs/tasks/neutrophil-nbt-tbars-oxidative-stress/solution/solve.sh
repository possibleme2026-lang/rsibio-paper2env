#!/bin/bash
set -euo pipefail
mkdir -p /app/output
cat > /app/run_pipeline.py <<'PY'
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

DATA = '/app/data'
params = json.load(open(DATA + '/assay_parameters.json'))
PX = float(params['pixel_size_um'])
DIL = float(params['tbars_dilution_factor'])

meta = pd.read_csv(DATA + '/subject_metadata.csv', dtype={'subject_id': str})
cells = pd.read_csv(DATA + '/nbt_cell_measurements.csv', dtype={'subject_id': str, 'field_id': str})
fields = pd.read_csv(DATA + '/field_qc.csv', dtype={'subject_id': str, 'field_id': str})
readings = pd.read_csv(DATA + '/tbars_plate_readings.csv', dtype={'plate_id': str, 'well': str})
wmap = pd.read_csv(DATA + '/tbars_well_map.csv', dtype={'plate_id': str, 'well': str, 'sample_id': str})

merged = cells.merge(fields[['subject_id', 'field_id', 'focus_score']],
                     on=['subject_id', 'field_id'], how='left')
n_cells_measured = int(len(merged))
n_fields_measured = int(len(fields))
focus_ok = merged['focus_score'] >= 0.60
n_exc_focus = int((~focus_ok).sum())
s1 = merged[focus_ok]
class_ok = s1['object_class'] == 'neutrophil'
n_exc_class = int((~class_ok).sum())
s2 = s1[class_ok]
area_ok = s2['cell_area_px'] >= 200.0
n_exc_area = int((~area_ok).sum())
kept = s2[area_ok].copy()
kept['area_um2'] = kept['formazan_area_px'] * PX ** 2
kept['positive'] = kept['area_um2'] >= 25.0
g = kept.groupby('subject_id')
per = pd.DataFrame({'n_cells_retained': g.size(),
                    'nbt_positive_pct': g['positive'].mean() * 100.0,
                    'nbt_reactivity': g['area_um2'].mean()})
n_cells_retained = int(len(kept))
n_fields_retained = int(kept.groupby(['subject_id', 'field_id']).ngroups)

r = readings.merge(wmap, on=['plate_id', 'well'], how='left')
r['a_corr'] = r['absorbance_532'] - r['absorbance_600']
curves = {}
for plate, sub in r[r['well_type'] == 'standard'].groupby('plate_id'):
    x = sub['mda_nmol_per_ml'].astype(float).to_numpy()
    y = sub['a_corr'].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    r2 = 1.0 - float(((y - yhat) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
    curves[str(plate)] = {'slope': float(slope), 'intercept': float(intercept), 'r_squared': r2}
samp = r[r['well_type'] == 'sample'].copy()
n_sat = int((samp['absorbance_532'] > 2.0).sum())
samp = samp[samp['absorbance_532'] <= 2.0]
vals = []
for row in samp.itertuples():
    cv = curves[str(row.plate_id)]
    vals.append((row.a_corr - cv['intercept']) / cv['slope'] * DIL)
samp = samp.assign(mda=vals)
mda_subj = samp.groupby('sample_id')['mda'].mean()

meta_idx = meta.set_index('subject_id')
per = per.reindex(meta_idx.index)
per['mda_nmol_per_ml'] = mda_subj.reindex(meta_idx.index)
per['group'] = meta_idx['group']
per['specimen_type'] = meta_idx['specimen_type']

group_stats = {}
for col in ('nbt_positive_pct', 'nbt_reactivity', 'mda_nmol_per_ml'):
    a = per.loc[per['group'] == 'fertile_control', col].astype(float)
    b = per.loc[per['group'] == 'infertile', col].astype(float)
    _, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    group_stats[col] = {'control_median': float(a.median()),
                        'infertile_median': float(b.median()),
                        'p_value': float(p)}

corr = {}
for c1, c2 in (('nbt_positive_pct', 'nbt_reactivity'),
               ('nbt_reactivity', 'mda_nmol_per_ml'),
               ('nbt_positive_pct', 'mda_nmol_per_ml')):
    rho, p = stats.spearmanr(per[c1].astype(float), per[c2].astype(float))
    corr[c1 + '_vs_' + c2] = {'rho': float(rho), 'p_value': float(p)}

y = (per['group'] == 'infertile').astype(int).to_numpy()
diag = {}
for col in ('nbt_positive_pct', 'nbt_reactivity'):
    v = per[col].astype(float).to_numpy()
    auc = float(roc_auc_score(y, v))
    best_j, best = -1.0, None
    for c in np.unique(v):
        pred = v >= c
        tp = int((pred & (y == 1)).sum())
        fn = int(((~pred) & (y == 1)).sum())
        tn = int(((~pred) & (y == 0)).sum())
        fp = int((pred & (y == 0)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        j = sens + spec - 1.0
        if j > best_j + 1e-12:
            best_j, best = j, (float(c), sens, spec)
    diag[col] = {'auc': auc, 'youden_cutoff': best[0],
                 'sensitivity': best[1], 'specificity': best[2]}

rows = []
for sid, row in per.iterrows():
    rows.append({'subject_id': sid, 'group': row['group'],
                 'specimen_type': row['specimen_type'],
                 'n_cells_retained': int(row['n_cells_retained']),
                 'nbt_positive_pct': float(row['nbt_positive_pct']),
                 'nbt_reactivity': float(row['nbt_reactivity']),
                 'mda_nmol_per_ml': float(row['mda_nmol_per_ml'])})

out = {'qc': {'n_cells_measured': n_cells_measured,
              'n_fields_measured': n_fields_measured,
              'n_cells_retained': n_cells_retained,
              'n_fields_retained': n_fields_retained,
              'n_excluded_focus': n_exc_focus,
              'n_excluded_object_class': n_exc_class,
              'n_excluded_cell_area': n_exc_area,
              'n_wells_saturated_excluded': n_sat},
       'standard_curves': curves,
       'per_subject': rows,
       'group_comparison': group_stats,
       'correlations': corr,
       'diagnostic_performance': diag}
json.dump(out, open('/app/output/results.json', 'w'), indent=2)
print('wrote /app/output/results.json')
PY
python /app/run_pipeline.py
