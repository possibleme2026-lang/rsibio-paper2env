#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python - <<'PY'
import json

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

DATA = '/app/data'
OUT = '/app/output'
ALPHA = 0.05
LFC_CUT = 1.0
CONTRASTS = ['TEADi_low', 'TEADi_high', 'TEADi_recovery', 'YAPTAZ_KD']

raw = pd.read_csv(DATA + '/kidney_counts.csv', index_col=0)
meta = pd.read_csv(DATA + '/sample_metadata.csv', index_col='sample_id')
gmt = {}
with open(DATA + '/gene_sets.gmt') as fh:
    for line in fh:
        parts = line.rstrip('\n').split('\t')
        gmt[parts[0]] = [g for g in parts[2:] if g]

keep = (raw >= 10).sum(axis=1) >= 6
counts = raw.loc[keep]
meta = meta.loc[counts.columns]

m = DeseqDataSet(counts=counts.T, metadata=meta, design='~condition')
m.deseq2()
res = {}
for grp in CONTRASTS:
    st = DeseqStats(m, contrast=['condition', grp, 'vehicle'], independent_filter=False)
    st.summary()
    res[grp] = st.results_df.copy()


def is_de(df):
    p = df['padj']
    return p.notna() & (p <= ALPHA) & (df['log2FoldChange'].abs() >= LFC_CUT)


de = {g: set(res[g].index[is_de(res[g]).to_numpy()]) for g in CONTRASTS}
hi = res['TEADi_high']
kd = res['YAPTAZ_KD']
lfc_hi = hi['log2FoldChange']
lfc_lo = res['TEADi_low']['log2FoldChange']
rec = res['TEADi_recovery']

concordant = sorted([g for g in (de['TEADi_high'] & de['YAPTAZ_KD'])
                     if np.sign(hi.loc[g, 'log2FoldChange']) == np.sign(kd.loc[g, 'log2FoldChange'])])

panel = [g for g in gmt['ALDOSTERONE_SIGNALING'] if g in hi.index]
panel_lfc = {g: float(lfc_hi[g]) for g in panel}
score = float(np.mean([panel_lfc[g] for g in panel]))

rng = np.random.default_rng(42)
universe = lfc_hi.to_numpy()
null = np.empty(1000)
for i in range(1000):
    null[i] = abs(float(np.mean(rng.choice(universe, size=len(panel), replace=False))))
p_emp = (1.0 + float((null >= abs(score)).sum())) / 1001.0

mono = [g for g in panel if lfc_hi[g] < lfc_lo[g] and lfc_hi[g] < 0.0]
mono_frac = len(mono) / float(len(panel))

hi_de = sorted(de['TEADi_high'])
persist = [g for g in hi_de
           if pd.notna(rec.loc[g, 'padj']) and rec.loc[g, 'padj'] <= ALPHA
           and abs(rec.loc[g, 'log2FoldChange']) >= LFC_CUT
           and np.sign(rec.loc[g, 'log2FoldChange']) == np.sign(hi.loc[g, 'log2FoldChange'])]
persist_frac = len(persist) / float(len(hi_de))

set_means = {}
for name, members in gmt.items():
    vals = [float(lfc_hi[g]) for g in members if g in hi.index]
    set_means[name] = float(np.mean(vals))
ranking = [n for n, _ in sorted(set_means.items(), key=lambda kv: -kv[1])]

tmp = hi[['padj']].dropna().copy()
tmp['gene'] = tmp.index.astype(str)
tmp.index.name = None
tmp = tmp.sort_values(['padj', 'gene'], kind='mergesort')
top20 = [str(g) for g in tmp['gene'].head(20)]

results = {
    'n_samples': int(counts.shape[1]),
    'n_genes_retained': int(counts.shape[0]),
    'de_counts': {g: int(len(de[g])) for g in CONTRASTS},
    'concordant_high_vs_KD': {'n': len(concordant), 'genes': concordant},
    'aldosterone_panel_log2fc': panel_lfc,
    'aldosterone_signature_score': score,
    'aldosterone_permutation_p': p_emp,
    'dose_monotonic_fraction': float(mono_frac),
    'recovery_persistent_fraction': float(persist_frac),
    'set_mean_log2fc_ranking': ranking,
    'top20_high_vs_vehicle': top20,
}
with open(OUT + '/results.json', 'w') as fh:
    json.dump(results, fh, indent=2)

out = hi.copy()
out.index.name = 'gene'
out = out.reset_index()
out = out[['gene', 'baseMean', 'log2FoldChange', 'lfcSE', 'stat', 'pvalue', 'padj']]
out.index.name = None
out = out.sort_values(['padj', 'gene'], kind='mergesort', na_position='last')
out.to_csv(OUT + '/de_TEADi_high_vs_vehicle.tsv', sep='\t', index=False)

print('n_genes_retained=%d n_high_DE=%d score=%.3f p=%.5f' % (
    counts.shape[0], len(de['TEADi_high']), score, p_emp))
PY
