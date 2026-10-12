import json
import numpy as np
import pandas as pd

np.random.seed(20260803)

n_mice = 36
n_genes = 200

sig_genes = {
    'MTS1': [f'MTS1_SIG_{i:02d}' for i in range(1, 21)],
    'MTS2': [f'MTS2_SIG_{i:02d}' for i in range(1, 21)],
    'MTS3': [f'MTS3_SIG_{i:02d}' for i in range(1, 21)],
}

genes = []
for mts in ['MTS1', 'MTS2', 'MTS3']:
    genes.extend(sig_genes[mts])
genes.extend(['Cxcr2', 'Cxcl1', 'Cxcl2', 'Il6', 'Mpo', 'Elane', 'Hmgb1', 'Tnf', 'Il1b', 'S100a8', 'S100a9'])
while len(genes) < n_genes:
    genes.append(f'GENE_{len(genes)+1:03d}')
genes = genes[:n_genes]
assert len(genes) == n_genes
assert len(set(genes)) == n_genes

mouse_ids = [f'M{i:02d}' for i in range(1, n_mice + 1)]

clusters = np.array(['MTS1'] * 12 + ['MTS2'] * 12 + ['MTS3'] * 12)
np.random.shuffle(clusters)

treatments = np.empty(n_mice, dtype=object)
for mts in ['MTS1', 'MTS2', 'MTS3']:
    idx = np.where(clusters == mts)[0]
    np.random.shuffle(idx)
    tr = np.array(['Vehicle'] * 6 + ['Danirixin'] * 6)
    for i, t in zip(idx, tr):
        treatments[i] = t

survival = np.zeros(n_mice, dtype=int)
for mts, veh_surv, dan_surv in [('MTS1', 5, 3), ('MTS2', 4, 4), ('MTS3', 1, 5)]:
    idx = np.where(clusters == mts)[0]
    veh_idx = [i for i in idx if treatments[i] == 'Vehicle']
    dan_idx = [i for i in idx if treatments[i] == 'Danirixin']
    np.random.shuffle(veh_idx)
    np.random.shuffle(dan_idx)
    for i in veh_idx[:veh_surv]:
        survival[i] = 1
    for i in dan_idx[:dan_surv]:
        survival[i] = 1

base_pathogen = {'MTS1': 4.0, 'MTS2': 5.0, 'MTS3': 6.0}
base_il6 = {'MTS1': 100.0, 'MTS2': 300.0, 'MTS3': 900.0}
base_netosis = {'MTS1': 5.0, 'MTS2': 10.0, 'MTS3': 20.0}
base_alt = {'MTS1': 50.0, 'MTS2': 100.0, 'MTS3': 200.0}

pathogen_effect = {'MTS1': 0.8, 'MTS2': 0.1, 'MTS3': -0.5}
il6_effect = {'MTS1': 20.0, 'MTS2': 10.0, 'MTS3': -50.0}
netosis_effect = {'MTS1': -3.0, 'MTS2': -1.0, 'MTS3': -8.0}
alt_effect = {'MTS1': -15.0, 'MTS2': -10.0, 'MTS3': -50.0}

pathogen = np.zeros(n_mice)
il6 = np.zeros(n_mice)
netosis = np.zeros(n_mice)
alt = np.zeros(n_mice)
for i in range(n_mice):
    mts = clusters[i]
    tr = treatments[i]
    p = base_pathogen[mts] + (pathogen_effect[mts] if tr == 'Danirixin' else 0) + np.random.normal(0, 0.2)
    il = base_il6[mts] + (il6_effect[mts] if tr == 'Danirixin' else 0) + np.random.normal(0, 20)
    ne = base_netosis[mts] + (netosis_effect[mts] if tr == 'Danirixin' else 0) + np.random.normal(0, 1.0)
    al = base_alt[mts] + (alt_effect[mts] if tr == 'Danirixin' else 0) + np.random.normal(0, 10.0)
    pathogen[i] = max(0, p)
    il6[i] = max(0, il)
    netosis[i] = max(0, ne)
    alt[i] = max(0, al)

meta = pd.DataFrame({
    'mouse_id': mouse_ids,
    'treatment': treatments,
    'survival_7d': survival,
    'pathogen_log10': pathogen,
    'IL6_pg_ml': il6,
    'NETosis_units': netosis,
    'hepatic_injury_ALT': alt,
})
meta.to_csv('/app/data/metadata.tsv', sep='\t', index=False)

def generate_tissue(tissue_name):
    base = np.random.normal(5, 1, size=n_genes)
    lib_factor = np.random.lognormal(0, 0.2, size=n_mice)
    counts = np.zeros((n_genes, n_mice), dtype=int)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    for i in range(n_mice):
        mts = clusters[i]
        log2_expr = base.copy()
        for mts_name, gene_list in sig_genes.items():
            for g in gene_list:
                idx = gene_to_idx[g]
                if mts_name == 'MTS1':
                    if mts == 'MTS1':
                        log2_expr[idx] += 2.0
                    elif mts == 'MTS3':
                        log2_expr[idx] -= 0.5
                elif mts_name == 'MTS2':
                    if mts == 'MTS2':
                        log2_expr[idx] += 2.0
                elif mts_name == 'MTS3':
                    if mts == 'MTS3':
                        log2_expr[idx] += 2.0
                    elif mts == 'MTS1':
                        log2_expr[idx] -= 0.5
        if 'Cxcr2' in gene_to_idx:
            idx = gene_to_idx['Cxcr2']
            if mts == 'MTS3':
                log2_expr[idx] -= 1.0
        if 'Il6' in gene_to_idx:
            idx = gene_to_idx['Il6']
            if mts == 'MTS3':
                log2_expr[idx] += 1.5
            elif mts == 'MTS2':
                log2_expr[idx] += 0.5
        log2_expr += np.random.normal(0, 0.3, size=n_genes)
        lam = (2 ** log2_expr) * lib_factor[i]
        counts[:, i] = np.random.poisson(lam)
    df = pd.DataFrame(counts, index=genes, columns=mouse_ids)
    df.to_csv(f'/app/data/{tissue_name}_counts.tsv', sep='\t')

generate_tissue('blood')
generate_tissue('liver')

with open('/app/data/signatures.json', 'w') as f:
    json.dump(sig_genes, f, indent=2)

print('Data generated successfully.')
