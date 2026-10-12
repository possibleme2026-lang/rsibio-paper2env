import numpy as np
import pandas as pd
import json
import os

np.random.seed(42)

groups = ['Control', 'DOX', 'DOX+RSV', 'DOX+NMN']
n_per_group = 6
samples = []
for g in groups:
    for i in range(1, n_per_group+1):
        samples.append(f'{g}_{i}')

oxphos_genes = [f'OXPHOS_{i}' for i in range(1, 51)]
fao_genes = [f'FAO_{i}' for i in range(1, 41)]
inflam_genes = [f'INFLAM_{i}' for i in range(1, 51)]
sirt_genes = ['Sirt1', 'Sirt3', 'Sirt5']
sod2_gene = ['SOD2']
other_genes = [f'Gene_{i}' for i in range(1, 357)]
all_genes = oxphos_genes + fao_genes + inflam_genes + sirt_genes + sod2_gene + other_genes

base_means = np.random.lognormal(mean=2, sigma=1, size=len(all_genes))
base_means = np.clip(base_means, 1, 5000)

log2fc = {}
for g in all_genes:
    log2fc[g] = {'Control': 0, 'DOX': 0, 'DOX+RSV': 0, 'DOX+NMN': 0}

for g in oxphos_genes + fao_genes:
    log2fc[g]['DOX'] = -1.5
    log2fc[g]['DOX+RSV'] = -0.5
    log2fc[g]['DOX+NMN'] = -0.7
for g in inflam_genes:
    log2fc[g]['DOX'] = 1.5
    log2fc[g]['DOX+RSV'] = 0.5
    log2fc[g]['DOX+NMN'] = 0.7
for g in sirt_genes:
    log2fc[g]['DOX'] = -1.0
    log2fc[g]['DOX+RSV'] = 0.5
    log2fc[g]['DOX+NMN'] = 0.2
for g in other_genes:
    for grp in groups:
        if grp != 'Control':
            log2fc[g][grp] = np.random.normal(0, 0.2)

counts = np.zeros((len(all_genes), len(samples)))
for i, g in enumerate(all_genes):
    for j, s in enumerate(samples):
        grp = s.split('_')[0]
        mu = base_means[i] * (2 ** log2fc[g][grp])
        phi = 0.1
        n = 1/phi
        p = n / (n + mu)
        counts[i, j] = np.random.negative_binomial(n, p)

rna_df = pd.DataFrame(counts, index=all_genes, columns=samples)
rna_df.index.name = 'gene'
rna_df.to_csv('/app/data/rna_counts.csv')

prot_genes = [f'Prot_{i}' for i in range(1, 301)]
prot_genes[0] = 'SOD2'
prot_genes[1] = 'SOD1'
prot_genes[2] = 'Catalase'
base_prot = np.random.lognormal(mean=5, sigma=1, size=len(prot_genes))
prot_effects = {}
for g in prot_genes:
    prot_effects[g] = {'Control': 0, 'DOX': 0, 'DOX+RSV': 0, 'DOX+NMN': 0}
for i, g in enumerate(prot_genes):
    if i < 50:
        prot_effects[g]['DOX'] = -1.0
        prot_effects[g]['DOX+RSV'] = -0.2
        prot_effects[g]['DOX+NMN'] = -0.3
    elif i < 100:
        prot_effects[g]['DOX'] = 1.0
        prot_effects[g]['DOX+RSV'] = 0.2
        prot_effects[g]['DOX+NMN'] = 0.3
    else:
        for grp in groups:
            if grp != 'Control':
                prot_effects[g][grp] = np.random.normal(0, 0.1)

prot_data = np.zeros((len(prot_genes), len(samples)))
for i, g in enumerate(prot_genes):
    for j, s in enumerate(samples):
        grp = s.split('_')[0]
        mu = base_prot[i] * (2 ** prot_effects[g][grp])
        prot_data[i, j] = mu * np.random.lognormal(mean=0, sigma=0.2)

prot_df = pd.DataFrame(prot_data, index=prot_genes, columns=samples)
prot_df.index.name = 'protein'
prot_df.to_csv('/app/data/proteomics.csv')

total = np.zeros(len(samples))
acetyl = np.zeros(len(samples))
for j, s in enumerate(samples):
    grp = s.split('_')[0]
    if grp == 'Control':
        total[j] = 100 * np.random.lognormal(0, 0.1)
        acetyl[j] = 20 * np.random.lognormal(0, 0.1)
    elif grp == 'DOX':
        total[j] = 100 * np.random.lognormal(0, 0.1)
        acetyl[j] = 50 * np.random.lognormal(0, 0.1)
    elif grp == 'DOX+RSV':
        total[j] = 100 * np.random.lognormal(0, 0.1)
        acetyl[j] = 25 * np.random.lognormal(0, 0.1)
    elif grp == 'DOX+NMN':
        total[j] = 100 * np.random.lognormal(0, 0.1)
        acetyl[j] = 30 * np.random.lognormal(0, 0.1)

sod2_df = pd.DataFrame({
    'sample': samples,
    'group': [s.split('_')[0] for s in samples],
    'total_SOD2': total,
    'acetylated_SOD2': acetyl
})
sod2_df.to_csv('/app/data/sod2_acetyl.csv', index=False)

wells = []
for g in groups:
    for i in range(1, n_per_group+1):
        wells.append({'well': f'{g}_{i}', 'group': g})

times = np.arange(0, 91, 3)
seahorse_rows = []
for well_info in wells:
    well = well_info['well']
    grp = well_info['group']
    if grp == 'Control':
        basal = 100
        max_resp = 150
        non_mito = 10
        atp_prod = 70
    elif grp == 'DOX':
        basal = 60
        max_resp = 80
        non_mito = 15
        atp_prod = 35
    elif grp == 'DOX+RSV':
        basal = 85
        max_resp = 120
        non_mito = 12
        atp_prod = 55
    elif grp == 'DOX+NMN':
        basal = 80
        max_resp = 110
        non_mito = 13
        atp_prod = 50
    basal *= np.random.lognormal(0, 0.05)
    max_resp *= np.random.lognormal(0, 0.05)
    non_mito *= np.random.lognormal(0, 0.05)
    atp_prod *= np.random.lognormal(0, 0.05)
    for t in times:
        if t < 20:
            ocr = basal + np.random.normal(0, 2)
        elif t < 40:
            ocr = (basal - atp_prod) + np.random.normal(0, 2)
        elif t < 60:
            ocr = max_resp + np.random.normal(0, 3)
        else:
            ocr = non_mito + np.random.normal(0, 1)
        seahorse_rows.append({'well': well, 'group': grp, 'time': t, 'OCR': max(0, ocr)})

seahorse_df = pd.DataFrame(seahorse_rows)
seahorse_df.to_csv('/app/data/seahorse_ocr.csv', index=False)

sim_rows = []
cell_id = 1
for grp in groups:
    for i in range(50):
        if grp == 'Control':
            ar = np.random.lognormal(mean=np.log(3.0), sigma=0.3)
            ff = np.random.beta(5, 2)
        elif grp == 'DOX':
            ar = np.random.lognormal(mean=np.log(1.5), sigma=0.4)
            ff = np.random.beta(2, 5)
        elif grp == 'DOX+RSV':
            ar = np.random.lognormal(mean=np.log(2.5), sigma=0.3)
            ff = np.random.beta(4, 2)
        elif grp == 'DOX+NMN':
            ar = np.random.lognormal(mean=np.log(2.3), sigma=0.3)
            ff = np.random.beta(3.5, 2)
        area = np.random.lognormal(mean=np.log(100), sigma=0.2)
        perimeter = 2 * np.pi * np.sqrt(area / np.pi) * np.random.lognormal(0, 0.1)
        mito_count = int(np.random.poisson(50 if grp == 'Control' else 80))
        sim_rows.append({
            'cell_id': f'cell_{cell_id}',
            'group': grp,
            'aspect_ratio': ar,
            'form_factor': ff,
            'area': area,
            'perimeter': perimeter,
            'mitochondrial_count': mito_count
        })
        cell_id += 1

sim_df = pd.DataFrame(sim_rows)
sim_df.to_csv('/app/data/sim_morphology.csv', index=False)

gene_sets = {
    'OXPHOS': oxphos_genes,
    'FAO': fao_genes,
    'Inflammation': inflam_genes
}
with open('/app/data/gene_sets.json', 'w') as f:
    json.dump(gene_sets, f, indent=2)

print('Data generation complete.')
