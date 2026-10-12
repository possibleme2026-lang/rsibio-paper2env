#!/usr/bin/env python3
"""Deterministic generator for the lower-respiratory-tract 16S ASV dataset.

Writes asv_table.tsv, taxonomy.tsv and metadata.tsv next to this script.
Run as: python /app/data/gen_inputs.py

The file is removed from the image right after the build so that the latent
structure encoded here (which ASVs are contaminants, which ASV is the
brush-specific Sphingomonadaceae indicator) is never readable by the agent.
"""
import os

import numpy as np
import pandas as pd

SEED = 20260802
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(SEED)

N_ASV = 400
TYPES = ['oral', 'bal', 'brush']
participants = ['P%02d' % i for i in range(1, 34)]
status = {p: ('cancer' if int(p[1:]) <= 8 else 'non_cancer') for p in participants}
part_samples = [(p, t) for p in participants for t in TYPES]
sample_ids = ['%s_%s' % (p, t) for p, t in part_samples]
control_ids = ['NC%02d' % i for i in range(1, 7)]
all_ids = sample_ids + control_ids
pos = {s: i for i, s in enumerate(all_ids)}
type_idx = {t: np.array([pos['%s_%s' % (p, t)] for p in participants]) for t in TYPES}
ctrl_idx = np.array([pos[c] for c in control_ids])
part_idx = np.arange(len(sample_ids))
asv_ids = ['ASV_%04d' % i for i in range(1, N_ASV + 1)]

counts = np.zeros((N_ASV, len(all_ids)), dtype=np.int64)


def draw(mean, idx):
    mean = np.asarray(mean, dtype=float)
    if mean.ndim == 0:
        mean = np.full(len(idx), float(mean))
    shape = 0.8
    lam = rng.gamma(shape, np.maximum(mean, 0.0) / shape)
    return rng.poisson(lam)


# block A (0-59): ASVs shared by oral rinse and BAL
for i in range(0, 60):
    base = float(np.exp(rng.normal(np.log(260.0), 0.8)))
    for t, mult in (('oral', 1.0), ('bal', rng.uniform(0.6, 1.6)), ('brush', 0.004)):
        counts[i, type_idx[t]] = draw(base * mult, type_idx[t])

# block B (60-99): BAL-enriched
for i in range(60, 100):
    base = float(np.exp(rng.normal(np.log(180.0), 0.8)))
    counts[i, type_idx['oral']] = draw(base * 0.01, type_idx['oral'])
    counts[i, type_idx['bal']] = draw(base, type_idx['bal'])
    counts[i, type_idx['brush']] = draw(base * 0.01, type_idx['brush'])

# block C (100-129): oral-enriched
for i in range(100, 130):
    base = float(np.exp(rng.normal(np.log(420.0), 0.8)))
    counts[i, type_idx['oral']] = draw(base, type_idx['oral'])
    counts[i, type_idx['bal']] = draw(base * 0.008, type_idx['bal'])
    counts[i, type_idx['brush']] = draw(base * 0.004, type_idx['brush'])

# block D (130-159): brush-enriched; ASV_0138 (index 137) is the Sphingomonadaceae one
for i in range(130, 160):
    base = float(np.exp(rng.normal(np.log(70.0), 0.6)))
    present = rng.random(len(type_idx['brush'])) < 0.72
    counts[i, type_idx['brush']] = draw(np.where(present, base, 0.0), type_idx['brush'])
    counts[i, type_idx['oral']] = draw(0.15, type_idx['oral'])
    counts[i, type_idx['bal']] = draw(0.15, type_idx['bal'])

# block E (160-179): reagent/library contaminants visible in the negative controls
brush_weights = np.ones(len(sample_ids))
for i, (p, t) in enumerate(part_samples):
    if t == 'brush':
        brush_weights[i] = 3.0
brush_weights = brush_weights / brush_weights.sum()
for i in range(160, 180):
    counts[i, ctrl_idx] = draw(rng.uniform(20.0, 400.0), ctrl_idx)
    n_carrier = int(rng.integers(18, 40))
    carriers = rng.choice(len(sample_ids), size=n_carrier, replace=False, p=brush_weights)
    counts[i, carriers] = draw(rng.uniform(1.0, 6.0), carriers)

# block F (180-399): sparse background community (participant samples only)
for i in range(180, N_ASV):
    p0 = rng.uniform(0.03, 0.45)
    base = float(np.exp(rng.normal(np.log(4.0), 1.4)))
    m = np.zeros(len(all_ids))
    m[part_idx] = np.where(rng.random(len(part_idx)) < p0, base, 0.0)
    counts[i] = draw(m, np.arange(len(all_ids)))

TAXA = [
    ('Firmicutes', 'Bacilli', 'Lactobacillales', 'Streptococcaceae', 'Streptococcus'),
    ('Firmicutes', 'Bacilli', 'Lactobacillales', 'Gemellaceae', 'Gemella'),
    ('Firmicutes', 'Bacilli', 'Lactobacillales', 'Aerococcaceae', 'Abiotrophia'),
    ('Firmicutes', 'Bacilli', 'Lactobacillales', 'Carnobacteriaceae', 'Granulicatella'),
    ('Firmicutes', 'Bacilli', 'Bacillales', 'Staphylococcaceae', 'Staphylococcus'),
    ('Firmicutes', 'Clostridia', 'Clostridiales', 'Lachnospiraceae', 'Lachnoanaerobaculum'),
    ('Firmicutes', 'Clostridia', 'Clostridiales', 'Ruminococcaceae', 'Ruminococcus'),
    ('Firmicutes', 'Negativicutes', 'Veillonellales', 'Veillonellaceae', 'Veillonella'),
    ('Firmicutes', 'Negativicutes', 'Veillonellales', 'Veillonellaceae', 'Megasphaera'),
    ('Bacteroidetes', 'Bacteroidia', 'Bacteroidales', 'Prevotellaceae', 'Prevotella'),
    ('Bacteroidetes', 'Bacteroidia', 'Bacteroidales', 'Bacteroidaceae', 'Bacteroides'),
    ('Bacteroidetes', 'Bacteroidia', 'Bacteroidales', 'Porphyromonadaceae', 'Porphyromonas'),
    ('Bacteroidetes', 'Flavobacteriia', 'Flavobacteriales', 'Flavobacteriaceae', 'Capnocytophaga'),
    ('Proteobacteria', 'Gammaproteobacteria', 'Pasteurellales', 'Pasteurellaceae', 'Haemophilus'),
    ('Proteobacteria', 'Betaproteobacteria', 'Neisseriales', 'Neisseriaceae', 'Neisseria'),
    ('Proteobacteria', 'Betaproteobacteria', 'Burkholderiales', 'Comamonadaceae', 'Comamonas'),
    ('Proteobacteria', 'Betaproteobacteria', 'Burkholderiales', 'Oxalobacteraceae', 'Janthinobacterium'),
    ('Proteobacteria', 'Alphaproteobacteria', 'Rhizobiales', 'Bradyrhizobiaceae', 'Bradyrhizobium'),
    ('Proteobacteria', 'Epsilonproteobacteria', 'Campylobacterales', 'Campylobacteraceae', 'Campylobacter'),
    ('Actinobacteria', 'Actinobacteria', 'Actinomycetales', 'Actinomycetaceae', 'Actinomyces'),
    ('Actinobacteria', 'Actinobacteria', 'Actinomycetales', 'Micrococcaceae', 'Rothia'),
    ('Actinobacteria', 'Actinobacteria', 'Actinomycetales', 'Corynebacteriaceae', 'Corynebacterium'),
    ('Actinobacteria', 'Actinobacteria', 'Actinomycetales', 'Propionibacteriaceae', 'Cutibacterium'),
    ('Fusobacteria', 'Fusobacteriia', 'Fusobacteriales', 'Fusobacteriaceae', 'Fusobacterium'),
    ('Fusobacteria', 'Fusobacteriia', 'Fusobacteriales', 'Leptotrichiaceae', 'Leptotrichia'),
]
SPHINGO = ('Proteobacteria', 'Alphaproteobacteria', 'Sphingomonadales',
           'Sphingomonadaceae', 'Sphingomonas')

tax_rows = []
for i, aid in enumerate(asv_ids):
    rec = SPHINGO if i == 137 else TAXA[int(rng.integers(0, len(TAXA)))]
    tax_rows.append({'asv_id': aid, 'Kingdom': 'Bacteria', 'Phylum': rec[0],
                     'Class': rec[1], 'Order': rec[2], 'Family': rec[3], 'Genus': rec[4]})
tax = pd.DataFrame(tax_rows).set_index('asv_id')

tab = pd.DataFrame(counts, index=asv_ids, columns=all_ids)
tab.index.name = 'asv_id'

batches = ['B1', 'B2', 'B3']
meta_rows = []
for k, (p, t) in enumerate(part_samples):
    meta_rows.append({'sample_id': '%s_%s' % (p, t), 'participant_id': p,
                      'sample_type': t, 'cancer_status': status[p],
                      'batch': batches[k % 3]})
for c in control_ids:
    meta_rows.append({'sample_id': c, 'participant_id': c, 'sample_type': 'control',
                      'cancer_status': 'NA', 'batch': 'B1'})
meta = pd.DataFrame(meta_rows)

tab.to_csv(os.path.join(HERE, 'asv_table.tsv'), sep='\t')
tax.to_csv(os.path.join(HERE, 'taxonomy.tsv'), sep='\t')
meta.to_csv(os.path.join(HERE, 'metadata.tsv'), sep='\t', index=False)
print('wrote %d ASVs x %d samples' % tab.shape)
