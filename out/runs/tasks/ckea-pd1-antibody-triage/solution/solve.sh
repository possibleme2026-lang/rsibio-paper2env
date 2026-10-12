#!/bin/bash
set -euo pipefail
mkdir -p /app/submission
python3 << 'EOF'
import os, numpy as np, pandas as pd
from abnumber import Chain
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import MDAnalysis as mda

DATA = "/app/data"
SUB = "/app/submission"

def read_fasta(path):
    seqs = {}
    name = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                name = line[1:]
                seqs[name] = ''
            else:
                seqs[name] += line
    return seqs

variants = read_fasta(os.path.join(DATA, 'variants.fasta'))
pd1 = read_fasta(os.path.join(DATA, 'pd1_epitope.fasta'))
pd1_seq = list(pd1.values())[0]
hotspot = pd.read_csv(os.path.join(DATA, 'hotspot_weights.csv'))
hotspot_w = dict(zip(hotspot['residue'], hotspot['weight']))
vids = sorted(set(k[:-2] for k in variants if k.endswith('_H')))

def get_cdr_positions(seq):
    chain = Chain(seq, scheme='imgt')
    cdrs = {}
    for name, cdr_seq in [('CDR1', chain.cdr1_seq), ('CDR2', chain.cdr2_seq), ('CDR3', chain.cdr3_seq)]:
        if cdr_seq:
            idx = seq.find(cdr_seq)
            if idx != -1:
                cdrs[name] = (idx, idx+len(cdr_seq))
    return cdrs

def count_glyc(seqs):
    c = 0
    for s in seqs:
        for i in range(len(s)-2):
            if s[i]=='N' and s[i+1]!='P' and s[i+2] in 'ST': c += 1
    return c

def count_deamid(seqs):
    c = 0
    for s in seqs:
        for i in range(len(s)-1):
            if s[i]=='N' and s[i+1] in 'GSNT': c += 1
    return c

results = []
for vid in vids:
    vh = variants[vid + '_H']
    vl = variants[vid + '_L']
    full = vh + vl
    vh_cdrs = get_cdr_positions(vh)
    vl_cdrs = get_cdr_positions(vl)
    vh_cdr_seqs = [vh[s:e] for s,e in vh_cdrs.values()]
    vl_cdr_seqs = [vl[s:e] for s,e in vl_cdrs.values()]
    all_cdr_seqs = vh_cdr_seqs + vl_cdr_seqs
    n_glyc = count_glyc(all_cdr_seqs)
    n_deamid = count_deamid(all_cdr_seqs)
    n_met = sum(s.count('M') for s in all_cdr_seqs)
    ana = ProteinAnalysis(full)
    net = ana.charge_at_pH(7.4)
    pI = ana.isoelectric_point()
    u = mda.Universe(os.path.join(DATA, 'structures', f'{vid}.pdb'),
                     os.path.join(DATA, 'trajectories', f'{vid}.dcd'))
    vh_cdr_res = set()
    for s,e in vh_cdrs.values(): vh_cdr_res.update(range(s,e))
    vl_cdr_res = set()
    for s,e in vl_cdrs.values(): vl_cdr_res.update(range(s,e))
    cdr_atoms = []
    for chain_id, res_set in [('H', vh_cdr_res), ('L', vl_cdr_res)]:
        for ri in res_set:
            sel = u.select_atoms(f"chainID {chain_id} and resid {ri+1} and name CA")
            if len(sel)==1: cdr_atoms.append(sel[0])
    pd1_atoms = []
    for i in range(len(pd1_seq)):
        sel = u.select_atoms(f"chainID P and resid {i+1} and name CA")
        if len(sel)==1: pd1_atoms.append(sel[0])
    n_frames = len(u.trajectory)
    F = np.zeros((len(cdr_atoms), len(pd1_atoms)))
    for ts in u.trajectory:
        for i, ai in enumerate(cdr_atoms):
            pi = ai.position
            for j, aj in enumerate(pd1_atoms):
                if np.linalg.norm(pi - aj.position) < 8.0: F[i,j] += 1
    F /= n_frames
    bscore = 0.0
    for j in range(len(pd1_atoms)):
        res = j+1
        w = hotspot_w.get(res, 1.0)
        bscore += w * (F[:,j].max() if len(cdr_atoms)>0 else 0.0)
    passes = (n_glyc==0 and n_deamid==0 and n_met==0 and
              -2.0 <= net <= 4.0 and 7.0 <= pI <= 9.5 and bscore >= 2.0)
    results.append({
        'variant_id': vid,
        'binding_score': round(bscore, 4),
        'net_charge': round(net, 2),
        'pI': round(pI, 2),
        'n_glyc_motifs': n_glyc,
        'deamidation_motifs': n_deamid,
        'met_in_cdr': n_met,
        'passes_filters': passes,
    })

df = pd.DataFrame(results)
df = df.sort_values(['binding_score', 'variant_id'], ascending=[False, True]).reset_index(drop=True)
df['rank'] = range(1, len(df)+1)
df.to_csv(os.path.join(SUB, 'analysis.csv'), index=False)
passing = df[df['passes_filters']].head(3).copy()
for idx, row in passing.iterrows():
    vh = variants[row['variant_id'] + '_H']
    vh_cdrs = get_cdr_positions(vh)
    if 'CDR3' in vh_cdrs:
        s,e = vh_cdrs['CDR3']
        passing.at[idx, 'cdr_h3_sequence'] = vh[s:e]
    else:
        passing.at[idx, 'cdr_h3_sequence'] = ''
passing = passing[['rank','variant_id','binding_score','net_charge','pI','cdr_h3_sequence']]
passing.to_csv(os.path.join(SUB, 'top3.csv'), index=False)
EOF
