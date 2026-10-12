#!/usr/bin/env python3
import os, sys, json, math
import numpy as np
import pandas as pd
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
            if s[i]=='N' and s[i+1]!='P' and s[i+2] in 'ST':
                c += 1
    return c

def count_deamid(seqs):
    c = 0
    for s in seqs:
        for i in range(len(s)-1):
            if s[i]=='N' and s[i+1] in 'GSNT':
                c += 1
    return c

def count_met(seqs):
    return sum(s.count('M') for s in seqs)

variants = read_fasta(os.path.join(DATA, 'variants.fasta'))
pd1 = read_fasta(os.path.join(DATA, 'pd1_epitope.fasta'))
pd1_seq = list(pd1.values())[0]
hotspot = pd.read_csv(os.path.join(DATA, 'hotspot_weights.csv'))
hotspot_w = dict(zip(hotspot['residue'], hotspot['weight']))

vids = sorted(set(k[:-2] for k in variants if k.endswith('_H')))
true_results = {}
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
    n_met = count_met(all_cdr_seqs)
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
                if np.linalg.norm(pi - aj.position) < 8.0:
                    F[i,j] += 1
    F /= n_frames
    bscore = 0.0
    for j in range(len(pd1_atoms)):
        res = j+1
        w = hotspot_w.get(res, 1.0)
        bscore += w * (F[:,j].max() if len(cdr_atoms)>0 else 0.0)
    passes = (n_glyc==0 and n_deamid==0 and n_met==0 and
              -2.0 <= net <= 4.0 and 7.0 <= pI <= 9.5 and bscore >= 2.0)
    true_results[vid] = {
        'variant_id': vid,
        'binding_score': bscore,
        'net_charge': net,
        'pI': pI,
        'n_glyc_motifs': n_glyc,
        'deamidation_motifs': n_deamid,
        'met_in_cdr': n_met,
        'passes_filters': passes,
    }

sorted_vids = sorted(true_results.keys(), key=lambda x: (-true_results[x]['binding_score'], x))
for rank, vid in enumerate(sorted_vids, 1):
    true_results[vid]['rank'] = rank
passing = [vid for vid in sorted_vids if true_results[vid]['passes_filters']]
top3_true = passing[:3]

agent_analysis = None
agent_top3 = None
try:
    agent_analysis = pd.read_csv(os.path.join(SUB, 'analysis.csv'))
except Exception as e:
    print(f"Error reading analysis.csv: {e}")
try:
    agent_top3 = pd.read_csv(os.path.join(SUB, 'top3.csv'))
except Exception as e:
    print(f"Error reading top3.csv: {e}")

scores = {}
if agent_analysis is None:
    scores['analysis_present'] = 0.0
else:
    agent_vids = set(agent_analysis['variant_id'])
    true_vids = set(true_results.keys())
    scores['analysis_present'] = 1.0 if agent_vids == true_vids else 0.0

def check_column(col, tol=None):
    if agent_analysis is None:
        return 0.0
    correct = 0
    total = len(true_results)
    for _, row in agent_analysis.iterrows():
        vid = row['variant_id']
        if vid in true_results:
            if tol is not None:
                if abs(row[col] - true_results[vid][col]) <= tol:
                    correct += 1
            else:
                if row[col] == true_results[vid][col]:
                    correct += 1
    return correct / total if total > 0 else 0.0

scores['binding_score'] = check_column('binding_score', tol=0.05)
scores['net_charge'] = check_column('net_charge', tol=0.1)
scores['pI'] = check_column('pI', tol=0.1)
if agent_analysis is not None:
    correct = 0
    total = len(true_results)
    for _, row in agent_analysis.iterrows():
        vid = row['variant_id']
        if vid in true_results:
            if (row['n_glyc_motifs'] == true_results[vid]['n_glyc_motifs'] and
                row['deamidation_motifs'] == true_results[vid]['deamidation_motifs'] and
                row['met_in_cdr'] == true_results[vid]['met_in_cdr']):
                correct += 1
    scores['motifs'] = correct / total if total > 0 else 0.0
else:
    scores['motifs'] = 0.0
scores['passes_filters'] = check_column('passes_filters')
scores['rank'] = check_column('rank')
if agent_top3 is not None and len(agent_top3) == 3:
    agent_top3_ids = list(agent_top3['variant_id'])
    scores['top3'] = 1.0 if agent_top3_ids == top3_true else 0.0
else:
    scores['top3'] = 0.0

weights = {
    'analysis_present': 0.05,
    'binding_score': 0.20,
    'net_charge': 0.10,
    'pI': 0.10,
    'motifs': 0.15,
    'passes_filters': 0.15,
    'rank': 0.10,
    'top3': 0.15,
}
final_score = sum(scores.get(k, 0.0) * weights[k] for k in weights)
os.makedirs('/logs/verifier', exist_ok=True)
with open('/logs/verifier/reward.json', 'w') as f:
    json.dump({'score': final_score, 'reward': final_score}, f)
with open('/logs/verifier/report.json', 'w') as f:
    json.dump({'scores': scores, 'weights': weights, 'final_score': final_score}, f)
print(json.dumps({'scores': scores, 'final_score': final_score}, indent=2))
