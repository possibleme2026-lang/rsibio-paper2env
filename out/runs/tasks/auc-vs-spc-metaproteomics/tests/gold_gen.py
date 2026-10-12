#!/usr/bin/env python3
import argparse, json, os
import numpy as np
import pandas as pd

def generate(outdir, truth_path, seed=20260405):
    rng = np.random.default_rng(seed)
    n_proteins = 400
    proteins = [f'PROT_{i:04d}' for i in range(n_proteins)]
    spiked = proteins[:60]
    base = rng.normal(20, 2, n_proteins)
    fc = np.zeros(n_proteins)
    fc[:60] = rng.uniform(-4, 4, 60)
    true_A = base
    true_B = base + fc

    peptides = []
    pep_prot = []
    for pidx, prot in enumerate(proteins):
        n_pep = int(rng.integers(3, 9))
        for j in range(n_pep):
            pep = f'PEP_{pidx:04d}_{j:02d}'
            peptides.append(pep)
            pep_prot.append(prot)
    shared_map = []
    for i in range(100):
        p1 = rng.choice(proteins[:200])
        p2 = rng.choice(proteins[:200])
        while p2 == p1:
            p2 = rng.choice(proteins[:200])
        pep = f'SHARED_{i:03d}'
        shared_map.append((pep, p1))
        shared_map.append((pep, p2))

    map_rows = []
    for pep, prot in zip(peptides, pep_prot):
        map_rows.append({'peptide_sequence': pep, 'protein_id': prot, 'mapping_type': 'unique'})
    for pep, prot in shared_map:
        map_rows.append({'peptide_sequence': pep, 'protein_id': prot, 'mapping_type': 'shared'})

    runs = ['A_rep1','A_rep2','A_rep3','B_rep1','B_rep2','B_rep3']
    run_sample = {r: r.split('_')[0] for r in runs}

    psm_rows = []
    feature_rows = []
    feature_id = 0

    for run in runs:
        sample = run_sample[run]
        true_prot = true_A if sample == 'A' else true_B
        for pep, prot in zip(peptides, pep_prot):
            pidx = proteins.index(prot)
            ab = true_prot[pidx]
            det_prob = 1 / (1 + np.exp(-(ab - 18)/2))
            if rng.random() < det_prob:
                n_psm = int(rng.integers(1, 3))
                for k in range(n_psm):
                    charge = int(rng.choice([2,3]))
                    is_mbr = 1 if rng.random() < 0.15 else 0
                    target = f'spec_orig_{run}_{pep}' if is_mbr else ''
                    score = float(rng.normal(60, 10))
                    spectrum_id = f'spec_{run}_{pep}_{k}'
                    psm_rows.append({
                        'run': run, 'spectrum_id': spectrum_id, 'peptide_sequence': pep,
                        'charge': charge, 'protein_id': prot, 'score': round(score,2),
                        'is_mbr': is_mbr, 'mbr_target_spectrum': target
                    })
                    feature_id += 1
                    auc = float(10 ** (ab/2 + rng.normal(0, 0.3)))
                    feature_rows.append({
                        'run': run, 'feature_id': f'F{feature_id:06d}',
                        'peptide_sequence': pep, 'charge': charge,
                        'mz': round(400 + rng.random()*800, 4),
                        'rt': round(rng.random()*60, 2),
                        'auc': round(auc, 2)
                    })
        for pep, prot in shared_map:
            if rng.random() < 0.1:
                charge = int(rng.choice([2,3]))
                is_mbr = 1 if rng.random() < 0.15 else 0
                psm_rows.append({
                    'run': run, 'spectrum_id': f'spec_{run}_{pep}_{prot}',
                    'peptide_sequence': pep, 'charge': charge,
                    'protein_id': prot, 'score': round(float(rng.normal(50,10)),2),
                    'is_mbr': is_mbr, 'mbr_target_spectrum': ''
                })
                feature_id += 1
                feature_rows.append({
                    'run': run, 'feature_id': f'F{feature_id:06d}',
                    'peptide_sequence': pep, 'charge': charge,
                    'mz': round(400 + rng.random()*800, 4),
                    'rt': round(rng.random()*60, 2),
                    'auc': round(float(10 ** (rng.normal(10, 1))), 2)
                })

    false_mbr = []
    df_psm_temp = pd.DataFrame(psm_rows)
    for run in runs:
        nonmbr_prots = set(df_psm_temp[(df_psm_temp.run == run) & (df_psm_temp.is_mbr == 0)].protein_id)
        candidates = [p for p in proteins if p not in nonmbr_prots]
        for prot in rng.choice(candidates, size=5, replace=False):
            peps = [p for p,pr in zip(peptides, pep_prot) if pr == prot]
            if not peps:
                continue
            pep = rng.choice(peps)
            charge = int(rng.choice([2,3]))
            spectrum_id = f'spec_false_{run}_{prot}'
            psm_rows.append({
                'run': run, 'spectrum_id': spectrum_id, 'peptide_sequence': pep,
                'charge': charge, 'protein_id': prot, 'score': round(float(rng.normal(45,10)),2),
                'is_mbr': 1, 'mbr_target_spectrum': f'spec_orig_{run}_{pep}'
            })
            false_mbr.append({'run': run, 'protein_id': prot, 'peptide_sequence': pep})
            feature_id += 1
            feature_rows.append({
                'run': run, 'feature_id': f'F{feature_id:06d}',
                'peptide_sequence': pep, 'charge': charge,
                'mz': round(400 + rng.random()*800, 4),
                'rt': round(rng.random()*60, 2),
                'auc': round(float(10 ** (rng.normal(9, 1))), 2)
            })

    df_psm = pd.DataFrame(psm_rows)
    df_feat = pd.DataFrame(feature_rows)
    df_map = pd.DataFrame(map_rows)
    df_meta = pd.DataFrame({
        'run': runs,
        'sample': [run_sample[r] for r in runs],
        'replicate': [int(r.split('_')[1].replace('rep','')) for r in runs],
        'condition': ['A' if run_sample[r]=='A' else 'B' for r in runs]
    })

    os.makedirs(outdir, exist_ok=True)
    df_psm.sort_values(['run','spectrum_id']).to_csv(os.path.join(outdir,'psm_table.tsv'), sep='\t', index=False)
    df_feat.sort_values(['run','feature_id']).to_csv(os.path.join(outdir,'feature_table.tsv'), sep='\t', index=False)
    df_map.sort_values(['peptide_sequence','protein_id']).to_csv(os.path.join(outdir,'peptide_protein_map.tsv'), sep='\t', index=False)
    df_meta.to_csv(os.path.join(outdir,'sample_metadata.tsv'), sep='\t', index=False)

    truth = {
        'proteins': proteins,
        'spiked_proteins': spiked,
        'true_A': {prot: float(true_A[i]) for i,prot in enumerate(proteins)},
        'true_B': {prot: float(true_B[i]) for i,prot in enumerate(proteins)},
        'false_mbr': false_mbr
    }
    os.makedirs(os.path.dirname(truth_path), exist_ok=True)
    with open(truth_path, 'w') as f:
        json.dump(truth, f, indent=2)
    print(f'Generated {len(df_psm)} PSMs, {len(df_feat)} features, {len(df_map)} mappings, {len(false_mbr)} false MBR transfers')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='/app/data')
    parser.add_argument('--truth', default='/tmp/hidden_truth.json')
    args = parser.parse_args()
    generate(args.outdir, args.truth)
