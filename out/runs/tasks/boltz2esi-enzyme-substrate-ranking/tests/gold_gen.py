import os, sys, math, json
import numpy as np
import pandas as pd
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, DataStructs, AllChem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import input_gen

AAs = 'ACDEFGHIKLMNPQRSTVWY'
ALL_KMERS = [a+b+c for a in AAs for b in AAs for c in AAs]

def parse_fasta(path):
    seqs = {}
    cur = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                cur = line[1:].split()[0]
                seqs[cur] = []
            else:
                seqs[cur].append(line)
    return {k: ''.join(v) for k,v in seqs.items()}

def parse_annotations(path):
    df = pd.read_csv(path, sep='\t')
    ann = {}
    for _, row in df.iterrows():
        cats = []
        for item in str(row['catalytic_residues']).split(';'):
            item = item.strip()
            if not item: continue
            res = item[0]
            pos = int(item[1:])
            cats.append((res, pos))
        ann[row['enzyme_id']] = {'family': row['family'], 'catalytic_residues': cats, 'length': int(row['length'])}
    return ann

def parse_substrates(path):
    subs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            subs[parts[0]] = parts[1]
    return subs

def parse_reference_pairs(path):
    return pd.read_csv(path, sep='\t')

def compute_ligand_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {'lig_mw': Descriptors.MolWt(mol), 'lig_logp': Crippen.MolLogP(mol), 'lig_tpsa': rdMolDescriptors.CalcTPSA(mol), 'lig_hbd': rdMolDescriptors.CalcNumHBD(mol), 'lig_hba': rdMolDescriptors.CalcNumHBA(mol), 'lig_rings': rdMolDescriptors.CalcNumRings(mol), 'lig_fsp3': rdMolDescriptors.CalcFractionCSP3(mol), 'lig_charge': Chem.GetFormalCharge(mol)}

def compute_pocket_features(seq, catalytic_residues):
    L = len(seq)
    A = set()
    for res, pos in catalytic_residues:
        start = max(1, pos-8)
        end = min(L, pos+8)
        for i in range(start, end+1):
            A.add(i)
    if not A:
        return {'pocket_hydrophobicity': 0.0, 'pocket_polarity': 0.0, 'pocket_charge': 0.0, 'pocket_aromatic': 0.0}
    hydro = set('AVILMFWY')
    polar = set('STNQCHKRDE')
    aromatic = set('FWYH')
    charge_map = {'D': -1.0, 'E': -1.0, 'K': 1.0, 'R': 1.0, 'H': 0.1}
    n_hydro = n_polar = n_arom = 0
    total_charge = 0.0
    for i in A:
        aa = seq[i-1]
        if aa in hydro: n_hydro += 1
        if aa in polar: n_polar += 1
        if aa in aromatic: n_arom += 1
        total_charge += charge_map.get(aa, 0.0)
    n = len(A)
    return {'pocket_hydrophobicity': n_hydro / n, 'pocket_polarity': n_polar / n, 'pocket_charge': total_charge / n, 'pocket_aromatic': n_arom / n}

def compute_kmer_vector(seq):
    counts = Counter(seq[i:i+3] for i in range(len(seq)-2))
    vec = np.array([counts.get(kmer, 0) + 0.5 for kmer in ALL_KMERS])
    return vec / vec.sum()

def jsd(p, q):
    m = 0.5 * (p + q)
    kl1 = np.sum(p * np.log2(p / m))
    kl2 = np.sum(q * np.log2(q / m))
    return math.sqrt(0.5 * kl1 + 0.5 * kl2)

def compute_kmer_jsd(enzyme_id, family, seqs, ann):
    vec_self = compute_kmer_vector(seqs[enzyme_id])
    family_members = [eid for eid, a in ann.items() if a['family'] == family and eid != enzyme_id]
    if len(family_members) >= 2:
        vecs = [compute_kmer_vector(seqs[eid]) for eid in family_members]
        consensus = np.mean(vecs, axis=0)
    else:
        other = [eid for eid in seqs if eid != enzyme_id]
        vecs = [compute_kmer_vector(seqs[eid]) for eid in other]
        consensus = np.mean(vecs, axis=0) if vecs else vec_self
    return jsd(vec_self, consensus)

def compute_tanimoto(smiles, enzyme_id, ref_df, sub_smiles, fp_cache):
    if smiles in fp_cache:
        fp = fp_cache[smiles]
    else:
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp_cache[smiles] = fp
    ref_sub_ids = ref_df[(ref_df['enzyme_id'] == enzyme_id) & (ref_df['interaction_score'] >= 0.7)]['substrate_id'].tolist()
    if not ref_sub_ids:
        return 0.0
    max_sim = 0.0
    for sid in ref_sub_ids:
        ref_smi = sub_smiles[sid]
        if ref_smi in fp_cache:
            ref_fp = fp_cache[ref_smi]
        else:
            mol = Chem.MolFromSmiles(ref_smi)
            ref_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_cache[ref_smi] = ref_fp
        sim = DataStructs.TanimotoSimilarity(fp, ref_fp)
        if sim > max_sim:
            max_sim = sim
    return max_sim

def main():
    gold_dir = '/tmp/gold_inputs'
    os.makedirs(gold_dir, exist_ok=True)
    input_gen.generate_all(gold_dir)
    seqs = parse_fasta(os.path.join(gold_dir, 'enzymes.fasta'))
    ann = parse_annotations(os.path.join(gold_dir, 'enzyme_annotations.tsv'))
    sub_smiles = parse_substrates(os.path.join(gold_dir, 'substrates.smi'))
    ref_df = parse_reference_pairs(os.path.join(gold_dir, 'reference_pairs.tsv'))
    lig_feats = {sid: compute_ligand_features(smi) for sid, smi in sub_smiles.items()}
    pocket_feats = {eid: compute_pocket_features(seqs[eid], a['catalytic_residues']) for eid, a in ann.items()}
    kmer_feats = {eid: compute_kmer_jsd(eid, a['family'], seqs, ann) for eid, a in ann.items()}
    fp_cache = {}
    for smi in sub_smiles.values():
        mol = Chem.MolFromSmiles(smi)
        fp_cache[smi] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    rows = []
    for eid in seqs:
        for sid in sub_smiles:
            feats = {}
            feats.update(lig_feats[sid])
            feats.update(pocket_feats[eid])
            feats['kmer_jsd'] = kmer_feats[eid]
            feats['tanimoto_max'] = compute_tanimoto(sub_smiles[sid], eid, ref_df, sub_smiles, fp_cache)
            rows.append({'enzyme_id': eid, 'substrate_id': sid, **feats})
    df = pd.DataFrame(rows)
    feat_cols = ['lig_mw','lig_logp','lig_tpsa','lig_hbd','lig_hba','lig_rings','lig_fsp3','lig_charge','pocket_hydrophobicity','pocket_polarity','pocket_charge','pocket_aromatic','kmer_jsd','tanimoto_max']
    ref_pairs = ref_df[['enzyme_id','substrate_id','interaction_score']].copy()
    ref_pairs = ref_pairs.merge(df, on=['enzyme_id','substrate_id'], how='left')
    X_ref = ref_pairs[feat_cols].values.astype(float)
    y_ref = ref_pairs['interaction_score'].values.astype(float)
    mean = X_ref.mean(axis=0)
    std = X_ref.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    X_ref_std = (X_ref - mean) / std
    y_mean = y_ref.mean()
    y_centered = y_ref - y_mean
    alpha = 1.0
    I = np.eye(X_ref_std.shape[1])
    w = np.linalg.solve(X_ref_std.T @ X_ref_std + alpha * I, X_ref_std.T @ y_centered)
    X_all = df[feat_cols].values.astype(float)
    X_all_std = (X_all - mean) / std
    pred = y_mean + X_all_std @ w
    pred = np.clip(pred, 0.0, 1.0)
    df['predicted_score'] = pred
    df[['enzyme_id','substrate_id'] + feat_cols].to_csv('/tmp/gold_features.tsv', sep='\t', index=False)
    df['rank'] = df.groupby('enzyme_id')['predicted_score'].rank(ascending=False, method='first').astype(int)
    df[['enzyme_id','substrate_id','predicted_score','rank']].to_csv('/tmp/gold_predictions.tsv', sep='\t', index=False)
    print('Gold generated.')

if __name__ == '__main__':
    main()
