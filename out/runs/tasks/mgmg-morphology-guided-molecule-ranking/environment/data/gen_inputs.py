#!/usr/bin/env python3
import os, sys, random, hashlib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, QED
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

FRAGMENTS = [
    'c1ccccc1', 'c1ccncc1', 'C1CCNCC1', 'C1CCOC1', 'C1CC1', 'C(=O)N', 'C(=O)O', 'S(=O)(=O)N',
    'C(F)(F)F', 'Cl', 'Br', 'OC', 'N', 'C#N', 'C=C', 'C#C', 'c1ccsc1', 'c1cc[nH]c1',
    'C1=CC=CC=C1', 'C1=CC=NC=C1', 'C1=CN=CC=C1', 'C1=COC=C1', 'C1=CSC=C1',
    'C1CC2CCC1C2', 'C1CC1', 'C1CCC1', 'C1CCN(CC1)', 'C1CCOCC1', 'C1CNCCN1', 'C1COCCN1'
]
LINKERS = ['', 'C', 'CC', 'O', 'N', 'C(=O)', 'S', 'C=C', 'C#C', 'CO', 'CN']

def generate_smiles(rng, n):
    smiles_set = set()
    attempts = 0
    while len(smiles_set) < n and attempts < n * 100:
        attempts += 1
        n_frag = rng.randint(2, 5)
        frags = [rng.choice(FRAGMENTS) for _ in range(n_frag)]
        mol_str = frags[0]
        for f in frags[1:]:
            mol_str += rng.choice(LINKERS) + f
        try:
            mol = Chem.MolFromSmiles(mol_str)
        except:
            mol = None
        if mol is None:
            continue
        try:
            smiles = Chem.MolToSmiles(mol)
        except:
            continue
        smiles_set.add(smiles)
    return list(smiles_set)

def text_to_embedding(text, dim=32):
    emb = np.zeros(dim)
    for w in text.split():
        h = int(hashlib.md5(w.encode()).hexdigest(), 16) % dim
        emb[h] += 1.0
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm
    return emb

TEXT_VOCAB = [
    'kinase', 'inhibitor', 'agonist', 'antagonist', 'GPCR', 'ion channel', 'protease',
    'nuclear receptor', 'epigenetic', 'oxidative stress', 'inflammation', 'apoptosis',
    'cell cycle', 'cytoskeleton', 'mitochondria', 'ER stress', 'autophagy', 'neuro',
    'cardio', 'immuno', 'metabolism', 'transporter', 'phosphatase', 'HDAC', 'COX',
    'lipoxygenase', 'tyrosine', 'serine', 'threonine', 'receptor', 'ligand', 'modulator',
    'antibacterial', 'antiviral', 'antifungal', 'antiparasitic', 'chemotherapy'
]

def generate_text(rng):
    n = rng.randint(3, 8)
    return ' '.join(rng.choice(TEXT_VOCAB) for _ in range(n))

def compute_descriptors(mol):
    return np.array([
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        QED.qed(mol),
        sascorer.calculateScore(mol),
        Descriptors.FractionCSP3(mol)
    ])

def compute_features(smiles, text_emb):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_arr = np.array(fp)
    desc = compute_descriptors(mol)
    return np.concatenate([fp_arr, desc, text_emb])

def main():
    os.makedirs('/app/data', exist_ok=True)
    rng = random.Random(SEED)
    np_rng = np.random.RandomState(SEED)

    ref_smiles = generate_smiles(rng, 500)
    cand_smiles = generate_smiles(random.Random(SEED+1), 20000)

    ref_text = [generate_text(rng) for _ in range(len(ref_smiles))]
    cand_text = [generate_text(random.Random(SEED+1000+i)) for i in range(len(cand_smiles))]

    ref_text_embs = np.array([text_to_embedding(t) for t in ref_text])
    cand_text_embs = np.array([text_to_embedding(t) for t in cand_text])

    feature_dim = 2048 + 10 + 32
    W = np_rng.randn(feature_dim, 256) * 0.1
    b = np_rng.randn(256) * 0.5

    ref_features = []
    for s, emb in zip(ref_smiles, ref_text_embs):
        f = compute_features(s, emb)
        if f is None:
            f = np.zeros(feature_dim)
        ref_features.append(f)
    ref_features = np.array(ref_features)

    ref_morph = ref_features @ W + b + np_rng.randn(len(ref_smiles), 256) * 0.5

    query_idx = np_rng.choice(len(ref_smiles), 3, replace=False)
    query = ref_morph[query_idx].mean(axis=0) + np_rng.randn(256) * 0.1

    query_text = 'kinase inhibitor agonist GPCR inflammation apoptosis'
    query_text_emb = text_to_embedding(query_text)

    ref_df = pd.DataFrame({
        'compound_id': [f'REF{i:04d}' for i in range(len(ref_smiles))],
        'smiles': ref_smiles,
        'text_description': ref_text,
    })
    for j in range(32):
        ref_df[f'text_emb_{j:02d}'] = ref_text_embs[:, j]
    for j in range(256):
        ref_df[f'morph_{j:03d}'] = ref_morph[:, j]
    ref_df.to_csv('/app/data/reference_compounds.csv', index=False)

    cand_df = pd.DataFrame({
        'candidate_id': [f'CAND{i:05d}' for i in range(len(cand_smiles))],
        'smiles': cand_smiles,
        'text_description': cand_text,
    })
    for j in range(32):
        cand_df[f'text_emb_{j:02d}'] = cand_text_embs[:, j]
    cand_df.to_csv('/app/data/candidates.csv', index=False)

    query_df = pd.DataFrame({f'morph_{j:03d}': [query[j]] for j in range(256)})
    query_df.to_csv('/app/data/query_profile.csv', index=False)

    query_text_df = pd.DataFrame({f'text_emb_{j:02d}': [query_text_emb[j]] for j in range(32)})
    query_text_df.to_csv('/app/data/query_text_embedding.csv', index=False)

    print('Data generation complete.')

if __name__ == '__main__':
    main()
