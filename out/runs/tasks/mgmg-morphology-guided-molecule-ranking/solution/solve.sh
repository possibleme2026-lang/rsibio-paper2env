#!/bin/bash
set -euo pipefail
cat > /app/solution.py << 'PY'
import os, sys, math
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, QED, DataStructs
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from sklearn.linear_model import Ridge
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

DATA_DIR = '/app/data'
OUTPUT_FILE = '/app/output/selected_molecules.csv'
N_MORPH = 256
N_TEXT = 32
N_FP = 2048
N_DESC = 10
FEATURE_DIM = N_FP + N_DESC + N_TEXT
ALPHA = 1.0
TOP_K = 20
DIVERSITY_THRESHOLD = 0.4

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
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=N_FP)
    fp_arr = np.array(fp)
    desc = compute_descriptors(mol)
    return np.concatenate([fp_arr, desc, text_emb])

def main():
    ref = pd.read_csv(os.path.join(DATA_DIR, 'reference_compounds.csv'))
    cand = pd.read_csv(os.path.join(DATA_DIR, 'candidates.csv'))
    query = pd.read_csv(os.path.join(DATA_DIR, 'query_profile.csv'))
    query_text = pd.read_csv(os.path.join(DATA_DIR, 'query_text_embedding.csv'))

    text_cols = [f'text_emb_{j:02d}' for j in range(N_TEXT)]
    ref_text_embs = ref[text_cols].values
    X_ref = []
    for smi, emb in zip(ref['smiles'], ref_text_embs):
        f = compute_features(smi, emb)
        if f is None:
            f = np.zeros(FEATURE_DIM)
        X_ref.append(f)
    X_ref = np.array(X_ref)
    y_ref = ref[[f'morph_{j:03d}' for j in range(N_MORPH)]].values

    model = Ridge(alpha=ALPHA)
    model.fit(X_ref, y_ref)

    cand_text_embs = cand[text_cols].values
    X_cand = []
    for smi, emb in zip(cand['smiles'], cand_text_embs):
        f = compute_features(smi, emb)
        if f is None:
            f = np.zeros(FEATURE_DIM)
        X_cand.append(f)
    X_cand = np.array(X_cand)

    y_pred = model.predict(X_cand)

    q = query.values.flatten()
    q_text = query_text.values.flatten()

    morph_scores = np.array([np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q)) for p in y_pred])
    text_scores = np.array([np.dot(emb, q_text) / (np.linalg.norm(emb) * np.linalg.norm(q_text)) for emb in cand_text_embs])
    combined = 0.8 * morph_scores + 0.2 * text_scores

    cand_scores = cand.copy()
    cand_scores['morph_score'] = morph_scores
    cand_scores['text_score'] = text_scores
    cand_scores['combined_score'] = combined

    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains_cat = FilterCatalog.FilterCatalog(pains_params)

    rows = []
    for idx, row in cand_scores.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is None:
            continue
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        qed = QED.qed(mol)
        sa = sascorer.calculateScore(mol)
        if not (250 <= mw <= 500 and -0.5 <= logp <= 4.5 and tpsa <= 120 and hbd <= 5 and hba <= 10 and qed >= 0.5 and sa <= 4.0):
            continue
        entry = pains_cat.GetFirstMatch(mol)
        if entry is not None:
            continue
        smi = row['smiles']
        if any(patt in smi for patt in ['[N+](=O)[O-]', 'C(=O)Cl', 'S(=O)(=O)Cl', 'C(=O)Br']):
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=N_FP)
        rows.append({
            'candidate_id': row['candidate_id'],
            'smiles': Chem.MolToSmiles(mol),
            'combined_score': row['combined_score'],
            'morphology_score': row['morph_score'],
            'text_score': row['text_score'],
            'MW': mw,
            'logP': logp,
            'QED': qed,
            'SA_score': sa,
            'fp': fp
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('combined_score', ascending=False)

    selected = []
    selected_fps = []
    for _, row in df.iterrows():
        fp = row['fp']
        max_sim = 0.0
        for sfp in selected_fps:
            sim = DataStructs.TanimotoSimilarity(fp, sfp)
            if sim > max_sim:
                max_sim = sim
        if max_sim <= DIVERSITY_THRESHOLD:
            selected.append(row)
            selected_fps.append(fp)
            if len(selected) == TOP_K:
                break

    out_df = pd.DataFrame(selected)
    out_df = out_df.drop(columns=['fp'])
    out_df = out_df.reset_index(drop=True)
    out_df['rank'] = out_df.index + 1
    out_df = out_df[['rank', 'candidate_id', 'smiles', 'combined_score', 'morphology_score', 'text_score', 'MW', 'logP', 'QED', 'SA_score']]
    out_df['combined_score'] = out_df['combined_score'].round(4)
    out_df['morphology_score'] = out_df['morphology_score'].round(4)
    out_df['text_score'] = out_df['text_score'].round(4)
    os.makedirs('/app/output', exist_ok=True)
    out_df.to_csv(OUTPUT_FILE, index=False)

if __name__ == '__main__':
    main()
PY
python /app/solution.py
