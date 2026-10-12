#!/usr/bin/env python3
import os, sys, json, math
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
REWARD_FILE = '/logs/verifier/reward.json'
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

def run_pipeline():
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

    gold = pd.DataFrame(selected)
    if len(gold) > 0:
        gold = gold.drop(columns=['fp'])
    return gold, cand_scores

def grade():
    constraints = []
    if not os.path.exists(OUTPUT_FILE):
        constraints.append(('output_exists', False))
        score = 0.0
        os.makedirs(os.path.dirname(REWARD_FILE), exist_ok=True)
        with open(REWARD_FILE, 'w') as f:
            json.dump({'score': score, 'reward': score}, f)
        print('Output file missing.')
        return

    agent = pd.read_csv(OUTPUT_FILE)
    required_cols = ['rank','candidate_id','smiles','combined_score','morphology_score','text_score','MW','logP','QED','SA_score']
    if list(agent.columns) != required_cols:
        constraints.append(('columns', False))
    else:
        constraints.append(('columns', True))

    if len(agent) != TOP_K:
        constraints.append(('num_rows', False))
    else:
        constraints.append(('num_rows', True))

    valid_smiles = True
    canonical_smiles = []
    for smi in agent['smiles']:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_smiles = False
            break
        canonical_smiles.append(Chem.MolToSmiles(mol))
    if not valid_smiles:
        constraints.append(('valid_smiles', False))
        constraints.append(('unique_smiles', False))
    else:
        constraints.append(('valid_smiles', True))
        constraints.append(('unique_smiles', len(set(canonical_smiles)) == len(canonical_smiles)))

    props_ok = True
    for smi in agent['smiles']:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            props_ok = False
            break
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        qed = QED.qed(mol)
        sa = sascorer.calculateScore(mol)
        if not (250 <= mw <= 500 and -0.5 <= logp <= 4.5 and tpsa <= 120 and hbd <= 5 and hba <= 10 and qed >= 0.5 and sa <= 4.0):
            props_ok = False
            break
    constraints.append(('drug_likeness', props_ok))

    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains_cat = FilterCatalog.FilterCatalog(pains_params)
    pains_ok = True
    for smi in agent['smiles']:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            pains_ok = False
            break
        if pains_cat.GetFirstMatch(mol) is not None:
            pains_ok = False
            break
    constraints.append(('pains', pains_ok))

    reactive_ok = True
    for smi in agent['smiles']:
        for patt in ['[N+](=O)[O-]', 'C(=O)Cl', 'S(=O)(=O)Cl', 'C(=O)Br']:
            if patt in smi:
                reactive_ok = False
                break
        if not reactive_ok:
            break
    constraints.append(('reactive_groups', reactive_ok))

    if valid_smiles:
        fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=N_FP) for smi in agent['smiles']]
        max_sim = 0.0
        for i in range(len(fps)):
            for j in range(i+1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                if sim > max_sim:
                    max_sim = sim
        constraints.append(('diversity', max_sim <= DIVERSITY_THRESHOLD))
    else:
        constraints.append(('diversity', False))

    gold, cand_scores = run_pipeline()

    score_map = {}
    for _, row in cand_scores.iterrows():
        score_map[row['candidate_id']] = {
            'combined_score': row['combined_score'],
            'morphology_score': row['morph_score'],
            'text_score': row['text_score']
        }
    score_ok = True
    for _, row in agent.iterrows():
        cid = row['candidate_id']
        if cid not in score_map:
            score_ok = False
            break
        if abs(row['combined_score'] - score_map[cid]['combined_score']) > 1e-3:
            score_ok = False
            break
        if abs(row['morphology_score'] - score_map[cid]['morphology_score']) > 1e-3:
            score_ok = False
            break
        if abs(row['text_score'] - score_map[cid]['text_score']) > 1e-3:
            score_ok = False
            break
    constraints.append(('score_consistency', score_ok))

    gold_ids = set(gold['candidate_id'].tolist()) if len(gold) > 0 else set()
    agent_ids = set(agent['candidate_id'].tolist())
    overlap = len(gold_ids & agent_ids)
    constraints.append(('overlap_18', overlap >= 18))

    if len(gold) > 0 and len(agent) > 0:
        constraints.append(('top1_match', gold.iloc[0]['candidate_id'] == agent.iloc[0]['candidate_id']))
    else:
        constraints.append(('top1_match', False))

    constraints.append(('exact_set', gold_ids == agent_ids))

    passed = sum(1 for _, ok in constraints if ok)
    score = passed / len(constraints)
    os.makedirs(os.path.dirname(REWARD_FILE), exist_ok=True)
    with open(REWARD_FILE, 'w') as f:
        json.dump({'score': score, 'reward': score}, f)

    for name, ok in constraints:
        print(f'{name}: {ok}')
    print(f'Score: {score}')

if __name__ == '__main__':
    grade()
