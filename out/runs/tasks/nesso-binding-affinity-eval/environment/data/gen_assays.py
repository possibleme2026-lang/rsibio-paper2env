#!/usr/bin/env python3
import os, random, numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from Bio import SeqIO

SEED = 42
OUT_DIR = "/app/data"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "assays"), exist_ok=True)
rng = random.Random(SEED)
np.random.seed(SEED)
AA = list("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AVILMFWY")
SMILES_LIST = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "CC(=O)NC1=CC=C(C=C1)O",
    "C1=CC=C(C=C1)C2=CC=CC=C2",
    "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
    "CC1=CC=C(C=C1)C(C)C(=O)O",
    "CCOC(=O)C1=CC=CC=C1",
    "CC(=O)OC1=CC=CC=C1",
    "C1=CC=C(C=C1)N",
    "CC(C)NCC(COC1=CC=CC=C1)O",
    "CN(C)CCN1C2=CC=CC=C2SC3=CC=CC=C31",
    "CC1=CC(=C(C=C1)C(=O)O)O",
    "C1=CC=C(C=C1)C(=O)O",
    "CC1=C(C=CC(=C1)C(=O)O)O",
    "C1=CC=C(C=C1)CO",
    "CC(=O)C1=CC=CC=C1",
    "C1CCCCC1",
    "C1=CC=NC=C1",
    "C1=CC=CN=C1",
    "C1=CN=CN=C1",
    "CC(C)C1=CC=CC=C1",
    "C1=CC=C(C=C1)Cl",
    "C1=CC=C(C=C1)Br",
    "C1=CC=C(C=C1)F",
    "CC(=O)N1C=CN=C1",
    "CC(C)(C)C1=CC=CC=C1",
    "C1=CC=C(C=C1)C#N",
    "C1=CC=C(C=C1)[N+](=O)[O-]",
    "CC1=CC=CC=C1",
    "C1=CC=C(C=C1)OC",
    "C1=CC=C(C=C1)SC",
    "C1=CC=C(C=C1)S(=O)(=O)N",
    "CC(=O)C1=CC=C(C=C1)O",
    "CC(C)(C)C1=CC=C(C=C1)O",
    "C1=CC=C(C=C1)C(=O)N",
    "C1=CC=C(C=C1)C(=O)Cl",
    "C1=CC=C(C=C1)C(=O)OC",
    "C1=CC=C(C=C1)C(=O)NCC",
    "C1=CC=C(C=C1)CCN",
    "C1=CC=C(C=C1)CC(=O)O",
    "C1=CC=C(C=C1)CC(C(=O)O)N",
    "C1=CC=C(C=C1)C(C(=O)O)N",
    "C1=CC=C(C=C1)C(C(=O)O)O",
    "CC1=CC=C(C=C1)C(C)C(=O)O",
    "CC1=CC=CC=C1C(=O)O",
    "CC1=C(C=CC(=C1)C(=O)O)O",
    "C1=CC=C(C=C1)C(=O)OC",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CC(C)NCC(COC1=CC=CC=C1)O",
]
SMILES_LIST = list(dict.fromkeys(SMILES_LIST))
ligands = []
for i, smi in enumerate(SMILES_LIST):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        ligands.append({"ligand_id": f"lig_{i:03d}", "smiles": smi})
train_proteins = []
for i in range(100):
    length = rng.randint(250, 350)
    seq = "".join(rng.choice(AA) for _ in range(length))
    train_proteins.append({"protein_id": f"train_{i:03d}", "sequence": seq})
with open(os.path.join(OUT_DIR, "train_proteins.fasta"), "w") as f:
    for p in train_proteins:
        f.write(f">{p['protein_id']}\n{p['sequence']}\n")
assay_proteins = []
for i in range(30):
    if i < 15:
        train_p = rng.choice(train_proteins)
        seq = list(train_p["sequence"])
        n_mut = rng.randint(int(0.1*len(seq)), int(0.3*len(seq)))
        positions = rng.sample(range(len(seq)), n_mut)
        for pos in positions:
            seq[pos] = rng.choice(AA)
        seq = "".join(seq)
    else:
        length = rng.randint(250, 350)
        seq = "".join(rng.choice(AA) for _ in range(length))
    assay_proteins.append({"protein_id": f"assay_prot_{i:02d}", "sequence": seq})
for p in assay_proteins:
    seq = p["sequence"]
    start = rng.randint(0, len(seq)-20)
    pocket = seq[start:start+20]
    p["pocket_start"] = start
    p["pocket_end"] = start+20
    p["pocket_hydrophobicity"] = sum(1 for aa in pocket if aa in HYDROPHOBIC) / 20.0
assay_metadata = []
for i, p in enumerate(assay_proteins):
    assay_id = f"assay_{i:02d}"
    protein_id = p["protein_id"]
    seq = p["sequence"]
    pocket_hydro = p["pocket_hydrophobicity"]
    n_compounds = rng.randint(80, 120)
    compounds = rng.choices(ligands, k=n_compounds)
    rows = []
    for j, lig in enumerate(compounds):
        mol = Chem.MolFromSmiles(lig["smiles"])
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        aromatic = Lipinski.NumAromaticRings(mol)
        tpsa = Descriptors.TPSA(mol)
        true_pkd = (5.0 + 0.5*logp - 0.005*mw + 1.0*pocket_hydro + 0.3*hbd - 0.2*tpsa/100 + 0.5*aromatic + np.random.normal(0, 0.4))
        if i < 15:
            nesso_noise = np.random.normal(0, 0.5)
            boltz_noise = np.random.normal(0, 0.5)
        else:
            nesso_noise = np.random.normal(0, 0.7)
            boltz_noise = np.random.normal(0, 1.0)
        nesso_pred = true_pkd + nesso_noise
        boltz_pred = true_pkd + boltz_noise
        measured = true_pkd
        if rng.random() < 0.05:
            measured = np.nan
        if rng.random() < 0.05:
            nesso_pred = np.nan
        if rng.random() < 0.05:
            boltz_pred = np.nan
        rows.append({
            "compound_id": f"{assay_id}_cpd_{j:03d}",
            "smiles": lig["smiles"],
            "protein_id": protein_id,
            "protein_sequence": seq,
            "measured_pKd": measured,
            "nesso_pred": nesso_pred,
            "boltz_pred": boltz_pred,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, f"assays/{assay_id}.csv"), index=False)
    assay_metadata.append({"assay_id": assay_id, "protein_id": protein_id})
pd.DataFrame(assay_metadata).to_csv(os.path.join(OUT_DIR, "assay_metadata.csv"), index=False)
selectivity_rows = []
for i in range(5):
    assay_id = f"sel_{i:02d}"
    on_p = rng.choice(assay_proteins)
    on_seq = on_p["sequence"]
    off_seq = list(on_seq)
    n_mut = int(0.2 * len(on_seq))
    positions = rng.sample(range(len(on_seq)), n_mut)
    for pos in positions:
        off_seq[pos] = rng.choice(AA)
    off_seq = "".join(off_seq)
    on_pocket = on_seq[on_p["pocket_start"]:on_p["pocket_end"]]
    on_hydro = sum(1 for aa in on_pocket if aa in HYDROPHOBIC) / len(on_pocket)
    off_pocket = off_seq[on_p["pocket_start"]:on_p["pocket_end"]]
    off_hydro = sum(1 for aa in off_pocket if aa in HYDROPHOBIC) / len(off_pocket)
    compounds = rng.choices(ligands, k=50)
    for j, lig in enumerate(compounds):
        mol = Chem.MolFromSmiles(lig["smiles"])
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        on_true = (5.0 + 0.5*logp - 0.005*mw + 1.0*on_hydro + 0.3*hbd - 0.2*tpsa/100 + np.random.normal(0, 0.3))
        off_true = (4.0 + 0.3*logp - 0.005*mw + 0.8*off_hydro + 0.2*hbd - 0.2*tpsa/100 + np.random.normal(0, 0.3))
        on_nesso = on_true + np.random.normal(0, 0.5)
        off_nesso = off_true + np.random.normal(0, 0.5)
        on_boltz = on_true + np.random.normal(0, 0.7)
        off_boltz = off_true + np.random.normal(0, 0.7)
        selectivity_rows.append({
            "assay_id": assay_id,
            "compound_id": f"{assay_id}_cpd_{j:03d}",
            "smiles": lig["smiles"],
            "on_protein_id": on_p["protein_id"],
            "on_protein_sequence": on_seq,
            "on_measured_pKd": on_true,
            "on_nesso_pred": on_nesso,
            "on_boltz_pred": on_boltz,
            "off_protein_id": f"off_{on_p['protein_id']}",
            "off_protein_sequence": off_seq,
            "off_measured_pKd": off_true,
            "off_nesso_pred": off_nesso,
            "off_boltz_pred": off_boltz,
        })
pd.DataFrame(selectivity_rows).to_csv(os.path.join(OUT_DIR, "selectivity.csv"), index=False)
print("Data generation complete.")
