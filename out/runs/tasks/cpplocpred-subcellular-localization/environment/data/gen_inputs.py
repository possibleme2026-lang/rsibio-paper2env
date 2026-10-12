#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

SEED = 20260728
ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")

CLASS_MOTIFS = {
    "Cytoplasm": None,
    "Nucleus": ["RKK", "RRK", "KRK", "RKR", "RKKR", "RRKK"],
    "Mitochondria": ["RR", "KK", "RKR", "KKR", "RRK", "RKK"],
    "Endo_lysosome": ["GPC", "GPG", "PGP", "CPG", "GCP", "PCG"],
    "Others": ["WLL", "YFL", "LWL", "FWL", "WYF", "LLW"]
}

BG_FREQ = {
    'A':0.08, 'C':0.02, 'D':0.05, 'E':0.06, 'F':0.04, 'G':0.07,
    'H':0.02, 'I':0.06, 'K':0.06, 'L':0.09, 'M':0.02, 'N':0.04,
    'P':0.05, 'Q':0.04, 'R':0.06, 'S':0.07, 'T':0.05, 'V':0.07,
    'W':0.01, 'Y':0.03
}
tot = sum(BG_FREQ.values())
for k in BG_FREQ: BG_FREQ[k] /= tot
PROBS = np.array([BG_FREQ[a] for a in ALPHABET])

def generate_sequence(rng, cls, length):
    if cls == "Cytoplasm" or CLASS_MOTIFS[cls] is None:
        return ''.join(rng.choice(ALPHABET, size=length, p=PROBS))
    else:
        motifs = CLASS_MOTIFS[cls]
        seq = []
        while len(seq) < length:
            motif = motifs[rng.randint(len(motifs))]
            seq.extend(list(motif))
        seq = seq[:length]
        mutation_rate = 0.20
        for i in range(len(seq)):
            if rng.rand() < mutation_rate:
                seq[i] = rng.choice(ALPHABET, p=PROBS)
        return ''.join(seq)

def generate_dataset():
    rng = np.random.RandomState(SEED)
    counts = {
        "Cytoplasm": 663,
        "Nucleus": 287,
        "Mitochondria": 57,
        "Endo_lysosome": 186,
        "Others": 328
    }
    train_val = []
    for cls, n in counts.items():
        for i in range(n):
            length = rng.randint(5, 31)
            seq = generate_sequence(rng, cls, length)
            train_val.append((f"{cls[:3]}_{i:04d}", seq, cls))
    df = pd.DataFrame(train_val, columns=["seq_id", "sequence", "localization"])
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_train = int(0.8 * len(df))
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:].copy()
    
    test_counts = {
        "Cytoplasm": 218,
        "Nucleus": 94,
        "Mitochondria": 19,
        "Endo_lysosome": 61,
        "Others": 108
    }
    test = []
    for cls, n in test_counts.items():
        for i in range(n):
            length = rng.randint(5, 31)
            seq = generate_sequence(rng, cls, length)
            test.append((f"test_{cls[:3]}_{i:04d}", seq, cls))
    test_df = pd.DataFrame(test, columns=["seq_id", "sequence", "localization"])
    test_df = test_df.sample(frac=1, random_state=SEED+1).reset_index(drop=True)
    
    return train, val, test_df

if __name__ == "__main__":
    os.makedirs("/app/data", exist_ok=True)
    train, val, test_df = generate_dataset()
    train.to_csv("/app/data/train.csv", index=False)
    val.to_csv("/app/data/val.csv", index=False)
    test_df[["seq_id", "sequence"]].to_csv("/app/data/test.csv", index=False)
    print("Data generated.")
