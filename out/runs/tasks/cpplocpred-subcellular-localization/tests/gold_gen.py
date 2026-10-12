import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

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

def get_bin(d):
    if d <= 1: return 0
    elif d == 2: return 1
    elif d <= 4: return 2
    elif d <= 6: return 3
    elif d <= 9: return 4
    elif d <= 13: return 5
    elif d <= 18: return 6
    elif d <= 24: return 7
    elif d <= 30: return 8
    else: return 9

def compute_ddr(sequences):
    n = len(sequences)
    features = np.zeros((n, 200))
    for idx, seq in enumerate(sequences):
        for aa_idx, aa in enumerate(ALPHABET):
            positions = [i for i, c in enumerate(seq) if c == aa]
            hist = np.zeros(10)
            total_pairs = 0
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    d = positions[j] - positions[i]
                    hist[get_bin(d)] += 1
                    total_pairs += 1
            if total_pairs > 0:
                hist /= total_pairs
            features[idx, aa_idx*10:(aa_idx+1)*10] = hist
    return features

_dataset = None
_ref_aucs = None

def _get_dataset():
    global _dataset
    if _dataset is None:
        _dataset = generate_dataset()
    return _dataset

def get_hidden_test_labels():
    _, _, test_df = _get_dataset()
    return test_df[['seq_id', 'sequence', 'localization']].copy()

def get_reference_aucs():
    global _ref_aucs
    if _ref_aucs is not None:
        return _ref_aucs
    train, val, test_df = _get_dataset()
    X_train = compute_ddr(train['sequence'].tolist())
    y_train = train['localization'].tolist()
    X_val = compute_ddr(val['sequence'].tolist())
    y_val = val['localization'].tolist()
    X_all = np.vstack([X_train, X_val])
    y_all = y_train + y_val
    X_test = compute_ddr(test_df['sequence'].tolist())
    
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        loss_function='MultiClass',
        verbose=0
    )
    model.fit(X_all, y_all)
    probs = model.predict_proba(X_test)
    classes = model.classes_
    aucs = {}
    for cls in ['Cytoplasm', 'Nucleus', 'Mitochondria', 'Endo_lysosome', 'Others']:
        idx = list(classes).index(cls)
        y_true = (test_df['localization'] == cls).astype(int)
        aucs[cls] = roc_auc_score(y_true, probs[:, idx])
    _ref_aucs = aucs
    return aucs
