#!/bin/bash
set -euo pipefail

mkdir -p /app/output

python3 << 'EOF'
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os

def read_fasta(path):
    seqs = {}
    with open(path) as f:
        name = None
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name is not None:
                    seqs[name] = ''.join(seq)
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
        if name is not None:
            seqs[name] = ''.join(seq)
    return seqs

db_seqs = read_fasta('/app/data/database.fasta')
query_seqs = read_fasta('/app/data/queries.fasta')
train = pd.read_csv('/app/data/train_pairs.tsv', sep='\t')

AA = list('ACDEFGHIKLMNPQRSTVWY')
aa_idx = {a:i for i,a in enumerate(AA)}

def comp(seq):
    vec = np.zeros(20)
    for a in seq:
        vec[aa_idx[a]] += 1
    return vec / len(seq)

def dipep(seq):
    vec = np.zeros(400)
    for i in range(len(seq)-1):
        a1 = aa_idx[seq[i]]
        a2 = aa_idx[seq[i+1]]
        vec[a1*20 + a2] += 1
    return vec / (len(seq)-1)

db_features = {}
for name, seq in db_seqs.items():
    db_features[name] = np.concatenate([comp(seq), dipep(seq)])

query_features = {}
for name, seq in query_seqs.items():
    query_features[name] = np.concatenate([comp(seq), dipep(seq)])

X_train = []
y_train = []
for _, row in train.iterrows():
    f1 = db_features[row['id1']]
    f2 = db_features[row['id2']]
    feat = np.concatenate([f1, f2, np.abs(f1-f2), f1*f2])
    X_train.append(feat)
    y_train.append(row['tm_score'])
X_train = np.array(X_train)
y_train = np.array(y_train)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

os.makedirs('/app/output', exist_ok=True)
with open('/app/output/predictions.tsv', 'w') as f:
    f.write('query_id\ttarget_id\tpredicted_tm\n')
    for qname, qfeat in query_features.items():
        for dname, dfeat in db_features.items():
            feat = np.concatenate([qfeat, dfeat, np.abs(qfeat-dfeat), qfeat*dfeat])
            pred = model.predict([feat])[0]
            pred = np.clip(pred, 0, 1)
            f.write(qname + '\t' + dname + '\t' + f'{pred:.4f}' + '\n')
EOF
