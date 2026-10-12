#!/bin/bash
set -euo pipefail

cat > /app/solve_mcsa.py << 'PY'
import csv, json, os
import numpy as np

L = 18
M = 8
AA = "ACDEFGHIKLMNPQRSTVWY"

ids = []
seqs = []
with open('/app/data/cyclic_peptides.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ids.append(row['peptide_id'])
        seqs.append(row['sequence'])

N = len(seqs)
aa_to_idx = {aa: i for i, aa in enumerate(AA)}
X = np.array([[aa_to_idx[c] for c in s] for s in seqs])

ref = X[0].copy()
rotations = np.zeros(N, dtype=int)
for iteration in range(5):
    for i in range(N):
        best_score = -1
        best_r = 0
        for r in range(L):
            rotated = np.roll(X[i], -r)
            score = np.sum(rotated == ref)
            if score > best_score:
                best_score = score
                best_r = r
        rotations[i] = best_r
    aligned = np.zeros((N, L), dtype=int)
    for i in range(N):
        aligned[i] = np.roll(X[i], -rotations[i])
    new_ref = np.zeros(L, dtype=int)
    for j in range(L):
        counts = np.bincount(aligned[:, j], minlength=len(AA))
        new_ref[j] = np.argmax(counts)
    ref = new_ref

best_start = 0
best_score = -1
for start in range(L):
    score = 0
    for j in range(M):
        pos = (start + j) % L
        counts = np.bincount(aligned[:, pos], minlength=len(AA))
        score += np.max(counts)
    if score > best_score:
        best_score = score
        best_start = start

shift = best_start
final_rotations = (rotations + shift) % L

final_aligned = np.zeros((N, L), dtype=int)
for i in range(N):
    final_aligned[i] = np.roll(X[i], -final_rotations[i])

motif = []
for j in range(M):
    counts = np.bincount(final_aligned[:, j], minlength=len(AA))
    motif.append(AA[np.argmax(counts)])
motif = ''.join(motif)

os.makedirs('/app/output', exist_ok=True)
out = {
    'consensus_motif': motif,
    'rotations': {ids[i]: int(final_rotations[i]) for i in range(N)}
}
with open('/app/output/mcsa_results.json', 'w') as f:
    json.dump(out, f, indent=2)
PY

python /app/solve_mcsa.py
