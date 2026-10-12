import numpy as np
import os

np.random.seed(42)
K = 4
M = 40
N = 200
seq_len = 2 * M
aas = list('ACDEFGHIKLMNPQRSTVWY')
groups = {
    'A':1, 'V':1, 'L':1, 'I':1, 'M':1, 'F':1, 'W':1, 'Y':1,
    'S':2, 'T':2, 'N':2, 'Q':2, 'C':2, 'G':2, 'P':2,
    'K':3, 'R':3, 'H':3,
    'D':4, 'E':4
}
group_to_aas = {1: [], 2: [], 3: [], 4: []}
for aa, g in groups.items():
    group_to_aas[g].append(aa)
pair_params = []
for p in range(M):
    J = np.zeros((K,K))
    for i in range(K):
        for j in range(i+1,K):
            val = np.random.normal(0, 0.8)
            J[i,j] = val
            J[j,i] = val
    h = np.zeros(K)
    pair_params.append((h, J))
pair_obs = []
for p in range(M):
    h, J = pair_params[p]
    P = np.zeros((K,K))
    for a in range(K):
        for b in range(K):
            P[a,b] = np.exp(h[a] + h[b] + J[a,b])
    P /= P.sum()
    obs = []
    for s in range(N):
        flat = P.flatten()
        idx = np.random.choice(K*K, p=flat)
        a = idx // K
        b = idx % K
        obs.append((a,b))
    pair_obs.append(obs)
sequences = []
for s in range(N):
    seq = []
    for p in range(M):
        a, b = pair_obs[p][s]
        aa_a = np.random.choice(group_to_aas[a+1])
        aa_b = np.random.choice(group_to_aas[b+1])
        seq.append(aa_a)
        seq.append(aa_b)
    sequences.append(''.join(seq))
sequences_missing = []
for seq in sequences:
    seq_list = list(seq)
    for i in range(len(seq_list)):
        if np.random.rand() < 0.03:
            seq_list[i] = 'X'
    sequences_missing.append(''.join(seq_list))
with open('alignment.fasta', 'w') as f:
    for i, seq in enumerate(sequences_missing):
        print(f'>seq{i+1}', file=f)
        print(seq, file=f)
with open('contacts.tsv', 'w') as f:
    print('pair_id' + chr(9) + 'site_i' + chr(9) + 'site_j', file=f)
    for p in range(M):
        print(f'{p+1}' + chr(9) + f'{2*p+1}' + chr(9) + f'{2*p+2}', file=f)
with open('alphabet_map.tsv', 'w') as f:
    print('amino_acid' + chr(9) + 'group', file=f)
    for aa in aas:
        print(f'{aa}' + chr(9) + f'{groups[aa]}', file=f)
