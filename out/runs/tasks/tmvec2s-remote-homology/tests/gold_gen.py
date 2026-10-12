import numpy as np
import os

SEED = 12345
L = 128
K = 20
N_DB = 200
N_QUERIES = 30
N_TRAIN_PAIRS = 2000
AA = list('ACDEFGHIKLMNPQRSTVWY')

def generate_base_structure(family_seed):
    rng = np.random.default_rng(family_seed)
    coords = np.zeros((L, 3))
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    for i in range(1, L):
        rot = rng.normal(scale=0.3, size=3)
        direction += rot
        direction /= np.linalg.norm(direction)
        coords[i] = coords[i-1] + direction
    coords -= coords.mean(axis=0)
    return coords

def kabsch(P, Q):
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    P_rot = Pc @ R.T
    d_i = np.linalg.norm(P_rot - Qc, axis=1)
    d0 = 1.24 * (L - 15)**(1/3) - 1.8
    tm = np.mean(1.0 / (1.0 + (d_i / d0)**2))
    return tm

def generate_all(seed=SEED):
    rng = np.random.default_rng(seed)
    family_probs = []
    family_structures = []
    for k in range(K):
        p = rng.dirichlet(np.ones(20)*0.5)
        family_probs.append(p)
        base = generate_base_structure(seed + k*1000)
        family_structures.append(base)
    family_seqs = []
    for k in range(K):
        seq = ''.join(rng.choice(AA, size=L, p=family_probs[k]))
        family_seqs.append(seq)
    db_domains = []
    for i in range(N_DB):
        fam = rng.integers(0, K)
        mu = rng.beta(2, 5)
        sigma = rng.uniform(0.5, 2.5)
        cons = family_seqs[fam]
        seq = list(cons)
        for pos in range(L):
            if rng.random() < mu:
                seq[pos] = rng.choice(AA)
        seq = ''.join(seq)
        base = family_structures[fam]
        noise = rng.normal(scale=sigma, size=(L, 3))
        struct = base + noise
        A = rng.normal(size=(3,3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:,0] = -Q[:,0]
        struct = struct @ Q.T
        db_domains.append({'id': f'DB{i+1:03d}', 'family': fam, 'seq': seq, 'struct': struct, 'mu': mu, 'sigma': sigma})
    queries = []
    for i in range(N_QUERIES):
        if i < 5:
            p = rng.dirichlet(np.ones(20)*0.5)
            cons = ''.join(rng.choice(AA, size=L, p=p))
            base = generate_base_structure(seed + 50000 + i*1000)
            fam = K + i
        else:
            fam = rng.integers(0, K)
            cons = family_seqs[fam]
            base = family_structures[fam]
        mu = rng.beta(2, 5)
        sigma = rng.uniform(0.5, 2.5)
        seq = list(cons)
        for pos in range(L):
            if rng.random() < mu:
                seq[pos] = rng.choice(AA)
        seq = ''.join(seq)
        noise = rng.normal(scale=sigma, size=(L, 3))
        struct = base + noise
        A = rng.normal(size=(3,3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:,0] = -Q[:,0]
        struct = struct @ Q.T
        queries.append({'id': f'Q{i+1:02d}', 'family': fam, 'seq': seq, 'struct': struct, 'mu': mu, 'sigma': sigma})
    return db_domains, queries

def get_gold():
    db_domains, queries = generate_all()
    gold = {}
    for q in queries:
        for d in db_domains:
            tm = kabsch(q['struct'], d['struct'])
            gold[(q['id'], d['id'])] = tm
    return gold
