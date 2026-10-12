#!/usr/bin/env python3
'''Deterministic generator of the simulated Hi-C subsampling benchmark.

Writes, under <out>:
  matrices/res100kb/rep<K>/depth_<D>M_sub<J>.npz  (300 x 300 int32, 100 kb bins, 30 Mb)
  matrices/res25kb/rep<K>/depth_<D>M_sub<J>.npz   (400 x 400 int32, 25 kb bins, 10 Mb)
with D in [1, 2, 5, 10, 20, 30, 60, 100] million read pairs, K in 1..4 and
J in 1..3.  Dense contact probabilities are built from a power-law distance
decay modulated by loops, A/B compartments and TAD-like domains; each matrix is
an independent Poisson draw of the subsampled counts (upper triangle, then
mirrored).  Everything is driven by SEED so the data are reproducible.
'''
import os
import sys

import numpy as np
import scipy.sparse as sp

SEED = 20260728
DEPTHS = [1, 2, 5, 10, 20, 30, 60, 100]
NSUB = 3
REPS = [1, 2, 3, 4]
LOOP_AMP = {1: 0.050, 2: 0.075, 3: 0.100, 4: 0.125}
LOOP_SIZE_BINS = {1: 5.0, 2: 5.4, 3: 5.8, 4: 6.2}
LOOP_SIGMA_BINS = 2.5
COMP_STR = {1: 0.195, 2: 0.200, 3: 0.205, 4: 0.210}
DOM_STR = {1: 0.50, 2: 0.55, 3: 0.60, 4: 0.65}

README = '''Simulated Hi-C subsampling benchmark
=====================================

matrices/res100kb/rep<K>/depth_<D>M_sub<J>.npz
    300 x 300 symmetric int32 count matrix, 100 kb bins, 30 Mb simulated chromosome.
matrices/res25kb/rep<K>/depth_<D>M_sub<J>.npz
    400 x 400 symmetric int32 count matrix, 25 kb bins, 10 Mb simulated region.

K = 1..4  biological replicate
D = 1, 2, 5, 10, 20, 30, 60, 100  millions of read pairs
J = 1..3  independent subsample realisation at that depth

Files are scipy.sparse CSR matrices written with scipy.sparse.save_npz.
M[i, j] is the number of simulated read pairs joining bin i and bin j.  The
diagonal (self ligation) is stored but is excluded from every analysis.  The
upper triangle holds the sampled counts; the lower triangle mirrors it.
'''


def domain_edges(rep, n=400):
    rng = np.random.default_rng(SEED + 7919 * rep)
    target = int(round(n / float(12 + rep)))
    edges = [0]
    pos = 0
    while True:
        pos += int(target + rng.integers(-4, 5))
        if pos > n - 6:
            break
        edges.append(pos)
    edges.append(n)
    return edges


def expected_100kb(rep, n=300):
    idx = np.arange(n)
    s = np.maximum(np.abs(idx[:, None] - idx[None, :]), 1).astype(float)
    M = s ** (-1.10)
    M = M * (1.0 + LOOP_AMP[rep] * np.exp(
        -0.5 * ((s - LOOP_SIZE_BINS[rep]) / LOOP_SIGMA_BINS) ** 2))
    sign = np.where(((idx // 12) % 2) == 0, 1.0, -1.0)
    M = M * (1.0 + COMP_STR[rep] * np.outer(sign, sign))
    np.fill_diagonal(M, 0.0)
    return M


def expected_25kb(rep, n=400):
    idx = np.arange(n)
    s = np.maximum(np.abs(idx[:, None] - idx[None, :]), 1).astype(float)
    M = s ** (-1.10)
    edges = domain_edges(rep, n)
    lab = np.zeros(n, dtype=int)
    for k in range(len(edges) - 1):
        lab[edges[k]:edges[k + 1]] = k
    same = (lab[:, None] == lab[None, :])
    M = M * np.where(same, 1.0 + DOM_STR[rep], 1.0 - 0.5 * DOM_STR[rep])
    np.fill_diagonal(M, 0.0)
    return M


def unit_matrix(M):
    iu = np.triu_indices_from(M, k=1)
    return M / M[iu].sum()


def sample_unit(U, depth_M, rng):
    lam = np.triu(U * (depth_M * 1e6), k=1)
    cnt = rng.poisson(lam)
    full = (cnt + cnt.T).astype(np.int32)
    return sp.csr_matrix(full)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else '/app/data'
    if os.path.basename(outdir.rstrip('/')) != 'data' and len(sys.argv) == 1:
        pass
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    truths = {'res100kb': {}, 'res25kb': {}}
    for rep in REPS:
        truths['res100kb'][rep] = unit_matrix(expected_100kb(rep))
        truths['res25kb'][rep] = unit_matrix(expected_25kb(rep))
    for res in ('res100kb', 'res25kb'):
        for rep in REPS:
            d = os.path.join(outdir, 'matrices', res, 'rep%d' % rep)
            os.makedirs(d, exist_ok=True)
            for depth in DEPTHS:
                for sub in range(1, NSUB + 1):
                    M = sample_unit(truths[res][rep], depth, rng)
                    sp.save_npz(os.path.join(d, 'depth_%dM_sub%d.npz' % (depth, sub)), M)
    with open(os.path.join(outdir, 'README.txt'), 'w') as fh:
        fh.write(README)
    print('generated Hi-C benchmark data under %s' % outdir)


if __name__ == '__main__':
    main()
