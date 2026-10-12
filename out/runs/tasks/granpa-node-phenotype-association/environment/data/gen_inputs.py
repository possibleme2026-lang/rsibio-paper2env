#!/usr/bin/env python3
'''Deterministic input generator for the GRANPA node-phenotype task.

Writes graph.gfa and phenotypes.tsv into the directory given as argv[1]
(default: the directory containing this file).  Everything derives from SEED,
so repeated runs are byte identical.
'''

import os
import random
import sys

SEED = 20260731
N_LOCI = 118
N_CASES = 10
N_CONTROLS = 14
N_SAMPLES = N_CASES + N_CONTROLS

# hidden causal design: locus indices x (#case carriers, #control carriers)
CAUSAL = [
    ([45, 46, 47], 9, 1),
    ([83, 84], 8, 1),
]


def dna(rng, n):
    s = [rng.choice('ACGT')]
    for _ in range(n - 1):
        if rng.random() < 0.62:
            if s[-1] in 'GC':
                s.append(rng.choice('GC'))
            else:
                s.append(rng.choice('AT'))
        else:
            s.append(rng.choice('ACGT'))
    return ''.join(s)


def build(outdir):
    rng = random.Random(SEED)
    samples = ['IND%02d' % i for i in range(1, N_SAMPLES + 1)]
    cases = samples[:N_CASES]
    controls = samples[N_CASES:]

    n_alleles = []
    for _ in range(N_LOCI):
        if rng.random() < 0.45:
            n_alleles.append(3 if rng.random() < 0.25 else 2)
        else:
            n_alleles.append(1)
    for idxs, _, _ in CAUSAL:
        for i in idxs:
            n_alleles[i] = 2

    freq = []
    for i in range(N_LOCI):
        k = n_alleles[i]
        if k == 1:
            freq.append([1.0])
        elif k == 2:
            f = rng.uniform(0.15, 0.85)
            freq.append([f, 1.0 - f])
        else:
            f = rng.uniform(0.40, 0.70)
            g = rng.uniform(0.10, max(0.11, 1.0 - f - 0.05))
            freq.append([f, g, 1.0 - f - g])

    allele = {}
    for s in samples:
        row = []
        for i in range(N_LOCI):
            k = n_alleles[i]
            if k == 1:
                row.append(0)
                continue
            r = rng.random()
            acc = 0.0
            pick = k - 1
            for j in range(k):
                acc += freq[i][j]
                if r <= acc:
                    pick = j
                    break
            row.append(pick)
        allele[s] = row

    for idxs, n_case_car, n_ctrl_car in CAUSAL:
        carriers = set(cases[:n_case_car]) | set(controls[:n_ctrl_car])
        for i in idxs:
            for s in samples:
                allele[s][i] = 1 if s in carriers else 0

    segs = []
    seg_of = {}
    counter = 1
    for i in range(N_LOCI):
        for j in range(n_alleles[i]):
            name = 's%04d' % counter
            counter += 1
            seg_of[(i, j)] = name
            if n_alleles[i] == 1:
                ln = rng.randint(400, 2500)
            else:
                ln = rng.randint(80, 1800)
            segs.append((name, dna(rng, ln)))

    paths = {}
    for s in samples:
        toks = []
        for i in range(N_LOCI):
            j = allele[s][i]
            orient = '+'
            if n_alleles[i] == 1 and rng.random() < 0.08:
                orient = '-'
            toks.append(seg_of[(i, j)] + orient)
        paths[s] = toks

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'graph.gfa'), 'w') as fh:
        fh.write('H\tVN:Z:1.0\n')
        for name, seq in segs:
            fh.write('S\t%s\t%s\n' % (name, seq))
        for i in range(N_LOCI - 1):
            for a in range(n_alleles[i]):
                for b in range(n_alleles[i + 1]):
                    fh.write('L\t%s\t+\t%s\t+\t0M\n' % (seg_of[(i, a)], seg_of[(i + 1, b)]))
        for s in samples:
            fh.write('P\t%s\t%s\t*\n' % (s, ','.join(paths[s])))

    with open(os.path.join(outdir, 'phenotypes.tsv'), 'w') as fh:
        fh.write('sample_id\tphenotype\n')
        for s in cases:
            fh.write('%s\tcase\n' % s)
        for s in controls:
            fh.write('%s\tcontrol\n' % s)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    build(out)
    print('wrote inputs to %s' % out)
