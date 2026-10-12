#!/usr/bin/env python3
"""Protected ground-truth generator for the DBayesCM-vMF task.

Reproduces the exact cohort that was baked into the image and emits the
hidden truth: community labels, transitional-sample flags, discriminative
feature annotation and taxon-host association pairs.
"""
import argparse
from pathlib import Path

import numpy as np

N = 240
K = 5
D = 8
CLUSTER_SIZES = [56, 52, 48, 44, 40]
NBOUND_PER_CLUSTER = [5, 5, 5, 5, 4]
N_TAXA = 55
N_METAB = 45
N_TX = 35
N_HOST = N_METAB + N_TX
KAPPA = 25.0
MIN_ANGLE_DEG = 45.0
SIGMA_R = 0.55
NOISE_M = 0.45
NOISE_H = 0.55
N_COOC = 70
TAXA_DIMS = [0, 1, 2, 3]
HOST_DIMS = [4, 5, 6, 7]
DEFAULT_SEED = 20260728


def make_directions(rng, k, d, min_angle_deg):
    mus = []
    guard = 0
    while len(mus) < k:
        guard += 1
        if guard > 200000:
            raise RuntimeError('could not place cluster directions')
        v = rng.normal(size=d)
        v /= np.linalg.norm(v)
        ok = True
        for m in mus:
            ang = np.degrees(np.arccos(np.clip(float(v @ m), -1.0, 1.0)))
            if ang < min_angle_deg:
                ok = False
                break
        if ok:
            mus.append(v)
    return np.array(mus)


def _vmf_w1(rng, d, kappa, n):
    b = (-2.0 * kappa + np.sqrt(4.0 * kappa ** 2 + (d - 1.0) ** 2)) / (d - 1.0)
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (d - 1.0) * np.log(1.0 - x0 ** 2)
    a = (d - 1.0) / 2.0
    out = np.empty(n)
    filled = 0
    while filled < n:
        m = 4 * (n - filled)
        z = rng.beta(a, a, size=m)
        u = rng.random(m)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        log_acc = kappa * w + (d - 1.0) * np.log(1.0 - x0 * w) - c
        take = w[log_acc >= np.log(np.maximum(u, 1e-300))]
        for value in take[: n - filled]:
            out[filled] = value
            filled += 1
            if filled == n:
                break
    return out


def sample_vmf(rng, mean, kappa, n, d):
    if n <= 0:
        return np.zeros((0, d))
    frame = np.eye(d)
    frame[:, 0] = mean
    q, _ = np.linalg.qr(frame)
    if float(q[:, 0] @ mean) < 0.0:
        q[:, 0] *= -1.0
    w1 = _vmf_w1(rng, d, kappa, n)
    tang = rng.normal(size=(n, d - 1))
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    s = np.sqrt(np.maximum(1.0 - w1 ** 2, 0.0))
    w = np.empty((n, d))
    w[:, 0] = w1
    w[:, 1:] = tang * s[:, None]
    return w @ q.T


def make_loadings(rng, n_feat, dims, frac_signal=0.30):
    w = np.zeros((n_feat, D))
    signal = rng.random(n_feat) < frac_signal
    for j in range(n_feat):
        n_nz = int(rng.integers(1, 4))
        pick = rng.choice(np.asarray(dims, dtype=int), size=n_nz, replace=False)
        direction = rng.normal(size=n_nz)
        direction /= np.linalg.norm(direction)
        amp = rng.uniform(0.8, 1.6) if signal[j] else rng.uniform(0.0, 0.20)
        w[j, pick] = amp * direction
    return w, signal


def simulate(seed=DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    mus = make_directions(rng, K, D, MIN_ANGLE_DEG)

    labels = np.repeat(np.arange(K), CLUSTER_SIZES)
    n = labels.size

    boundary = np.zeros(n, dtype=bool)
    picked = []
    for c in range(K):
        idx = np.flatnonzero(labels == c)
        sel = rng.choice(idx, size=NBOUND_PER_CLUSTER[c], replace=False)
        picked.extend(int(x) for x in sel)
    boundary[np.array(sorted(picked))] = True

    u = np.zeros((n, D))
    for i in range(n):
        c = int(labels[i])
        if boundary[i]:
            other = int(rng.integers(0, K - 1))
            if other >= c:
                other += 1
            base = mus[c] + mus[other]
            base = base / np.linalg.norm(base)
        else:
            base = mus[c]
        u[i] = sample_vmf(rng, base, KAPPA, 1, D)[0]
    radii = np.exp(rng.normal(0.0, SIGMA_R, size=n))
    f = u * radii[:, None]

    w_m, sig_m = make_loadings(rng, N_TAXA, TAXA_DIMS)
    w_metab, sig_metab = make_loadings(rng, N_METAB, HOST_DIMS)
    w_tx, sig_tx = make_loadings(rng, N_TX, HOST_DIMS)
    w_h = np.vstack([w_metab, w_tx])
    feature_signal = np.concatenate([sig_m, sig_metab, sig_tx])

    pairs = []
    seen = set()
    while len(pairs) < N_COOC:
        j = int(rng.integers(0, N_TAXA))
        k = int(rng.integers(0, N_HOST))
        if (j, k) in seen:
            continue
        seen.add((j, k))
        pairs.append((j, k))
    betas = rng.uniform(0.9, 1.5, size=N_COOC)

    log_abund = (rng.normal(0.0, 1.5, size=N_TAXA)[None, :]
                 + f @ w_m.T
                 + rng.normal(0.0, NOISE_M, size=(n, N_TAXA)))
    host = np.hstack([
        f @ w_metab.T + rng.normal(0.0, NOISE_H, size=(n, N_METAB)),
        f @ w_tx.T + rng.normal(0.0, NOISE_H, size=(n, N_TX)),
    ])
    for p, (j, k) in enumerate(pairs):
        effect = betas[p] * rng.normal(0.0, 1.0, size=n)
        log_abund[:, j] += effect
        host[:, k] += effect

    abund = np.exp(log_abund - log_abund.max(axis=1, keepdims=True))
    abund /= abund.sum(axis=1, keepdims=True)
    abund = np.maximum(abund, 1e-5)
    abund /= abund.sum(axis=1, keepdims=True)

    metab = 5.0 + host[:, :N_METAB]
    tx = 6.0 + host[:, N_METAB:]

    return dict(mus=mus, labels=labels, boundary=boundary, w_m=w_m, w_h=w_h,
                feature_signal=feature_signal, pairs=pairs, betas=betas,
                abund=abund, metab=metab, tx=tx)


def build_truth(seed=DEFAULT_SEED):
    sim = simulate(seed)
    n = sim['labels'].size
    er = np.exp(SIGMA_R ** 2 / 2.0)
    pi = np.array(CLUSTER_SIZES, dtype=float) / n
    mu_mean = pi @ sim['mus']
    sigma_u = np.zeros((D, D))
    for c in range(K):
        m = sim['mus'][c]
        sigma_u += pi[c] * (np.outer(m, m) + (1.0 / KAPPA) * (np.eye(D) - np.outer(m, m)))
    sigma_f = (er ** 2) * (sigma_u - np.outer(mu_mean, mu_mean))
    w_m, w_h = sim['w_m'], sim['w_h']
    cov = w_m @ sigma_f @ w_h.T
    var_m = np.einsum('ij,jk,ik->i', w_m, sigma_f, w_m) + NOISE_M ** 2
    var_h = np.einsum('ij,jk,ik->i', w_h, sigma_f, w_h) + NOISE_H ** 2
    for p, (j, k) in enumerate(sim['pairs']):
        b2 = float(sim['betas'][p]) ** 2
        cov[j, k] += b2
        var_m[j] += b2
        var_h[k] += b2
    corr = cov / np.sqrt(np.outer(var_m, var_h))
    cooc_mask = np.abs(corr) >= 0.45
    return sim, cooc_mask


def write_data(sim, seed, outdir):
    import pandas as pd
    n = sim['labels'].size
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    sample_ids = ['S%03d' % (i + 1) for i in range(n)]
    taxa = ['taxon_%02d' % (j + 1) for j in range(N_TAXA)]
    metab = ['metab_%02d' % (j + 1) for j in range(N_METAB)]
    tx = ['tx_%02d' % (j + 1) for j in range(N_TX)]
    pd.DataFrame(sim['abund'], index=sample_ids, columns=taxa).to_csv(
        out / 'microbiome_abundance.csv', float_format='%.6g')
    pd.DataFrame(sim['metab'], index=sample_ids, columns=metab).to_csv(
        out / 'host_metabolomics.csv', float_format='%.5g')
    pd.DataFrame(sim['tx'], index=sample_ids, columns=tx).to_csv(
        out / 'host_transcriptomics.csv', float_format='%.5g')
    mg = np.random.default_rng(seed + 1)
    meta = pd.DataFrame({'sample_id': sample_ids,
                         'cohort': mg.choice(['C1', 'C2'], size=n),
                         'sex': mg.choice(['F', 'M'], size=n),
                         'age': np.round(mg.normal(58.0, 11.0, size=n), 1),
                         'bmi': np.round(mg.normal(27.0, 4.5, size=n), 1)})
    meta.to_csv(out / 'sample_metadata.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--out', required=True)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    sim, cooc_mask = build_truth(args.seed)
    feat_names = (['taxon_%02d' % (j + 1) for j in range(N_TAXA)]
                  + ['metab_%02d' % (j + 1) for j in range(N_METAB)]
                  + ['tx_%02d' % (j + 1) for j in range(N_TX)])
    np.savez(args.out, labels=sim['labels'], boundary=sim['boundary'],
             feature_signal=sim['feature_signal'], cooc_mask=cooc_mask,
             k_true=K, feature_names=np.array(feat_names),
             taxa_names=np.array(feat_names[:N_TAXA]),
             host_names=np.array(feat_names[N_TAXA:]))
    if args.outdir:
        write_data(sim, args.seed, args.outdir)
    print('gold written to', args.out, 'positives:', int(cooc_mask.sum()),
          'signal features:', int(sim['feature_signal'].sum()))


if __name__ == '__main__':
    main()
