#!/bin/bash
set -euo pipefail
mkdir -p /app/results
cat > /app/solve.py << 'PYEOF'
import json, os
import numpy as np
from scipy.optimize import minimize

def unpack_params(params):
    u = params[:3]
    h = np.zeros(4)
    h[:3] = u
    h[3] = -u.sum()
    v = params[3:]
    J = np.zeros((4,4))
    J[0,1] = J[1,0] = v[0]
    J[0,2] = J[2,0] = v[1]
    J[0,3] = J[3,0] = v[2]
    J[1,2] = J[2,1] = v[3]
    J[1,3] = J[3,1] = v[4]
    J[2,3] = J[3,2] = v[5]
    return h, J

def pack_params(h, J):
    u = h[:3]
    v = np.array([J[0,1], J[0,2], J[0,3], J[1,2], J[1,3], J[2,3]])
    return np.concatenate([u, v])

def potts_prob(h, J):
    logits = h[:,None] + h[None,:] + J
    max_logit = logits.max()
    exp_logits = np.exp(logits - max_logit)
    Z = exp_logits.sum()
    P = exp_logits / Z
    logZ = max_logit + np.log(Z)
    return P, logZ

def neg_ll_and_grad(params, obs_weights):
    h, J = unpack_params(params)
    P, logZ = potts_prob(h, J)
    marg = P.sum(axis=1)
    grad_h = np.zeros(4)
    grad_J_full = np.zeros((4,4))
    total_loglike = 0.0
    for a, b, w in obs_weights:
        logp = h[a] + h[b] + J[a,b] - logZ
        total_loglike += w * logp
        for k in range(4):
            grad_h[k] += w * ((a==k) + (b==k) - 2*marg[k])
        if a != b:
            u, v = (a,b) if a < b else (b,a)
            grad_J_full[u,v] += w * (1 - 2*P[u,v])
    grad_u = grad_h[:3] - grad_h[3]
    grad_v = np.array([grad_J_full[0,1], grad_J_full[0,2], grad_J_full[0,3],
                       grad_J_full[1,2], grad_J_full[1,3], grad_J_full[2,3]])
    grad = np.concatenate([grad_u, grad_v])
    return -total_loglike, -grad

def em_reference(pairs_obs, C=2, max_iter=100, tol=1e-6):
    M = len(pairs_obs)
    pi = np.ones(C) / C
    h = [np.zeros(4) for _ in range(C)]
    J = [np.zeros((4,4)) for _ in range(C)]
    elbo_history = []
    for it in range(max_iter):
        loglik = np.zeros((M, C))
        for c in range(C):
            h_c, J_c = h[c], J[c]
            P_c, logZ_c = potts_prob(h_c, J_c)
            for p, obs in enumerate(pairs_obs):
                ll = 0.0
                for a,b in obs:
                    ll += h_c[a] + h_c[b] + J_c[a,b] - logZ_c
                loglik[p,c] = ll
        log_pi = np.log(pi)
        log_joint = loglik + log_pi[None,:]
        max_log = log_joint.max(axis=1, keepdims=True)
        log_sum = max_log + np.log(np.exp(log_joint - max_log).sum(axis=1, keepdims=True))
        r = np.exp(log_joint - log_sum)
        pi = r.mean(axis=0)
        for c in range(C):
            obs_weights = []
            for p, obs in enumerate(pairs_obs):
                w = r[p,c]
                if w > 1e-12:
                    for a,b in obs:
                        obs_weights.append((a,b,w))
            x0 = pack_params(h[c], J[c])
            res = minimize(neg_ll_and_grad, x0, args=(obs_weights,), method='L-BFGS-B', jac=True, tol=1e-8, options={'maxiter': 1000})
            h[c], J[c] = unpack_params(res.x)
        elbo = 0.0
        for p in range(M):
            for c in range(C):
                if r[p,c] > 1e-12:
                    elbo += r[p,c] * (np.log(pi[c]) + loglik[p,c] - np.log(r[p,c]))
        elbo_history.append(elbo)
        if it > 0 and abs(elbo_history[-1] - elbo_history[-2]) < tol:
            break
    return pi, np.array(h), np.array(J), r, elbo_history

def load_data():
    data_dir = '/app/data'
    aa_to_group = {}
    with open(os.path.join(data_dir, 'alphabet_map.tsv')) as f:
        next(f)
        for line in f:
            aa, g = line.strip().split(chr(9))
            aa_to_group[aa] = int(g)
    pairs = []
    with open(os.path.join(data_dir, 'contacts.tsv')) as f:
        next(f)
        for line in f:
            parts = line.strip().split(chr(9))
            pair_id = int(parts[0])
            site_i = int(parts[1]) - 1
            site_j = int(parts[2]) - 1
            pairs.append((pair_id, site_i, site_j))
    seqs = []
    with open(os.path.join(data_dir, 'alignment.fasta')) as f:
        header = None
        seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header is not None:
                    seqs.append(seq)
                header = line
                seq = ''
            else:
                seq += line
        if header is not None:
            seqs.append(seq)
    pairs_obs = []
    for pair_id, i, j in pairs:
        obs = []
        for seq in seqs:
            aa_i = seq[i]
            aa_j = seq[j]
            if aa_i == 'X' or aa_j == 'X':
                continue
            g_i = aa_to_group[aa_i] - 1
            g_j = aa_to_group[aa_j] - 1
            obs.append((g_i, g_j))
        pairs_obs.append(obs)
    return pairs_obs

def main():
    pairs_obs = load_data()
    pi, h, J, r, elbo_hist = em_reference(pairs_obs, C=2, max_iter=100, tol=1e-6)
    os.makedirs('/app/results', exist_ok=True)
    with open('/app/results/params.json', 'w') as f:
        json.dump({
            'pi': pi.tolist(),
            'h': h.tolist(),
            'J': J.tolist(),
            'elbo': float(elbo_hist[-1]),
            'iterations': len(elbo_hist)
        }, f)
    with open('/app/results/responsibilities.tsv', 'w') as f:
        f.write('pair_id' + chr(9) + 'r1' + chr(9) + 'r2' + chr(10))
        for i, (r1, r2) in enumerate(r):
            f.write(f'{i+1}' + chr(9) + f'{r1}' + chr(9) + f'{r2}' + chr(10))
    with open('/app/results/elbo_history.tsv', 'w') as f:
        f.write('iteration' + chr(9) + 'elbo' + chr(10))
        for i, e in enumerate(elbo_hist):
            f.write(f'{i+1}' + chr(9) + f'{e}' + chr(10))

if __name__ == '__main__':
    main()
PYEOF
python /app/solve.py
