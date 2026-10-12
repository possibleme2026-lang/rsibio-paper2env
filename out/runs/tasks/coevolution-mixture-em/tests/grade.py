#!/usr/bin/env python3
import json, os, sys, math
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
    pi_ref, h_ref, J_ref, r_ref, elbo_hist_ref = em_reference(pairs_obs, C=2, max_iter=100, tol=1e-6)
    elbo_ref = elbo_hist_ref[-1]
    constraints = []
    try:
        with open('/app/results/params.json') as f:
            params = json.load(f)
        with open('/app/results/responsibilities.tsv') as f:
            resp_lines = [l.strip().split(chr(9)) for l in f if l.strip()]
        with open('/app/results/elbo_history.tsv') as f:
            elbo_lines = [l.strip().split(chr(9)) for l in f if l.strip()]
        constraints.append(('files_exist_and_parse', True))
    except Exception:
        constraints.append(('files_exist_and_parse', False))
        score = sum(1 for _, ok in constraints if ok) / 11.0
        with open('/logs/verifier/reward.json', 'w') as f:
            json.dump({'score': score, 'reward': score}, f)
        with open('/logs/verifier/report.json', 'w') as f:
            json.dump({'constraints': constraints}, f)
        print(f'Score: {score}')
        for name, ok in constraints:
            status = 'PASS' if ok else 'FAIL'
            print(f'{name}: {status}')
        return
    try:
        pi = np.array(params['pi'])
        h = np.array(params['h'])
        J = np.array(params['J'])
        elbo = float(params['elbo'])
        iterations = int(params['iterations'])
        assert pi.shape == (2,)
        assert h.shape == (2,4)
        assert J.shape == (2,4,4)
        constraints.append(('params_shapes', True))
    except Exception:
        constraints.append(('params_shapes', False))
        score = sum(1 for _, ok in constraints if ok) / 11.0
        with open('/logs/verifier/reward.json', 'w') as f:
            json.dump({'score': score, 'reward': score}, f)
        with open('/logs/verifier/report.json', 'w') as f:
            json.dump({'constraints': constraints}, f)
        print(f'Score: {score}')
        for name, ok in constraints:
            status = 'PASS' if ok else 'FAIL'
            print(f'{name}: {status}')
        return
    try:
        assert abs(pi.sum() - 1) < 1e-6
        assert np.all(pi >= 0.01)
        constraints.append(('pi_valid', True))
    except Exception:
        constraints.append(('pi_valid', False))
    try:
        assert np.all(np.abs(h.sum(axis=1)) < 1e-6)
        constraints.append(('h_sum_zero', True))
    except Exception:
        constraints.append(('h_sum_zero', False))
    try:
        for c in range(2):
            assert np.allclose(J[c], J[c].T, atol=1e-6)
            assert np.allclose(np.diag(J[c]), 0, atol=1e-6)
        constraints.append(('J_symmetric_zero_diag', True))
    except Exception:
        constraints.append(('J_symmetric_zero_diag', False))
    try:
        elbo_vals = [float(l[1]) for l in elbo_lines[1:]]
        for i in range(1, len(elbo_vals)):
            assert elbo_vals[i] >= elbo_vals[i-1] - 1e-6
        constraints.append(('elbo_history_monotonic', True))
    except Exception:
        constraints.append(('elbo_history_monotonic', False))
    try:
        elbo_vals = [float(l[1]) for l in elbo_lines[1:]]
        assert abs(elbo - elbo_vals[-1]) < 1e-6
        constraints.append(('elbo_self_consistent', True))
    except Exception:
        constraints.append(('elbo_self_consistent', False))
    try:
        best_rmse_h = np.inf
        best_rmse_J = np.inf
        best_perm = None
        for perm in [(0,1), (1,0)]:
            h_ref_perm = h_ref[list(perm)]
            J_ref_perm = J_ref[list(perm)]
            rmse_h = np.sqrt(np.mean((h - h_ref_perm)**2))
            rmse_J = np.sqrt(np.mean((J - J_ref_perm)**2))
            if rmse_h + rmse_J < best_rmse_h + best_rmse_J:
                best_rmse_h = rmse_h
                best_rmse_J = rmse_J
                best_perm = perm
        assert best_rmse_h < 0.1 and best_rmse_J < 0.1
        constraints.append(('parameter_recovery', True))
    except Exception:
        constraints.append(('parameter_recovery', False))
        best_perm = None
    try:
        header = resp_lines[0]
        assert header == ['pair_id', 'r1', 'r2']
        resp_data = []
        for line in resp_lines[1:]:
            assert len(line) == 3
            resp_data.append((int(line[0]), float(line[1]), float(line[2])))
        resp_data.sort(key=lambda x: x[0])
        r_agent = np.array([[r1, r2] for _, r1, r2 in resp_data])
        assert r_agent.shape[0] == len(pairs_obs)
        assert np.all(r_agent >= 0) and np.all(r_agent <= 1)
        assert np.allclose(r_agent.sum(axis=1), 1, atol=1e-6)
        if best_perm is not None:
            r_ref_perm = r_ref[:, list(best_perm)]
            mae = np.mean(np.abs(r_agent - r_ref_perm))
            assert mae < 0.05
        constraints.append(('responsibilities_mae', True))
    except Exception:
        constraints.append(('responsibilities_mae', False))
    try:
        if best_perm is not None:
            assign_agent = np.argmax(r_agent, axis=1)
            assign_ref = np.argmax(r_ref[:, list(best_perm)], axis=1)
            acc = np.mean(assign_agent == assign_ref)
            assert acc >= 0.9
        constraints.append(('assignment_accuracy', True))
    except Exception:
        constraints.append(('assignment_accuracy', False))
    try:
        rel_err = abs(elbo - elbo_ref) / abs(elbo_ref)
        assert rel_err < 1e-3
        constraints.append(('elbo_vs_reference', True))
    except Exception:
        constraints.append(('elbo_vs_reference', False))
    total = 11
    passed = sum(1 for _, ok in constraints if ok)
    score = passed / total
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({'score': score, 'reward': score}, f)
    with open('/logs/verifier/report.json', 'w') as f:
        json.dump({'constraints': constraints, 'score': score}, f)
    print(f'Score: {score}')
    for name, ok in constraints:
        status = 'PASS' if ok else 'FAIL'
        print(f'{name}: {status}')

if __name__ == '__main__':
    main()
