import numpy as np, pandas as pd, json, os
import ot

def compute_gold():
    data_dir = '/app/data'
    latent = np.load(os.path.join(data_dir, 'latent_pca.npy'))
    anchors = np.load(os.path.join(data_dir, 'anchors.npy'))
    meta = pd.read_csv(os.path.join(data_dir, 'cell_metadata.csv'))
    counts = np.load(os.path.join(data_dir, 'counts.npy'))
    with open(os.path.join(data_dir, 'gene_modules.json')) as f:
        modules = json.load(f)
    
    def true_fate_from_clone(clone_id):
        if 1 <= clone_id <= 60: return 0
        elif 61 <= clone_id <= 120: return 1
        else: return 2
    
    meta['true_fate'] = meta['clone_id'].apply(true_fate_from_clone)
    true_fate_df = meta[['cell_id', 'true_fate']]
    
    T_list = []
    for t in range(4):
        idx_t = meta[meta['timepoint'] == t].index.values
        idx_t1 = meta[meta['timepoint'] == t+1].index.values
        Z_t = latent[idx_t]
        Z_t1 = latent[idx_t1]
        clone_t = meta.iloc[idx_t]['clone_id'].values
        clone_t1 = meta.iloc[idx_t1]['clone_id'].values
        C = np.zeros((len(idx_t), len(idx_t1)))
        for i in range(len(idx_t)):
            for j in range(len(idx_t1)):
                dist = np.sum((Z_t[i] - Z_t1[j])**2)
                same = 1.0 if clone_t[i] == clone_t1[j] else 0.0
                C[i,j] = dist + 5.0 * (1.0 - same)
        a = np.ones(len(idx_t)) / len(idx_t)
        b = np.ones(len(idx_t1)) / len(idx_t1)
        T = ot.sinkhorn(a, b, C, reg=0.05, numItermax=10000, stopThr=1e-9)
        T_list.append(T)
    
    time_indices = [meta[meta['timepoint'] == t].index.values for t in range(5)]
    N = len(meta)
    fate_probs = np.zeros((N, 3))
    for i in range(N):
        t = meta.iloc[i]['timepoint']
        pos_t = np.where(time_indices[t] == i)[0][0]
        mass = np.zeros(len(time_indices[t]))
        mass[pos_t] = 1.0
        for s in range(t, 4):
            mass = mass @ T_list[s]
        Z_4 = latent[time_indices[4]]
        probs = np.zeros(3)
        for j in range(len(mass)):
            if mass[j] > 1e-12:
                d = np.sum((Z_4[j] - anchors)**2, axis=1)
                p = np.exp(-d / 1.0)
                p = p / p.sum()
                probs += mass[j] * p
        fate_probs[i] = probs
    
    fate_df = meta[['cell_id', 'timepoint']].copy()
    fate_df['fate_0'] = fate_probs[:, 0]
    fate_df['fate_1'] = fate_probs[:, 1]
    fate_df['fate_2'] = fate_probs[:, 2]
    
    t4_idx = time_indices[4]
    t4_fate = fate_probs[t4_idx]
    t4_max = t4_fate.argmax(axis=1)
    commitment_times = {}
    for k in range(3):
        cells_k = t4_idx[t4_max == k]
        if len(cells_k) == 0:
            commitment_times[k] = 4
            continue
        mean_probs = []
        for t in range(5):
            probs_t = fate_probs[cells_k, k]
            mean_probs.append(probs_t.mean())
        commit_t = 4
        conf = mean_probs[4]
        for t in range(5):
            if mean_probs[t] >= 0.7:
                commit_t = t
                conf = mean_probs[t]
                break
        commitment_times[k] = commit_t
    
    module_dynamics = []
    for branch in range(3):
        for t in range(5):
            idx_t = time_indices[t]
            P_t = fate_probs[idx_t, branch]
            counts_t = counts[idx_t]
            for mod_name, gene_idx in modules.items():
                gene_idx = np.array(gene_idx)
                cell_mod_expr = counts_t[:, gene_idx].mean(axis=1)
                weighted_mean = np.sum(P_t * cell_mod_expr) / (np.sum(P_t) + 1e-12)
                module_dynamics.append({'branch': branch, 'timepoint': t, 'module': mod_name, 'mean_expression': weighted_mean})
    module_df = pd.DataFrame(module_dynamics)
    
    return {
        'T': T_list,
        'fate_probs': fate_df,
        'commitment_times': commitment_times,
        'module_dynamics': module_df,
        'true_fate': true_fate_df
    }
