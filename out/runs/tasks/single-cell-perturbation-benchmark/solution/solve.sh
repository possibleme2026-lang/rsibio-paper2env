#!/bin/bash
set -euo pipefail
mkdir -p /app/solution /app/output
cat > /app/solution/evaluate.py << 'PY'
import argparse, json, os, numpy as np
import anndata as ad
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

def bh_fdr(pvals):
    pvals = np.asarray(pvals)
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    ranked = pvals[order]
    qvals = ranked * n / (np.arange(n) + 1)
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    qvals = np.clip(qvals, 0, 1)
    out = np.empty(n)
    out[order] = qvals
    return out

def de_genes(pert_cells, ctrl_cells, gene_names):
    t, p = stats.ttest_ind(pert_cells, ctrl_cells, axis=0, equal_var=False)
    mean_pert = np.mean(np.expm1(pert_cells), axis=0)
    mean_ctrl = np.mean(np.expm1(ctrl_cells), axis=0)
    log2fc = np.log2((mean_pert + 1) / (mean_ctrl + 1))
    q = bh_fdr(p)
    de_mask = (q < 0.05) & (np.abs(log2fc) > 0.5)
    return set(np.array(gene_names)[de_mask])

def energy_distance(X, Y):
    d_xy = pairwise_distances(X, Y).mean()
    d_xx = pairwise_distances(X, X).mean()
    d_yy = pairwise_distances(Y, Y).mean()
    return max(0.0, 2 * d_xy - d_xx - d_yy)

def compute_task_metrics(task_dir):
    meta = json.load(open(os.path.join(task_dir, 'perturbations.json')))
    perturbations = meta['perturbations']
    gene_names = meta['genes']
    ctrl = ad.read_h5ad(os.path.join(task_dir, 'control.h5ad'))
    ctrl_cells = ctrl.X
    methods = ['mean', 'linear', 'mlp', 'perturbnet']
    n_comp = min(20, ctrl_cells.shape[1], ctrl_cells.shape[0])
    pca = PCA(n_components=n_comp, random_state=0)
    pca.fit(ctrl_cells)
    metrics = {m: {'mse': [], 'delta_pearson': [], 'de_f1': [], 'dist_sim': []} for m in methods}
    for pert in perturbations:
        true = ad.read_h5ad(os.path.join(task_dir, 'true', pert + '.h5ad'))
        true_cells = true.X
        mu_true = true_cells.mean(axis=0)
        mu_ctrl = ctrl_cells.mean(axis=0)
        true_de = de_genes(true_cells, ctrl_cells, gene_names)
        true_proj = pca.transform(true_cells)
        for m in methods:
            pred = ad.read_h5ad(os.path.join(task_dir, 'methods', m, 'pred_' + pert + '.h5ad'))
            pred_cells = pred.X
            mu_pred = pred_cells.mean(axis=0)
            mse = np.mean((mu_pred - mu_true) ** 2)
            delta_true = mu_true - mu_ctrl
            delta_pred = mu_pred - mu_ctrl
            if np.std(delta_true) == 0 or np.std(delta_pred) == 0:
                pear = 0.0
            else:
                pear = np.corrcoef(delta_true, delta_pred)[0, 1]
            pred_de = de_genes(pred_cells, ctrl_cells, gene_names)
            if len(true_de) == 0:
                f1 = 1.0 if len(pred_de) == 0 else 0.0
            else:
                tp = len(true_de & pred_de)
                prec = tp / len(pred_de) if len(pred_de) > 0 else 0.0
                rec = tp / len(true_de)
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            pred_proj = pca.transform(pred_cells)
            ed = energy_distance(true_proj, pred_proj)
            dist_sim = 1.0 / (1.0 + ed)
            metrics[m]['mse'].append(mse)
            metrics[m]['delta_pearson'].append(pear)
            metrics[m]['de_f1'].append(f1)
            metrics[m]['dist_sim'].append(dist_sim)
    agg = {m: {k: float(np.mean(v)) for k, v in metrics[m].items()} for m in methods}
    return agg, methods

def compute_ranking_and_overall(agg, methods):
    ranking = {}
    ranking['mse'] = sorted(methods, key=lambda m: (agg[m]['mse'], m))
    for metric in ['delta_pearson', 'de_f1', 'dist_sim']:
        ranking[metric] = sorted(methods, key=lambda m: (-agg[m][metric], m))
    overall = {}
    for metric in ['mse', 'delta_pearson', 'de_f1', 'dist_sim']:
        vals = np.array([agg[m][metric] for m in methods])
        if metric == 'mse':
            norm = 1.0 - (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
        else:
            norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
        for i, m in enumerate(methods):
            overall[m] = overall.get(m, 0.0) + norm[i]
    for m in methods:
        overall[m] /= 4.0
    best = sorted(methods, key=lambda m: (-overall[m], m))[0]
    return ranking, overall, best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = {}
    for task in ['task1', 'task2']:
        task_dir = os.path.join(args.input, task)
        agg, methods = compute_task_metrics(task_dir)
        ranking, overall, best = compute_ranking_and_overall(agg, methods)
        result[task] = {
            'methods': agg,
            'ranking': ranking,
            'overall_score': overall,
            'best_method': best
        }
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print('Wrote', args.output)

if __name__ == '__main__':
    main()
PY
chmod +x /app/solution/evaluate.py
