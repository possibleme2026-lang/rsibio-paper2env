import numpy as np, pandas as pd, anndata as ad, os, json
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

def generate_data(seed, out_root):
    rng = np.random.default_rng(seed)
    n_genes = 300
    n_ctrl = 500
    n_cells = 200
    genes = [f'GENE_{i:03d}' for i in range(n_genes)]
    mu0 = rng.uniform(0.5, 4.0, n_genes)
    loadings = rng.normal(0, 0.3, (5, n_genes))
    def make_cells(n, mu, latent_scale=1.0, noise=0.2):
        latent = rng.normal(0, latent_scale, (n, 5))
        X = mu + latent @ loadings + rng.normal(0, noise, (n, n_genes))
        return np.clip(X, 0, 10).astype(np.float32)
    ctrl = make_cells(n_ctrl, mu0)
    def save(expr, path):
        adata = ad.AnnData(X=expr, obs=pd.DataFrame(index=[f'cell_{i}' for i in range(expr.shape[0])]), var=pd.DataFrame(index=genes))
        adata.write_h5ad(path)
    for task in ['task1', 'task2']:
        task_dir = os.path.join(out_root, task)
        os.makedirs(os.path.join(task_dir, 'true'), exist_ok=True)
        for m in ['mean', 'linear', 'mlp', 'perturbnet']:
            os.makedirs(os.path.join(task_dir, 'methods', m), exist_ok=True)
        save(ctrl, os.path.join(task_dir, 'control.h5ad'))
        if task == 'task1':
            perts = [f'P{i}' for i in range(1, 7)]
            targets = rng.choice(n_genes, size=6, replace=False)
        else:
            perts = [f'C{i}' for i in range(1, 6)]
            targets = [rng.choice(n_genes, size=2, replace=False) for _ in range(5)]
        meta = {'perturbations': perts, 'genes': genes}
        json.dump(meta, open(os.path.join(task_dir, 'perturbations.json'), 'w'))
        for pi, pert in enumerate(perts):
            if task == 'task1':
                delta = rng.normal(0, 0.1, n_genes)
                tg = targets[pi]
                eff_genes = rng.choice(n_genes, size=15, replace=False)
                eff_genes = np.union1d(eff_genes, [tg])
                delta[eff_genes] = rng.normal(1.5, 0.5, len(eff_genes))
                mu_true = mu0 + delta
                true_cells = make_cells(n_cells, mu_true)
                save(true_cells, os.path.join(task_dir, 'true', f'{pert}.h5ad'))
                for m in ['mean', 'linear', 'mlp', 'perturbnet']:
                    if m == 'mean':
                        mu_pred = mu0
                    elif m == 'linear':
                        mu_pred = mu0 + 0.5 * delta + rng.normal(0, 0.2, n_genes)
                    elif m == 'mlp':
                        mu_pred = mu0 + delta + rng.normal(0, 0.15, n_genes)
                    else:
                        mu_pred = mu0 + delta + rng.normal(0, 0.05, n_genes)
                    scale = 0.7 if m == 'mlp' else (0.9 if m == 'perturbnet' else 1.0)
                    noise = 0.15 if m == 'perturbnet' else 0.2
                    pred_cells = make_cells(n_cells, mu_pred, latent_scale=scale, noise=noise)
                    save(pred_cells, os.path.join(task_dir, 'methods', m, f'pred_{pert}.h5ad'))
            else:
                delta = rng.normal(0, 0.1, n_genes)
                for tg in targets[pi]:
                    eff_genes = rng.choice(n_genes, size=15, replace=False)
                    eff_genes = np.union1d(eff_genes, [tg])
                    delta[eff_genes] += rng.normal(1.5, 0.5, len(eff_genes))
                delta = 0.8 * delta
                mu_true = mu0 + delta
                true_cells = make_cells(n_cells, mu_true)
                save(true_cells, os.path.join(task_dir, 'true', f'{pert}.h5ad'))
                for m in ['mean', 'linear', 'mlp', 'perturbnet']:
                    if m == 'mean':
                        mu_pred = mu0
                    elif m == 'linear':
                        mu_pred = mu0 + 0.5 * delta + rng.normal(0, 0.2, n_genes)
                    elif m == 'mlp':
                        mu_pred = mu0 + 0.8 * delta + rng.normal(0, 0.15, n_genes)
                    else:
                        mu_pred = mu0 + 0.95 * delta + rng.normal(0, 0.05, n_genes)
                    scale = 0.7 if m == 'mlp' else (0.9 if m == 'perturbnet' else 1.0)
                    noise = 0.15 if m == 'perturbnet' else 0.2
                    pred_cells = make_cells(n_cells, mu_pred, latent_scale=scale, noise=noise)
                    save(pred_cells, os.path.join(task_dir, 'methods', m, f'pred_{pert}.h5ad'))

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

def compute_gold(data_root):
    result = {}
    for task in ['task1', 'task2']:
        task_dir = os.path.join(data_root, task)
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
            true = ad.read_h5ad(os.path.join(task_dir, 'true', f'{pert}.h5ad'))
            true_cells = true.X
            mu_true = true_cells.mean(axis=0)
            mu_ctrl = ctrl_cells.mean(axis=0)
            true_de = de_genes(true_cells, ctrl_cells, gene_names)
            true_proj = pca.transform(true_cells)
            for m in methods:
                pred = ad.read_h5ad(os.path.join(task_dir, 'methods', m, f'pred_{pert}.h5ad'))
                pred_cells = pred.X
                mu_pred = pred_cells.mean(axis=0)
                mse = float(np.mean((mu_pred - mu_true) ** 2))
                delta_true = mu_true - mu_ctrl
                delta_pred = mu_pred - mu_ctrl
                if np.std(delta_true) == 0 or np.std(delta_pred) == 0:
                    pear = 0.0
                else:
                    pear = float(np.corrcoef(delta_true, delta_pred)[0, 1])
                pred_de = de_genes(pred_cells, ctrl_cells, gene_names)
                if len(true_de) == 0:
                    f1 = 1.0 if len(pred_de) == 0 else 0.0
                else:
                    tp = len(true_de & pred_de)
                    prec = tp / len(pred_de) if len(pred_de) > 0 else 0.0
                    rec = tp / len(true_de)
                    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                pred_proj = pca.transform(pred_cells)
                ed = energy_distance(true_proj, pred_proj)
                dist_sim = float(1.0 / (1.0 + ed))
                metrics[m]['mse'].append(mse)
                metrics[m]['delta_pearson'].append(pear)
                metrics[m]['de_f1'].append(f1)
                metrics[m]['dist_sim'].append(dist_sim)
        agg = {m: {k: float(np.mean(v)) for k, v in metrics[m].items()} for m in methods}
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
            overall[m] = float(overall[m] / 4.0)
        best = sorted(methods, key=lambda m: (-overall[m], m))[0]
        result[task] = {
            'methods': agg,
            'ranking': ranking,
            'overall_score': overall,
            'best_method': best
        }
    return result
