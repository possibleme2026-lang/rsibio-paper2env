import numpy as np, pandas as pd, anndata as ad, os, json

def generate(seed, out_root):
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

if __name__ == '__main__':
    generate(42, '/app/data')
