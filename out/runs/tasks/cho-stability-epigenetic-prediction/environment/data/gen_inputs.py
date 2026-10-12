import numpy as np
import pandas as pd
import os

def generate_data(seed=42):
    rng = np.random.default_rng(seed)
    n_train = 1000
    n_test = 400
    n_genes = 50
    marks = ["H3K4me3", "H3K27ac", "H3K9me3", "H3K27me3", "H3K36me3", "H3K4me1", "5mC"]
    features = []
    for g in range(1, n_genes+1):
        for m in marks:
            features.append(f"GENE{g:02d}_{m}")
    n_features = len(features)
    n_samples = n_train + n_test
    gene_latent = rng.normal(0, 1, size=(n_samples, n_genes))
    X = np.zeros((n_samples, n_features))
    idx = 0
    for g in range(n_genes):
        for j, m in enumerate(marks):
            noise = rng.normal(0, 0.8, size=n_samples)
            X[:, idx] = gene_latent[:, g] * 0.6 + noise
            idx += 1
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    mask = rng.random(X.shape) < 0.03
    X[mask] = np.nan
    X_train = X[:n_train]
    X_test = X[n_train:]
    n_causal = 8
    causal_idx = rng.choice(n_features, size=n_causal, replace=False)
    beta = rng.normal(0, 1.2, size=n_causal)
    score_train = X_train[:, causal_idx] @ beta + rng.normal(0, 1.0, size=n_train)
    threshold = np.percentile(score_train, 65)
    y_train = (score_train > threshold).astype(int)
    flip = rng.random(n_train) < 0.12
    y_train[flip] = 1 - y_train[flip]
    score_test = X_test[:, causal_idx] @ beta + rng.normal(0, 1.0, size=n_test)
    y_test = (score_test > threshold).astype(int)
    flip_test = rng.random(n_test) < 0.05
    y_test[flip_test] = 1 - y_test[flip_test]
    sample_ids_train = [f"TRAIN_{i:04d}" for i in range(n_train)]
    sample_ids_test = [f"TEST_{i:04d}" for i in range(n_test)]
    df_train = pd.DataFrame(X_train, columns=features)
    df_train.insert(0, "sample_id", sample_ids_train)
    df_train["batch_id"] = rng.integers(1, 6, size=n_train)
    df_train["passage_number"] = rng.integers(10, 40, size=n_train)
    df_train["stable"] = y_train
    df_test = pd.DataFrame(X_test, columns=features)
    df_test.insert(0, "sample_id", sample_ids_test)
    gene_names = [f"GENE{g:02d}" for g in range(1, n_genes+1) for _ in marks]
    mark_names = marks * n_genes
    metadata = pd.DataFrame({"feature": features, "gene": gene_names, "mark": mark_names})
    causal_features = [features[i] for i in causal_idx]
    return df_train, df_test, metadata, causal_features

if __name__ == '__main__':
    df_train, df_test, metadata, causal_features = generate_data()
    df_train.to_csv('train.csv', index=False)
    df_test.to_csv('test_features.csv', index=False)
    metadata.to_csv('feature_metadata.csv', index=False)
    with open('README.md', 'w') as f:
        f.write("""# CHO Cell Stability Epigenetic Dataset
This dataset contains synthetic epigenetic profiles for 1,000 training and 400 test CHO cell clones.
Features are named by gene and epigenetic mark (e.g., GENE01_H3K4me3).
The training set includes a binary `stable` label (1 = stable, 0 = unstable).
Metadata columns `batch_id` and `passage_number` are only present in the training set and are not available for test samples.
Approximately 3% of feature values are missing.
""")
    print("Data generated.")
