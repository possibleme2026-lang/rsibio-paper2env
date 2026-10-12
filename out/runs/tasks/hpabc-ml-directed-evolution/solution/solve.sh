#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python3 - <<'PY'
import csv, json, math, re
import numpy as np

DATA = '/app/data'
OUT = '/app/output'
CONSERVED = {101, 104, 106, 109, 205, 397}
MUT_RE = re.compile(r'^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$')


def read_fasta(path):
    parts = []
    for line in open(path):
        if line.startswith('>'):
            continue
        parts.append(line.strip())
    return ''.join(parts)


def parse_muts(text):
    out = []
    for tok in re.split(r'[;,]', text or ''):
        tok = tok.strip()
        if not tok:
            continue
        m = MUT_RE.match(tok)
        out.append((int(m.group(2)), m.group(3)))
    return out


def spearman(x, y):
    a = np.asarray(x, float)
    b = np.asarray(y, float)
    if a.size < 3:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


seq = read_fasta(DATA + '/hpabc_alpha.fasta')
zero = {}
with open(DATA + '/zeroshot_scores.csv') as fh:
    for r in csv.DictReader(fh):
        zero[(int(r['position']), r['mut_aa'])] = float(r['zeroshot_score'])

single, multi = [], []
with open(DATA + '/single_mutants.csv') as fh:
    for r in csv.DictReader(fh):
        single.append(([(int(r['position']), r['mut_aa'])], float(r['relative_activity'])))
with open(DATA + '/de_variants.csv') as fh:
    for r in csv.DictReader(fh):
        multi.append((parse_muts(r['mutations']), float(r['relative_activity'])))
test = []
with open(DATA + '/test_variants.csv') as fh:
    for r in csv.DictReader(fh):
        test.append({'id': r['variant_id'], 'text': r['mutations'], 'muts': parse_muts(r['mutations'])})

# 1) calibrate the zero-shot scale on the measured single mutants
zs = np.array([zero[m[0]] for m, _ in single])
ys = np.array([math.log(a) for _, a in single])
b = float(zs @ ys / (zs @ zs))


def feature_matrix(rows, key_index):
    X = np.zeros((len(rows), len(key_index) + 1))
    y = np.zeros(len(rows))
    for i, (muts, act) in enumerate(rows):
        for k in muts:
            if k in key_index:
                X[i, key_index[k]] = 1.0
            else:
                X[i, -1] += zero[k]
        y[i] = math.log(act)
    return X, y


def ridge_fit(X, y, alpha):
    A = X.T @ X + alpha * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


# 2) choose the ridge strength by 5-fold CV on all experimental data
train = single + multi
keys = sorted({k for rows in (single, multi) for m, _ in rows for k in m})
idx = {k: i for i, k in enumerate(keys)}
X, y = feature_matrix(train, idx)
best_alpha, best_rho = 1.0, -9.0
for alpha in (0.3, 1.0, 3.0, 10.0, 30.0):
    pred = np.zeros(len(train))
    for fold in range(5):
        te = [i for i in range(len(train)) if i % 5 == fold]
        tr = [i for i in range(len(train)) if i % 5 != fold]
        pred[te] = X[te] @ ridge_fit(X[tr], y[tr], alpha)
    rho = spearman(pred, y)
    if rho > best_rho:
        best_alpha, best_rho = alpha, rho
coef = ridge_fit(X, y, best_alpha)

# 3) predict the held-out designs
pred = {}
for row in test:
    total = 0.0
    for k in row['muts']:
        if k in idx:
            total += coef[idx[k]]
        else:
            total += coef[-1] * zero[k]
    value = math.exp(total)
    if any(p in CONSERVED for p, _ in row['muts']):
        value = min(value, 0.05)
    pred[row['id']] = round(min(max(value, 0.01), 8.0), 4)

with open(OUT + '/predictions.csv', 'w', newline='') as fh:
    wr = csv.writer(fh)
    wr.writerow(['variant_id', 'predicted_activity'])
    for row in test:
        wr.writerow([row['id'], '%.4f' % pred[row['id']]])

ranked = sorted(pred, key=lambda v: (-pred[v], v))
with open(OUT + '/top10.csv', 'w', newline='') as fh:
    wr = csv.writer(fh)
    wr.writerow(['rank', 'variant_id', 'predicted_activity'])
    for i, vid in enumerate(ranked[:10]):
        wr.writerow([i + 1, vid, '%.4f' % pred[vid]])

by_id = {row['id']: row for row in test}
eligible = [row['id'] for row in test
            if len(row['muts']) <= 4 and not any(p in CONSERVED for p, _ in row['muts'])]
panel = sorted(eligible, key=lambda v: (-pred[v], v))[:5]
with open(OUT + '/panel.csv', 'w', newline='') as fh:
    wr = csv.writer(fh)
    wr.writerow(['variant_id', 'mutations', 'predicted_activity', 'rationale'])
    for vid in panel:
        wr.writerow([vid, by_id[vid]['text'], '%.4f' % pred[vid],
                     'best eligible design by cross-validated predicted activity'])


def cv_folds(rows, alpha):
    index = {k: i for i, k in enumerate(sorted({k for m, _ in rows for k in m}))}
    Xc, yc = feature_matrix(rows, index)
    out = np.zeros(len(rows))
    for fold in range(5):
        te = [i for i in range(len(rows)) if i % 5 == fold]
        tr = [i for i in range(len(rows)) if i % 5 != fold]
        out[te] = Xc[te] @ ridge_fit(Xc[tr], yc[tr], alpha)
    folds = []
    for fold in range(5):
        ii = [i for i in range(len(rows)) if i % 5 == fold]
        folds.append(round(spearman(out[ii], yc[ii]), 4))
    return folds


configs = []
for name, rows in (('single_mutant_ridge', single), ('single_plus_de_ridge', single + multi)):
    folds = cv_folds(rows, best_alpha)
    configs.append({'name': name, 'folds': folds, 'mean': round(sum(folds) / 5.0, 4)})
zpred = [b * sum(zero[k] for k in m) for m, _ in train]
ytrain = [math.log(a) for _, a in train]
zfolds = []
for fold in range(5):
    ii = [i for i in range(len(train)) if i % 5 == fold]
    zfolds.append(round(spearman([zpred[i] for i in ii], [ytrain[i] for i in ii]), 4))
configs.append({'name': 'zeroshot_only', 'folds': zfolds, 'mean': round(sum(zfolds) / 5.0, 4)})
best_cfg = max(configs, key=lambda c: c['mean'])
with open(OUT + '/cv_results.json', 'w') as fh:
    json.dump({'method': 'ridge regression on mutation indicator features with a zero-shot '
                         'slope fallback for substitutions lacking experimental data',
               'configs': configs, 'best_config': best_cfg['name'],
               'best_cv_spearman': best_cfg['mean']}, fh, indent=2)
print('outputs written')
PY
