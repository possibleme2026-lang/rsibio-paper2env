import numpy as np, pandas as pd, json, os

np.random.seed(42)
N_T = 5
N_CELLS_PER_T = 300
G = 24
N_CLONES = 180

fate_anchors_2d = np.array([[-2.0, 1.0], [2.0, 1.0], [0.0, -2.0]])
speeds = [1.0, 0.8, 0.6]

clone_fates = np.zeros(N_CLONES + 1, dtype=int)
clone_fates[1:61] = 0
clone_fates[61:121] = 1
clone_fates[121:181] = 2

clone_init = {}
for c in range(1, N_CLONES+1):
    clone_init[c] = np.random.normal(0, 0.5, size=2)

cells = []
for t in range(N_T):
    for i in range(N_CELLS_PER_T):
        clone = np.random.randint(1, N_CLONES+1)
        fate = clone_fates[clone]
        init = clone_init[clone]
        anchor = fate_anchors_2d[fate]
        frac = (t / (N_T - 1)) * speeds[fate]
        z = init + frac * (anchor - init) + np.random.normal(0, 0.1, size=2)
        cells.append({'cell_id': f'cell_t{t}_i{i}', 'timepoint': t, 'clone_id': clone, 'z2d': z, 'fate': fate})

proj = np.random.normal(0, 1, size=(2, 10))
latent = np.zeros((len(cells), 10))
for idx, cell in enumerate(cells):
    z = cell['z2d']
    latent[idx] = z @ proj + np.random.normal(0, 0.1, size=10)

anchors_10d = fate_anchors_2d @ proj

intercepts = np.random.normal(2.0, 0.5, size=G)
loadings = np.random.normal(0, 1.0, size=(G, 2))
modules = {'M1': list(range(0, 6)), 'M2': list(range(6, 12)), 'M3': list(range(12, 18)), 'M4': list(range(18, 24))}
module_effects = {'M1': {0: 1.5, 1: 0.0, 2: 0.0}, 'M2': {0: 0.0, 1: 1.5, 2: 0.0}, 'M3': {0: 0.0, 1: 0.0, 2: 1.5}, 'M4': {0: 0.5, 1: 0.5, 2: 0.5}}

counts = np.zeros((len(cells), G))
for idx, cell in enumerate(cells):
    z = cell['z2d']
    fate = cell['fate']
    t = cell['timepoint']
    for g in range(G):
        rate = np.exp(intercepts[g] + z @ loadings[g])
        for mod_name, gene_idx in modules.items():
            if g in gene_idx:
                rate *= np.exp(module_effects[mod_name].get(fate, 0.0))
                if mod_name == 'M4':
                    rate *= np.exp(0.2 * t)
        counts[idx, g] = np.random.poisson(rate * 5)

meta = pd.DataFrame([{'cell_id': c['cell_id'], 'timepoint': c['timepoint'], 'clone_id': c['clone_id']} for c in cells])
counts = counts.astype(np.int32)
latent = latent.astype(np.float32)
anchors_10d = anchors_10d.astype(np.float32)

os.makedirs('/app/data', exist_ok=True)
np.save('/app/data/counts.npy', counts)
np.save('/app/data/latent_pca.npy', latent)
np.save('/app/data/anchors.npy', anchors_10d)
meta.to_csv('/app/data/cell_metadata.csv', index=False)
with open('/app/data/gene_modules.json', 'w') as f:
    json.dump(modules, f)

tree = {'root': 0, 'branches': {'0': list(range(1, 61)), '1': list(range(61, 121)), '2': list(range(121, 181))}, 'edges': []}
for b in range(3):
    clones = tree['branches'][str(b)]
    for i in range(len(clones)-1):
        tree['edges'].append([clones[i], clones[i+1]])
for b in range(3):
    tree['edges'].append([0, tree['branches'][str(b)][0]])
with open('/app/data/lineage_tree.json', 'w') as f:
    json.dump(tree, f)

print('Data generated successfully.')
