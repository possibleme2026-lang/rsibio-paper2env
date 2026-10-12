#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python3 - <<'PY'
import json, numpy as np
from skimage.morphology import skeletonize
from scipy import ndimage
from scipy.spatial import cKDTree
import networkx as nx
from collections import deque

vol = np.load('/app/data/volume.npy')
meta = json.load(open('/app/data/metadata.json'))
vox = np.array(meta['voxel_size_um'])
seed_vox = np.array(meta['seed_voxel'])

thr = 0.25
mask = vol > thr
skel = skeletonize(mask)
dist = ndimage.distance_transform_edt(mask)
coords = np.argwhere(skel)
if len(coords) == 0:
    raise SystemExit('No skeleton')
tree = cKDTree(coords)
pairs = tree.query_pairs(r=1.8, output_type='ndarray')
G = nx.Graph()
G.add_nodes_from(range(len(coords)))
G.add_edges_from(pairs)

_, idx = tree.query(seed_vox)
root_idx = idx
parent = {root_idx: -1}
q = deque([root_idx])
while q:
    u = q.popleft()
    for v in G.neighbors(u):
        if v not in parent:
            parent[v] = u
            q.append(v)

children = {i: [] for i in range(len(coords))}
for v, p in parent.items():
    if p != -1:
        children[p].append(v)
deg = {i: len(children[i]) + (1 if parent.get(i, -1) != -1 else 0) for i in range(len(coords))}
endpoints = [i for i in range(len(coords)) if deg[i] == 1 and i != root_idx]
branch_nodes = [i for i in range(len(coords)) if len(children[i]) >= 2]
key_nodes = set([root_idx] + endpoints + branch_nodes)

depth = {root_idx: 0}
q = deque([root_idx])
while q:
    u = q.popleft()
    for v in children[u]:
        depth[v] = depth[u] + 1
        q.append(v)

selected = set([root_idx] + endpoints + branch_nodes)
for i in range(len(coords)):
    if depth.get(i, 0) % 4 == 0:
        selected.add(i)
selected = sorted(selected, key=lambda i: depth.get(i, 0))

new_id = {}
new_nodes = []
for i in selected:
    new_id[i] = len(new_nodes) + 1
    x, y, z = coords[i]
    r_vox = dist[tuple(coords[i])]
    r_um = r_vox * np.mean(vox)
    new_nodes.append({'id': new_id[i], 'type': 1 if i == root_idx else 3,
                      'x': float(x * vox[0]), 'y': float(y * vox[1]), 'z': float(z * vox[2]),
                      'radius': float(r_um), 'parent': -1})

for i in selected:
    if i == root_idx:
        continue
    p = parent[i]
    while p not in new_id:
        p = parent[p]
    new_nodes[new_id[i]-1]['parent'] = new_id[p]

with open('/app/output/trace.swc', 'w') as f:
    f.write('# id type x y z radius parent\n')
    for n in new_nodes:
        line = str(n['id']) + ' ' + str(n['type']) + ' ' + format(n['x'], '.4f') + ' ' + format(n['y'], '.4f') + ' ' + format(n['z'], '.4f') + ' ' + format(n['radius'], '.4f') + ' ' + str(n['parent']) + '\n'
        f.write(line)
print('Wrote ' + str(len(new_nodes)) + ' nodes')
PY
