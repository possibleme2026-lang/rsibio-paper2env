#!/bin/bash
set -euo pipefail
mkdir -p /app/data/results
python - <<'PY'
import os
import csv
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure
import trimesh
import trimesh.curvature

N = 97
COORDS = np.arange(N, dtype=np.float64) - 48.0

def roi_mask(expr, x, y, z):
    if expr == 'all':
        return np.ones_like(x, dtype=bool)
    return eval(expr)

def process_volume(vol_id, roi_expr):
    path = f'/app/data/volumes/{vol_id}.npy'
    vol = np.load(path)
    vol_smooth = gaussian_filter(vol, sigma=1.0)
    verts, faces, normals, values = measure.marching_cubes(vol_smooth, level=0.5)
    verts_phys = verts - 48.0
    mesh = trimesh.Trimesh(vertices=verts_phys, faces=faces)
    H = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, 3.0)
    K = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, 3.0)
    vertex_areas = np.zeros(len(mesh.vertices))
    for i, face in enumerate(mesh.faces):
        a = mesh.area_faces[i] / 3.0
        vertex_areas[face] += a
    centroids = mesh.triangles_center
    cx, cy, cz = centroids[:,0], centroids[:,1], centroids[:,2]
    face_mask = roi_mask(roi_expr, cx, cy, cz)
    surface_area = float(np.sum(mesh.area_faces[face_mask]))
    vx, vy, vz = mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2]
    vert_mask = roi_mask(roi_expr, vx, vy, vz)
    w = vertex_areas[vert_mask]
    mean_abs_H = float(np.sum(np.abs(H[vert_mask]) * w) / np.sum(w))
    mean_abs_K = float(np.sum(np.abs(K[vert_mask]) * w) / np.sum(w))
    return {
        'surface_area_nm2': surface_area,
        'mean_abs_H_nm_inv': mean_abs_H,
        'mean_abs_K_nm_inv2': mean_abs_K
    }, mesh, H, K

metadata = {}
with open('/app/data/metadata.csv', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        metadata[row['volume_id']] = row['roi_expr']

results = {}
for vid in ['vol_00', 'vol_01', 'vol_02', 'vol_03', 'vol_04']:
    metrics, mesh, H, K = process_volume(vid, metadata[vid])
    results[vid] = metrics

with open('/app/data/results/curvature_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['volume_id', 'surface_area_nm2', 'mean_abs_H_nm_inv', 'mean_abs_K_nm_inv2'])
    for vid in ['vol_00', 'vol_01', 'vol_02', 'vol_03', 'vol_04']:
        m = results[vid]
        writer.writerow([vid, m['surface_area_nm2'], m['mean_abs_H_nm_inv'], m['mean_abs_K_nm_inv2']])

vid = 'vol_03'
vol = np.load(f'/app/data/volumes/{vid}.npy')
vol_smooth = gaussian_filter(vol, sigma=1.0)
verts, faces, normals, values = measure.marching_cubes(vol_smooth, level=0.5)
verts_phys = verts - 48.0
mesh = trimesh.Trimesh(vertices=verts_phys, faces=faces)
H = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, 3.0)
K = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, 3.0)
mask = (np.abs(H) < 0.1) & (K < -0.005) & (np.abs(mesh.vertices[:,2]) <= 8.0)
if np.sum(mask) > 0:
    neck_center = np.mean(mesh.vertices[mask], axis=0)
    neck_indices = np.where(mask)[0]
    idx = np.argmin(K[mask])
    best_idx = neck_indices[idx]
    H_val = H[best_idx]
    K_val = K[best_idx]
    disc = H_val**2 - K_val
    if disc < 0:
        disc = 0.0
    sqrt_disc = np.sqrt(disc)
    k1 = H_val + sqrt_disc
    k2 = H_val - sqrt_disc
    aniso = abs(k1 - k2) / (abs(k1) + abs(k2) + 1e-12)
else:
    neck_center = [0.0, 0.0, 0.0]
    k1, k2, aniso = 0.0, 0.0, 0.0

neck_data = {
    'volume_id': vid,
    'neck_center_nm': [float(x) for x in neck_center],
    'kappa1_nm_inv': float(k1),
    'kappa2_nm_inv': float(k2),
    'anisotropy_index': float(aniso)
}
with open('/app/data/results/neck_metrics.json', 'w') as f:
    json.dump(neck_data, f, indent=2)
PY
