import numpy as np
from skimage import measure
import trimesh
import trimesh.curvature

N = 97
COORDS = np.arange(N, dtype=np.float64) - 48.0

def get_sdf(vol_id):
    X, Y, Z = np.meshgrid(COORDS, COORDS, COORDS, indexing='ij')
    if vol_id == 'vol_00':
        return np.sqrt(X**2 + Y**2 + Z**2) - 20.0
    elif vol_id == 'vol_01':
        return np.maximum(np.sqrt(X**2 + Y**2) - 12.0, np.abs(Z) - 30.0)
    elif vol_id == 'vol_02':
        q = np.sqrt(X**2 + Y**2) - 25.0
        return np.sqrt(q**2 + Z**2) - 8.0
    elif vol_id == 'vol_03':
        a = 8.0
        return np.sqrt(X**2 + Y**2) - a * np.cosh(Z / a)
    elif vol_id == 'vol_04':
        A = 5.0
        lam = 32.0
        return Z - A * np.sin(2 * np.pi * X / lam) * np.cos(2 * np.pi * Y / lam)
    else:
        raise ValueError(vol_id)

def get_roi_expr(vol_id):
    if vol_id == 'vol_00':
        return 'all'
    elif vol_id == 'vol_01':
        return 'abs(z)<=25'
    elif vol_id == 'vol_02':
        return 'all'
    elif vol_id == 'vol_03':
        return 'abs(z)<=8'
    elif vol_id == 'vol_04':
        return '(abs(x)<=24)&(abs(y)<=24)'
    else:
        raise ValueError(vol_id)

def compute_gold_metrics():
    gold = {}
    for vol_id in ['vol_00', 'vol_01', 'vol_02', 'vol_03', 'vol_04']:
        sdf = get_sdf(vol_id)
        intensity = np.exp(-(sdf / 1.2)**2) + 0.05
        verts, faces, normals, values = measure.marching_cubes(intensity, level=0.5)
        verts_phys = verts - 48.0
        mesh = trimesh.Trimesh(vertices=verts_phys, faces=faces)
        H = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, 3.0)
        K = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, 3.0)
        vertex_areas = np.zeros(len(mesh.vertices))
        for i, face in enumerate(mesh.faces):
            a = mesh.area_faces[i] / 3.0
            vertex_areas[face] += a
        centroids = mesh.triangles_center
        x, y, z = centroids[:, 0], centroids[:, 1], centroids[:, 2]
        roi_expr = get_roi_expr(vol_id)
        if roi_expr == 'all':
            face_mask = np.ones(len(mesh.faces), dtype=bool)
        else:
            face_mask = eval(roi_expr)
        surface_area = float(np.sum(mesh.area_faces[face_mask]))
        vx, vy, vz = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
        if roi_expr == 'all':
            vert_mask = np.ones(len(mesh.vertices), dtype=bool)
        else:
            # x,y,z hold FACE centroids (used for face_mask); the vertex
            # ROI must be evaluated on the vertex coordinates
            # vx,vy,vz, otherwise the mask has len(faces) entries and
            # vertex_areas[vert_mask] raises IndexError.
            x, y, z = vx, vy, vz
            vert_mask = eval(roi_expr)
        w = vertex_areas[vert_mask]
        mean_abs_H = float(np.sum(np.abs(H[vert_mask]) * w) / np.sum(w))
        mean_abs_K = float(np.sum(np.abs(K[vert_mask]) * w) / np.sum(w))
        gold[vol_id] = {
            'surface_area_nm2': surface_area,
            'mean_abs_H_nm_inv': mean_abs_H,
            'mean_abs_K_nm_inv2': mean_abs_K
        }
    gold['neck'] = {
        'volume_id': 'vol_03',
        'neck_center_nm': [0.0, 0.0, 0.0],
        'kappa1_nm_inv': 0.125,
        'kappa2_nm_inv': -0.125,
        'anisotropy_index': 1.0
    }
    return gold

if __name__ == '__main__':
    import json
    print(json.dumps(compute_gold_metrics(), indent=2))
