#!/usr/bin/env python3
import argparse, json, numpy as np
from scipy.ndimage import gaussian_filter

def rotate(vec, axis, angle):
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return (vec * cos_a + np.cross(axis, vec) * sin_a + axis * np.dot(axis, vec) * (1 - cos_a))

def generate_tree(seed, shape=(128, 128, 64)):
    rng = np.random.RandomState(seed)
    nodes = []
    root_x = rng.uniform(shape[0]*0.4, shape[0]*0.6)
    root_y = rng.uniform(shape[1]*0.4, shape[1]*0.6)
    root_z = rng.uniform(shape[2]*0.3, shape[2]*0.7)
    nodes.append({'id': 1, 'type': 1, 'x': root_x, 'y': root_y, 'z': root_z, 'radius': 3.0, 'parent': -1})
    next_id = 2
    def add_segment(parent_id, start, direction, length, radius, depth):
        nonlocal next_id
        n_seg = max(1, int(length / rng.uniform(6, 10)))
        pos = np.array(start, dtype=float)
        dir = np.array(direction, dtype=float)
        dir /= np.linalg.norm(dir)
        for i in range(n_seg):
            seg_len = rng.uniform(4, 8)
            angle = rng.normal(0, 0.35)
            axis = rng.normal(size=3)
            if np.linalg.norm(axis) < 1e-6:
                axis = np.array([0, 0, 1.0])
            dir = rotate(dir, axis, angle)
            new_pos = pos + dir * seg_len
            new_pos = np.clip(new_pos, [1, 1, 1], [shape[0]-2, shape[1]-2, shape[2]-2])
            rad = max(0.6, radius * rng.uniform(0.85, 0.95))
            nodes.append({'id': next_id, 'type': 3, 'x': float(new_pos[0]), 'y': float(new_pos[1]), 'z': float(new_pos[2]), 'radius': float(rad), 'parent': parent_id})
            next_id += 1
            parent_id = next_id - 1
            pos = new_pos
            radius = rad
            if depth < 4 and rng.rand() < 0.35:
                n_branches = rng.randint(1, 3)
                for _ in range(n_branches):
                    branch_dir = dir + rng.normal(0, 0.9, size=3)
                    branch_dir /= np.linalg.norm(branch_dir)
                    branch_len = rng.uniform(12, 25)
                    add_segment(parent_id, pos, branch_dir, branch_len, radius*0.8, depth+1)
    init_dir = rng.normal(size=3)
    init_dir[2] *= 0.4
    init_dir /= np.linalg.norm(init_dir)
    add_segment(1, [root_x, root_y, root_z], init_dir, rng.uniform(35, 55), 2.5, 0)
    init_dir2 = rng.normal(size=3)
    init_dir2[2] *= 0.4
    init_dir2 /= np.linalg.norm(init_dir2)
    add_segment(1, [root_x, root_y, root_z], init_dir2, rng.uniform(25, 45), 2.5, 0)
    return nodes

def point_segment_distance(pt, p0, p1):
    v = p1 - p0
    w = pt - p0
    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(pt - p0)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(pt - p1)
    b = c1 / c2
    pb = p0 + b * v
    return np.linalg.norm(pt - pb)

def render_volume(nodes, shape=(128, 128, 64), seed=12345):
    rng = np.random.RandomState(seed)
    vol = np.zeros(shape, dtype=np.float32)
    for node in nodes:
        if node['parent'] == -1:
            continue
        parent = nodes[node['parent'] - 1]
        p0 = np.array([parent['x'], parent['y'], parent['z']])
        p1 = np.array([node['x'], node['y'], node['z']])
        r = (parent['radius'] + node['radius']) / 2.0
        min_xyz = np.floor(np.minimum(p0, p1) - 3*r).astype(int)
        max_xyz = np.ceil(np.maximum(p0, p1) + 3*r).astype(int)
        min_xyz = np.maximum(min_xyz, 0)
        max_xyz = np.minimum(max_xyz, np.array(shape) - 1)
        if np.any(min_xyz > max_xyz):
            continue
        for x in range(min_xyz[0], max_xyz[0] + 1):
            for y in range(min_xyz[1], max_xyz[1] + 1):
                for z in range(min_xyz[2], max_xyz[2] + 1):
                    pt = np.array([x, y, z], dtype=float)
                    d = point_segment_distance(pt, p0, p1)
                    if d < 2.5 * r:
                        intensity = np.exp(-(d ** 2) / (2 * (r ** 2)))
                        vol[x, y, z] += intensity
    vol += rng.normal(0, 0.05, size=shape)
    vol += rng.poisson(0.1, size=shape).astype(np.float32) * 0.05
    vol = gaussian_filter(vol, sigma=0.6)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    return vol

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--shape', type=str, default='128,128,64')
    args = parser.parse_args()
    shape = tuple(int(x) for x in args.shape.split(','))
    nodes = generate_tree(args.seed, shape)
    vol = render_volume(nodes, shape, seed=args.seed + 1)
    out_dir = args.out_dir
    np.save(f'{out_dir}/volume.npy', vol)
    root = nodes[0]
    voxel_size = [0.2, 0.2, 0.2]
    metadata = {
        'voxel_size_um': voxel_size,
        'volume_shape': list(shape),
        'seed_voxel': [float(root['x']), float(root['y']), float(root['z'])],
        'orientation': 'xyz',
        'description': 'Synthetic 3D fluorescence microscopy volume of a neuron. Voxel size 0.2 um isotropic.'
    }
    with open(f'{out_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == '__main__':
    main()
