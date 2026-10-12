import numpy as np

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

def generate_gold(seed=872634918273, shape=(128, 128, 64)):
    return generate_tree(seed, shape)
