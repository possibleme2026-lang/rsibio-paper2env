#!/usr/bin/env python3
import json, sys, os, math
import numpy as np
from collections import defaultdict, deque
from scipy.spatial import cKDTree

sys.path.append('/tests')
import gold_gen

VOLUME_SHAPE = (128, 128, 64)
VOXEL_SIZE = np.array([0.2, 0.2, 0.2])
SEED_VOXEL = None

def load_metadata():
    with open('/app/data/metadata.json') as f:
        return json.load(f)

def parse_swc(path):
    errors = []
    nodes = []
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) != 7:
            errors.append(f'Line {i}: expected 7 columns, got {len(parts)}')
            continue
        try:
            nid = int(parts[0])
            ntype = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            radius = float(parts[5])
            parent = int(parts[6])
        except ValueError:
            errors.append(f'Line {i}: non-numeric value')
            continue
        nodes.append({'id': nid, 'type': ntype, 'x': x, 'y': y, 'z': z, 'radius': radius, 'parent': parent, 'line': i})
    return nodes, errors

def check_swc_format(nodes, errors):
    if errors:
        return False, errors
    if len(nodes) == 0:
        return False, ['No nodes found']
    ids = [n['id'] for n in nodes]
    if len(ids) != len(set(ids)):
        return False, ['Duplicate IDs']
    if any(i <= 0 for i in ids):
        return False, ['Non-positive ID']
    roots = [n for n in nodes if n['parent'] == -1]
    if len(roots) != 1:
        return False, [f'Expected exactly 1 root, found {len(roots)}']
    if roots[0]['type'] != 1:
        return False, ['Root type must be 1 (soma)']
    id_set = set(ids)
    for n in nodes:
        if n['parent'] != -1:
            if n['parent'] not in id_set:
                return False, ['Node ' + str(n['id']) + ' parent ' + str(n['parent']) + ' does not exist']
            if n['parent'] >= n['id']:
                return False, ['Node ' + str(n['id']) + ' parent ' + str(n['parent']) + ' is not smaller (cycle risk)']
    return True, ['OK']

def check_connectivity(nodes):
    if not nodes:
        return False, 'No nodes'
    id_to_node = {n['id']: n for n in nodes}
    root = next((n for n in nodes if n['parent'] == -1), None)
    if root is None:
        return False, 'No root'
    children = defaultdict(list)
    for n in nodes:
        if n['parent'] != -1:
            children[n['parent']].append(n['id'])
    visited = set()
    q = deque([root['id']])
    while q:
        nid = q.popleft()
        if nid in visited:
            return False, f'Cycle detected at {nid}'
        visited.add(nid)
        for c in children[nid]:
            q.append(c)
    if len(visited) != len(nodes):
        return False, f'Not all nodes reachable: visited {len(visited)} of {len(nodes)}'
    return True, 'OK'

def compute_graph_metrics(nodes):
    id_to_node = {n['id']: n for n in nodes}
    total_len = 0.0
    branch_points = []
    degrees = defaultdict(int)
    for n in nodes:
        if n['parent'] != -1:
            p = id_to_node[n['parent']]
            dx = n['x'] - p['x']
            dy = n['y'] - p['y']
            dz = n['z'] - p['z']
            total_len += math.sqrt(dx*dx + dy*dy + dz*dz) * np.mean(VOXEL_SIZE)
            degrees[n['id']] += 1
            degrees[n['parent']] += 1
    for n in nodes:
        if degrees[n['id']] >= 3:
            branch_points.append(n)
    return total_len, branch_points

def f1_score(gold_nodes, agent_nodes, tol_um):
    if not gold_nodes or not agent_nodes:
        return 0.0, 0.0, 0.0
    gold_xyz = np.array([[n['x'], n['y'], n['z']] for n in gold_nodes])
    agent_xyz = np.array([[n['x'], n['y'], n['z']] for n in agent_nodes])
    tree_agent = cKDTree(agent_xyz)
    dist_gold, _ = tree_agent.query(gold_xyz)
    recall = np.mean(dist_gold <= tol_um)
    tree_gold = cKDTree(gold_xyz)
    dist_agent, _ = tree_gold.query(agent_xyz)
    precision = np.mean(dist_agent <= tol_um)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def branch_f1(gold_nodes, agent_nodes, tol_um):
    def get_branch_points(nodes):
        id_to_node = {n['id']: n for n in nodes}
        degrees = defaultdict(int)
        for n in nodes:
            if n['parent'] != -1:
                degrees[n['id']] += 1
                degrees[n['parent']] += 1
        bps = []
        for n in nodes:
            if degrees[n['id']] >= 3:
                bps.append([n['x'], n['y'], n['z']])
        return np.array(bps) if bps else np.zeros((0,3))
    gold_bp = get_branch_points(gold_nodes)
    agent_bp = get_branch_points(agent_nodes)
    if len(gold_bp) == 0 or len(agent_bp) == 0:
        return 0.0, 0.0, 0.0
    tree_agent = cKDTree(agent_bp)
    dist_gold, _ = tree_agent.query(gold_bp)
    recall = np.mean(dist_gold <= tol_um)
    tree_gold = cKDTree(gold_bp)
    dist_agent, _ = tree_gold.query(agent_bp)
    precision = np.mean(dist_agent <= tol_um)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def radius_mae(gold_nodes, agent_nodes, tol_um=2.0):
    if not gold_nodes or not agent_nodes:
        return 999.0
    gold_xyz = np.array([[n['x'], n['y'], n['z']] for n in gold_nodes])
    agent_xyz = np.array([[n['x'], n['y'], n['z']] for n in agent_nodes])
    tree_agent = cKDTree(agent_xyz)
    dist, idx = tree_agent.query(gold_xyz)
    mask = dist <= tol_um
    if mask.sum() == 0:
        return 999.0
    gold_r = np.array([gold_nodes[i]['radius'] for i in range(len(gold_nodes)) if mask[i]])
    agent_r = np.array([agent_nodes[idx[i]]['radius'] for i in range(len(gold_nodes)) if mask[i]])
    return np.mean(np.abs(gold_r - agent_r))

def main():
    global VOXEL_SIZE, SEED_VOXEL
    meta = load_metadata()
    VOXEL_SIZE = np.array(meta['voxel_size_um'])
    SEED_VOXEL = np.array(meta['seed_voxel'])
    seed_um = SEED_VOXEL * VOXEL_SIZE

    agent_path = '/app/output/trace.swc'
    if not os.path.exists(agent_path):
        print('ERROR: /app/output/trace.swc not found')
        reward = {'score': 0.0, 'reward': 0.0}
        os.makedirs('/logs/verifier', exist_ok=True)
        with open('/logs/verifier/reward.json', 'w') as f:
            json.dump(reward, f)
        return

    nodes, parse_errors = parse_swc(agent_path)
    gold_nodes_vox = gold_gen.generate_gold(seed=872634918273, shape=VOLUME_SHAPE)
    gold_nodes_um = []
    for n in gold_nodes_vox:
        gold_nodes_um.append({
            'id': n['id'],
            'type': n['type'],
            'x': n['x'] * VOXEL_SIZE[0],
            'y': n['y'] * VOXEL_SIZE[1],
            'z': n['z'] * VOXEL_SIZE[2],
            'radius': n['radius'] * np.mean(VOXEL_SIZE),
            'parent': n['parent']
        })

    constraints = {}
    details = {}

    fmt_ok, fmt_details = check_swc_format(nodes, parse_errors)
    constraints['swc_format'] = 1.0 if fmt_ok else 0.0
    details['swc_format'] = fmt_details

    if fmt_ok:
        conn_ok, conn_details = check_connectivity(nodes)
    else:
        conn_ok, conn_details = False, 'Skipped due to format errors'
    constraints['connectivity'] = 1.0 if conn_ok else 0.0
    details['connectivity'] = conn_details

    if fmt_ok and nodes:
        root = next((n for n in nodes if n['parent'] == -1), None)
        if root:
            dist = math.sqrt((root['x'] - seed_um[0])**2 + (root['y'] - seed_um[1])**2 + (root['z'] - seed_um[2])**2)
            constraints['root_proximity'] = 1.0 if dist <= 1.5 else max(0.0, 1.0 - (dist - 1.5) / 5.0)
            details['root_proximity'] = f'distance={dist:.3f} um (tol 1.5)'
        else:
            constraints['root_proximity'] = 0.0
            details['root_proximity'] = 'No root'
    else:
        constraints['root_proximity'] = 0.0
        details['root_proximity'] = 'Skipped'

    gold_len, gold_bps = compute_graph_metrics(gold_nodes_um)
    if nodes:
        agent_len, agent_bps = compute_graph_metrics(nodes)
        if gold_len > 0:
            ratio = agent_len / gold_len
            if 0.7 <= ratio <= 1.3:
                constraints['path_length_ratio'] = 1.0
            else:
                constraints['path_length_ratio'] = max(0.0, 1.0 - abs(ratio - 1.0) / 0.3)
            details['path_length_ratio'] = f'agent={agent_len:.2f} um, gold={gold_len:.2f} um, ratio={ratio:.3f}'
        else:
            constraints['path_length_ratio'] = 0.0
            details['path_length_ratio'] = 'Gold length zero'
    else:
        constraints['path_length_ratio'] = 0.0
        details['path_length_ratio'] = 'No agent nodes'

    if nodes:
        prec, rec, f1 = f1_score(gold_nodes_um, nodes, tol_um=2.0)
        constraints['node_f1'] = min(1.0, f1 / 0.7)
        details['node_f1'] = f'precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f} (target 0.7)'
    else:
        constraints['node_f1'] = 0.0
        details['node_f1'] = 'No agent nodes'

    if nodes:
        prec, rec, f1 = branch_f1(gold_nodes_um, nodes, tol_um=3.0)
        constraints['branch_f1'] = min(1.0, f1 / 0.6)
        details['branch_f1'] = f'precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f} (target 0.6)'
    else:
        constraints['branch_f1'] = 0.0
        details['branch_f1'] = 'No agent nodes'

    if nodes:
        mae = radius_mae(gold_nodes_um, nodes, tol_um=2.0)
        if mae <= 0.5:
            constraints['radius_mae'] = 1.0
        else:
            constraints['radius_mae'] = max(0.0, 1.0 - (mae - 0.5) / 0.5)
        details['radius_mae'] = f'MAE={mae:.3f} um (target <= 0.5)'
    else:
        constraints['radius_mae'] = 0.0
        details['radius_mae'] = 'No agent nodes'

    if nodes:
        min_xyz = np.array([min(n['x'] for n in nodes), min(n['y'] for n in nodes), min(n['z'] for n in nodes)])
        max_xyz = np.array([max(n['x'] for n in nodes), max(n['y'] for n in nodes), max(n['z'] for n in nodes)])
        phys_shape = np.array(VOLUME_SHAPE) * VOXEL_SIZE
        in_bounds = (min_xyz >= -0.01).all() and (max_xyz <= phys_shape + 0.01).all()
        constraints['bounds'] = 1.0 if in_bounds else 0.0
        details['bounds'] = f'min={min_xyz}, max={max_xyz}, shape={phys_shape}'
    else:
        constraints['bounds'] = 0.0
        details['bounds'] = 'No agent nodes'

    total = sum(constraints.values())
    score = total / len(constraints)
    reward = {'score': float(score), 'reward': float(score)}

    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump(reward, f)
    report = {'constraints': constraints, 'details': details, 'score': score}
    with open('/logs/verifier/report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
