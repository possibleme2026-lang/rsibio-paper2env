import json, math, os, sys
import numpy as np
from collections import deque

def parse_pdb(path):
    atoms = []
    with open(path) as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                record = line[:6].strip()
                serial = int(line[6:11])
                name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21]
                resseq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip()
                atoms.append({'record': record, 'serial': serial, 'name': name,
                              'resname': resname, 'chain': chain, 'resseq': resseq,
                              'x': x, 'y': y, 'z': z, 'element': element})
    return atoms

def dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2

def expected_active_site(atoms):
    co2 = [a for a in atoms if a['resname'] == 'CO2']
    co2_coords = [(a['x'], a['y'], a['z']) for a in co2]
    residues = {}
    for a in atoms:
        if a['chain'] == 'A' and a['resname'] != 'CO2':
            key = (a['resseq'], a['resname'])
            residues.setdefault(key, []).append(a)
    active = set()
    for (resseq, resname), atms in residues.items():
        for atm in atms:
            for cx, cy, cz in co2_coords:
                if dist2((atm['x'], atm['y'], atm['z']), (cx, cy, cz)) <= 25.0:
                    active.add((resseq, resname))
                    break
            else:
                continue
            break
    sorted_active = sorted(active, key=lambda x: x[0])
    return [f"CupB:{r}:{n}" for r, n in sorted_active]

def expected_proton_wire(atoms):
    chain_names = {'A': 'CupB', 'B': 'NdhF4', 'C': 'NdhD4'}
    polar_res = {'SER','THR','TYR','ASN','GLN','HIS','GLU','ASP','LYS','ARG'}
    side_atoms = {'OG','OG1','OH','OD1','ND2','OE1','NE2','ND1','OE2','NZ','NH1','NH2'}
    nodes = []
    for a in atoms:
        if a['resname'] == 'CO2':
            nodes.append(('CO2', (a['x'], a['y'], a['z']), True))
    for a in atoms:
        if a['resname'] == 'HOH':
            nodes.append((f"HOH:{a['resseq']}", (a['x'], a['y'], a['z']), False))
    for a in atoms:
        if a['chain'] in chain_names and a['resname'] in polar_res and a['name'] in side_atoms:
            label = f"{chain_names[a['chain']]}:{a['resseq']}:{a['resname']}:{a['name']}"
            nodes.append((label, (a['x'], a['y'], a['z']), False))
    n = len(nodes)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if dist2(nodes[i][1], nodes[j][1]) <= 12.25:
                adj[i].append(j)
                adj[j].append(i)
    chain_b_indices = [i for i, nd in enumerate(nodes) if nd[0].startswith('NdhF4:')]
    co2_indices = [i for i, nd in enumerate(nodes) if nd[2]]
    best_path = None
    for start in co2_indices:
        q = deque([(start, [start])])
        visited = {start}
        while q:
            cur, path = q.popleft()
            if cur in chain_b_indices:
                if best_path is None or len(path) < len(best_path):
                    best_path = path
                break
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, path + [nxt]))
    if best_path is None:
        return [], None
    path_labels = []
    for idx in best_path:
        if nodes[idx][2]:
            continue
        path_labels.append(nodes[idx][0])
    terminal = None
    for idx in best_path:
        if nodes[idx][0].startswith('NdhF4:'):
            parts = nodes[idx][0].split(':')
            terminal = f"{parts[0]}:{parts[1]}:{parts[2]}"
            break
    return path_labels, terminal

def expected_blocked_channel(atoms, terminal_label):
    parts = terminal_label.split(':')
    chain_name, resseq, resname = parts[0], int(parts[1]), parts[2]
    terminal_atom = None
    for a in atoms:
        if a['chain'] == 'B' and a['resseq'] == resseq and a['name'] == 'NE2':
            terminal_atom = (a['x'], a['y'], a['z'])
            break
    if terminal_atom is None:
        return []
    bulky = {'PHE':203.4, 'TRP':237.7, 'TYR':203.4, 'LEU':166.9, 'ILE':166.9, 'VAL':140.0, 'MET':162.9}
    residues = {}
    for a in atoms:
        if a['chain'] == 'B' and a['resname'] in bulky:
            key = (a['resseq'], a['resname'])
            residues.setdefault(key, []).append(a)
    candidates = []
    for (resseq, resname), atms in residues.items():
        min_d2 = min(dist2((a['x'], a['y'], a['z']), terminal_atom) for a in atms)
        if min_d2 <= 64.0:
            candidates.append((resseq, resname, bulky[resname]))
    candidates.sort(key=lambda x: (-x[2], x[0]))
    top4 = candidates[:4]
    return [f"NdhF4:{r}:{n}" for r, n, v in top4]

def count_pq_contacts(atoms):
    pq = [a for a in atoms if a['resname'] == 'PQ']
    if not pq:
        return 0
    pq_coords = [(a['x'], a['y'], a['z']) for a in pq]
    count = 0
    for a in atoms:
        if a['record'] == 'ATOM':
            for px, py, pz in pq_coords:
                if dist2((a['x'], a['y'], a['z']), (px, py, pz)) <= 20.25:
                    count += 1
                    break
    return count

def compute_pq_rmsd(bound_atoms, free_atoms):
    def get_backbone(atoms):
        bb = []
        for a in atoms:
            if a['chain'] in ('A','B','C') and a['name'] in ('N','CA','C','O'):
                bb.append((a['x'], a['y'], a['z']))
        return np.array(bb)
    def get_pq(atoms):
        pq = []
        for a in atoms:
            if a['resname'] == 'PQ':
                pq.append((a['x'], a['y'], a['z']))
        return np.array(pq)
    bb_bound = get_backbone(bound_atoms)
    bb_free = get_backbone(free_atoms)
    pq_bound = get_pq(bound_atoms)
    pq_free = get_pq(free_atoms)
    if len(bb_bound) != len(bb_free) or len(pq_bound) != len(pq_free):
        return None
    bb_bound_mean = bb_bound.mean(axis=0)
    bb_free_mean = bb_free.mean(axis=0)
    bb_bound_c = bb_bound - bb_bound_mean
    bb_free_c = bb_free - bb_free_mean
    H = bb_free_c.T @ bb_bound_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    pq_free_c = pq_free - bb_free_mean
    pq_free_aligned = (R @ pq_free_c.T).T + bb_bound_mean
    diff = pq_bound - pq_free_aligned
    rmsd = np.sqrt((diff**2).sum() / len(pq_bound))
    return round(float(rmsd), 2)

def main():
    answer_path = '/app/answer.json'
    if not os.path.exists(answer_path):
        print("answer.json not found")
        reward = {"score": 0.0, "reward": 0.0}
        os.makedirs('/logs/verifier', exist_ok=True)
        with open('/logs/verifier/reward.json', 'w') as f:
            json.dump(reward, f)
        return
    try:
        with open(answer_path) as f:
            ans = json.load(f)
    except Exception as e:
        print(f"Failed to parse answer.json: {e}")
        reward = {"score": 0.0, "reward": 0.0}
        os.makedirs('/logs/verifier', exist_ok=True)
        with open('/logs/verifier/reward.json', 'w') as f:
            json.dump(reward, f)
        return
    bound_atoms = parse_pdb('/app/data/ndh1ms_ndhv_bound.pdb')
    free_atoms = parse_pdb('/app/data/ndh1ms_ndhv_free.pdb')
    exp_active = expected_active_site(bound_atoms)
    exp_proton_wire, exp_terminal = expected_proton_wire(bound_atoms)
    exp_blocked = expected_blocked_channel(bound_atoms, exp_terminal)
    exp_pq_bound = count_pq_contacts(bound_atoms)
    exp_pq_free = count_pq_contacts(free_atoms)
    exp_rmsd = compute_pq_rmsd(bound_atoms, free_atoms)
    exp_stab = exp_pq_bound > exp_pq_free
    checks = []
    required_keys = ['active_site_residues','proton_wire','terminal_ndhf4_residue',
                     'blocked_channel_residues','pq_contacts_bound','pq_contacts_free',
                     'pq_rmsd','ndhv_stabilizes_pq']
    checks.append(('C1_valid_json_keys', all(k in ans for k in required_keys)))
    checks.append(('C2_active_site', ans.get('active_site_residues') == exp_active))
    checks.append(('C3_proton_wire', ans.get('proton_wire') == exp_proton_wire))
    checks.append(('C4_terminal', ans.get('terminal_ndhf4_residue') == exp_terminal))
    checks.append(('C5_blocked_channel', ans.get('blocked_channel_residues') == exp_blocked))
    checks.append(('C6_pq_contacts', ans.get('pq_contacts_bound') == exp_pq_bound and ans.get('pq_contacts_free') == exp_pq_free))
    c7 = False
    if exp_rmsd is not None and 'pq_rmsd' in ans:
        try:
            c7 = abs(float(ans['pq_rmsd']) - exp_rmsd) <= 0.1
        except:
            pass
    checks.append(('C7_pq_rmsd', c7))
    checks.append(('C8_stabilization', ans.get('ndhv_stabilizes_pq') == exp_stab))
    passed = sum(1 for _, ok in checks if ok)
    score = passed / len(checks)
    print("="*50)
    print("GRADER REPORT")
    print("="*50)
    for name, ok in checks:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print("-"*50)
    print(f"Expected active site: {exp_active}")
    print(f"Agent active site:    {ans.get('active_site_residues')}")
    print(f"Expected proton wire: {exp_proton_wire}")
    print(f"Agent proton wire:    {ans.get('proton_wire')}")
    print(f"Expected terminal:    {exp_terminal}")
    print(f"Agent terminal:       {ans.get('terminal_ndhf4_residue')}")
    print(f"Expected blocked:     {exp_blocked}")
    print(f"Agent blocked:        {ans.get('blocked_channel_residues')}")
    print(f"Expected contacts:    bound={exp_pq_bound}, free={exp_pq_free}")
    print(f"Agent contacts:       bound={ans.get('pq_contacts_bound')}, free={ans.get('pq_contacts_free')}")
    print(f"Expected RMSD:        {exp_rmsd}")
    print(f"Agent RMSD:           {ans.get('pq_rmsd')}")
    print(f"Expected stabilizes:  {exp_stab}")
    print(f"Agent stabilizes:     {ans.get('ndhv_stabilizes_pq')}")
    print("="*50)
    print(f"Score: {score:.3f} ({passed}/{len(checks)})")
    os.makedirs('/logs/verifier', exist_ok=True)
    reward = {"score": float(score), "reward": float(score)}
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump(reward, f)

if __name__ == '__main__':
    main()
