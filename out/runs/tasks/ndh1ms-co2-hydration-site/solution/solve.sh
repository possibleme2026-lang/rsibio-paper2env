#!/bin/bash
set -euo pipefail
cat > /app/solve.py << 'EOF'
import json, math
from collections import deque
import numpy as np

def parse_pdb(path):
    atoms = []
    for line in open(path):
        if line.startswith(('ATOM','HETATM')):
            atoms.append({'record': line[:6].strip(), 'name': line[12:16].strip(),
                          'resname': line[17:20].strip(), 'chain': line[21],
                          'resseq': int(line[22:26]), 'x': float(line[30:38]),
                          'y': float(line[38:46]), 'z': float(line[46:54])})
    return atoms

def dist2(a,b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2

bound = parse_pdb('/app/data/ndh1ms_ndhv_bound.pdb')
free = parse_pdb('/app/data/ndh1ms_ndhv_free.pdb')

co2 = [a for a in bound if a['resname']=='CO2']
co2c = [(a['x'],a['y'],a['z']) for a in co2]
active = set()
for a in bound:
    if a['chain']=='A' and a['resname']!='CO2':
        for c in co2c:
            if dist2((a['x'],a['y'],a['z']),c) <= 25.0:
                active.add((a['resseq'],a['resname'])); break
active_site = [f"CupB:{r}:{n}" for r,n in sorted(active)]

chain_names = {'A':'CupB','B':'NdhF4','C':'NdhD4'}
polar = {'SER','THR','TYR','ASN','GLN','HIS','GLU','ASP','LYS','ARG'}
side = {'OG','OG1','OH','OD1','ND2','OE1','NE2','ND1','OE2','NZ','NH1','NH2'}
nodes = []
for a in bound:
    if a['resname']=='CO2':
        nodes.append(('CO2',(a['x'],a['y'],a['z']),True))
for a in bound:
    if a['resname']=='HOH':
        nodes.append((f"HOH:{a['resseq']}",(a['x'],a['y'],a['z']),False))
for a in bound:
    if a['chain'] in chain_names and a['resname'] in polar and a['name'] in side:
        nodes.append((f"{chain_names[a['chain']]}:{a['resseq']}:{a['resname']}:{a['name']}",(a['x'],a['y'],a['z']),False))
n = len(nodes)
adj = [[] for _ in range(n)]
for i in range(n):
    for j in range(i+1,n):
        if dist2(nodes[i][1],nodes[j][1]) <= 12.25:
            adj[i].append(j); adj[j].append(i)
co2_idx = [i for i,nd in enumerate(nodes) if nd[2]]
b_idx = [i for i,nd in enumerate(nodes) if nd[0].startswith('NdhF4:')]
best = None
for s in co2_idx:
    q = deque([(s,[s])]); vis={s}
    while q:
        cur,path = q.popleft()
        if cur in b_idx:
            if best is None or len(path)<len(best): best = path
            break
        for nxt in adj[cur]:
            if nxt not in vis:
                vis.add(nxt); q.append((nxt,path+[nxt]))
proton_wire = [nodes[i][0] for i in best if not nodes[i][2]]
terminal = None
for i in best:
    if nodes[i][0].startswith('NdhF4:'):
        p = nodes[i][0].split(':'); terminal = f"{p[0]}:{p[1]}:{p[2]}"; break

t_atm = None
for a in bound:
    if a['chain']=='B' and a['resseq']==410 and a['name']=='NE2':
        t_atm = (a['x'],a['y'],a['z'])
bulky = {'PHE':203.4,'TRP':237.7,'TYR':203.4,'LEU':166.9,'ILE':166.9,'VAL':140.0,'MET':162.9}
res = {}
for a in bound:
    if a['chain']=='B' and a['resname'] in bulky:
        res.setdefault((a['resseq'],a['resname']),[]).append(a)
cand = []
for (r,nm),atms in res.items():
    md = min(dist2((x['x'],x['y'],x['z']),t_atm) for x in atms)
    if md <= 64.0: cand.append((r,nm,bulky[nm]))
cand.sort(key=lambda x:(-x[2],x[0]))
blocked = [f"NdhF4:{r}:{nm}" for r,nm,v in cand[:4]]

def pq_contacts(atoms):
    pq = [(a['x'],a['y'],a['z']) for a in atoms if a['resname']=='PQ']
    cnt = 0
    for a in atoms:
        if a['record']=='ATOM':
            for p in pq:
                if dist2((a['x'],a['y'],a['z']),p) <= 20.25: cnt += 1; break
    return cnt
pq_b = pq_contacts(bound)
pq_f = pq_contacts(free)

def bb(atoms):
    return np.array([(a['x'],a['y'],a['z']) for a in atoms if a['chain'] in ('A','B','C') and a['name'] in ('N','CA','C','O')])
def pq_coords(atoms):
    return np.array([(a['x'],a['y'],a['z']) for a in atoms if a['resname']=='PQ'])
bb_b = bb(bound); bb_f = bb(free)
pq_b_coords = pq_coords(bound); pq_f_coords = pq_coords(free)
bb_bm = bb_b.mean(0); bb_fm = bb_f.mean(0)
H = (bb_f-bb_fm).T @ (bb_b-bb_bm)
U,S,Vt = np.linalg.svd(H)
d = np.sign(np.linalg.det(Vt.T @ U.T))
R = Vt.T @ np.diag([1,1,d]) @ U.T
pq_f_al = (R @ (pq_f_coords-bb_fm).T).T + bb_bm
rmsd = round(float(np.sqrt(((pq_b_coords-pq_f_al)**2).sum()/len(pq_b_coords))),2)

ans = {
    "active_site_residues": active_site,
    "proton_wire": proton_wire,
    "terminal_ndhf4_residue": terminal,
    "blocked_channel_residues": blocked,
    "pq_contacts_bound": pq_b,
    "pq_contacts_free": pq_f,
    "pq_rmsd": rmsd,
    "ndhv_stabilizes_pq": pq_b > pq_f
}
with open('/app/answer.json','w') as f:
    json.dump(ans, f, indent=2)
print(json.dumps(ans, indent=2))
EOF
python /app/solve.py
