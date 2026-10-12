#!/usr/bin/env python3
import os
import numpy as np
from abnumber import Chain
import MDAnalysis as mda
import pandas as pd
import random

os.makedirs('/app/data/structures', exist_ok=True)
os.makedirs('/app/data/trajectories', exist_ok=True)

nivo_VH = "QVQLVESGGGVVQPGRSLRLDCKASGITFSNSGMHWVRQAPGKGLEWVAVIWYDGSKRYYADSVKGRFTISRDNSKNTLFLQMNSLRAEDTAVYYCATNDDYWGQGTLVTVSS"
nivo_VL = "EIVLTQSPATLSLSPGERATLSCRASQSVSSYLAWYQQKPGQAPRLLIYDASNRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQQSSNWPRTFGQGTKVEIK"
pembro_VH = "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS"
pembro_VL = "EIVLTQSPATLSLSPGERATLSCRASKGVSTSGYSYLHWYQQKPGQAPRLLIYLASYLESGVPARFSGSGSGTDFTLTISSLEPEDFAVYYCQHSRDLPLTFGGGTKVEIK"
pd1_seq = "APKAQIKESL"

aa3 = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU','G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}

def get_cdr_positions(seq):
    chain = Chain(seq, scheme='imgt')
    cdrs = {}
    for name, cdr_seq in [('CDR1', chain.cdr1_seq), ('CDR2', chain.cdr2_seq), ('CDR3', chain.cdr3_seq)]:
        if cdr_seq:
            idx = seq.find(cdr_seq)
            if idx != -1:
                cdrs[name] = (idx, idx+len(cdr_seq))
    return cdrs

def mutate_seq(seq, cdr_positions, mutations):
    seq_list = list(seq)
    for cdr_name, offset, new_aa in mutations:
        if cdr_name in cdr_positions:
            start, end = cdr_positions[cdr_name]
            pos = start + offset
            if pos < end:
                seq_list[pos] = new_aa
    return "".join(seq_list)

nivo_VH_cdrs = get_cdr_positions(nivo_VH)
nivo_VL_cdrs = get_cdr_positions(nivo_VL)
pembro_VH_cdrs = get_cdr_positions(pembro_VH)
pembro_VL_cdrs = get_cdr_positions(pembro_VL)

safe_aa = 'AVLIGSTP'
def random_safe_mutations(cdr_positions, n_muts, seed):
    rng = random.Random(seed)
    muts = []
    cdr_names = list(cdr_positions.keys())
    for _ in range(n_muts):
        cdr = rng.choice(cdr_names)
        start, end = cdr_positions[cdr]
        offset = rng.randint(0, end-start-1)
        aa = rng.choice(safe_aa)
        muts.append((cdr, offset, aa))
    return muts

variants = []
loops = ['H1','H2','H3','L1','L2','L3']
distances = [4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
for i in range(12):
    base = 0 if i < 6 else 1
    loop = loops[i % len(loops)]
    dist = distances[i % len(distances)]
    if base == 0:
        vh_muts = random_safe_mutations(nivo_VH_cdrs, 2, seed=i*10)
        vl_muts = random_safe_mutations(nivo_VL_cdrs, 2, seed=i*10+1)
    else:
        vh_muts = random_safe_mutations(pembro_VH_cdrs, 2, seed=i*10)
        vl_muts = random_safe_mutations(pembro_VL_cdrs, 2, seed=i*10+1)
    variants.append((base, loop, dist, vh_muts, vl_muts))

for i in range(4):
    base = 0 if i < 2 else 1
    loop = loops[i % len(loops)]
    dist = distances[(i+4) % len(distances)]
    if base == 0:
        vh_muts = [('CDR3', 0, 'N'), ('CDR3', 1, 'S')]
        vl_muts = random_safe_mutations(nivo_VL_cdrs, 1, seed=100+i)
    else:
        vh_muts = [('CDR3', 0, 'N'), ('CDR3', 1, 'S')]
        vl_muts = random_safe_mutations(pembro_VL_cdrs, 1, seed=200+i)
    variants.append((base, loop, dist, vh_muts, vl_muts))

for i in range(4):
    base = 0 if i < 2 else 1
    loop = loops[(i+2) % len(loops)]
    dist = distances[(i+6) % len(distances)]
    if base == 0:
        vh_muts = [('CDR2', 0, 'N'), ('CDR2', 1, 'G')]
        vl_muts = random_safe_mutations(nivo_VL_cdrs, 1, seed=300+i)
    else:
        vh_muts = [('CDR2', 0, 'N'), ('CDR2', 1, 'G')]
        vl_muts = random_safe_mutations(pembro_VL_cdrs, 1, seed=400+i)
    variants.append((base, loop, dist, vh_muts, vl_muts))

for i in range(4):
    base = 0 if i < 2 else 1
    loop = loops[(i+3) % len(loops)]
    dist = distances[(i+2) % len(distances)]
    if base == 0:
        vh_muts = [('CDR1', 0, 'M')]
        vl_muts = random_safe_mutations(nivo_VL_cdrs, 1, seed=500+i)
    else:
        vh_muts = [('CDR1', 0, 'M')]
        vl_muts = random_safe_mutations(pembro_VL_cdrs, 1, seed=600+i)
    variants.append((base, loop, dist, vh_muts, vl_muts))

variant_data = {}
for idx, (base, loop, dist, vh_muts, vl_muts) in enumerate(variants, 1):
    vid = f"P2N_{idx}"
    if base == 0:
        vh = mutate_seq(nivo_VH, nivo_VH_cdrs, vh_muts)
        vl = mutate_seq(nivo_VL, nivo_VL_cdrs, vl_muts)
    else:
        vh = mutate_seq(pembro_VH, pembro_VH_cdrs, vh_muts)
        vl = mutate_seq(pembro_VL, pembro_VL_cdrs, vl_muts)
    variant_data[vid] = {'vh': vh, 'vl': vl, 'binding_loop': loop, 'distance': dist}

with open('/app/data/variants.fasta', 'w') as f:
    for vid, data in variant_data.items():
        f.write(f">{vid}_H\n{data['vh']}\n")
        f.write(f">{vid}_L\n{data['vl']}\n")

with open('/app/data/pd1_epitope.fasta', 'w') as f:
    f.write(f">PD1_epitope\n{pd1_seq}\n")

weights = []
for i in range(len(pd1_seq)):
    resid = i + 1
    w = 2.0 if resid in (3, 7) else 1.0
    weights.append((resid, w))
pd.DataFrame(weights, columns=['residue', 'weight']).to_csv('/app/data/hotspot_weights.csv', index=False)

def random_coil(n, seed):
    rng = np.random.RandomState(seed)
    coords = np.zeros((n, 3))
    step = 3.8
    for i in range(1, n):
        vec = rng.randn(3)
        vec /= np.linalg.norm(vec)
        coords[i] = coords[i-1] + step * vec
    return coords

def write_pdb(filename, chains):
    with open(filename, 'w') as f:
        atom_serial = 1
        for chain_id, seq, coords in chains:
            for i, (aa, xyz) in enumerate(zip(seq, coords)):
                resname = aa3.get(aa, 'UNK')
                f.write(f"ATOM  {atom_serial:5d}  CA  {resname:3s} {chain_id}{i+1:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C\n")
                atom_serial += 1
        f.write("END\n")

for vid, data in variant_data.items():
    vh = data['vh']
    vl = data['vl']
    loop = data['binding_loop']
    dist = data['distance']
    vh_cdrs = get_cdr_positions(vh)
    vl_cdrs = get_cdr_positions(vl)
    if loop.startswith('H'):
        chain_id = 'H'
        cdr_name = 'CDR' + loop[1]
        if cdr_name in vh_cdrs:
            start, end = vh_cdrs[cdr_name]
            loop_res = list(range(start, end))
        else:
            loop_res = []
    else:
        chain_id = 'L'
        cdr_name = 'CDR' + loop[1]
        if cdr_name in vl_cdrs:
            start, end = vl_cdrs[cdr_name]
            loop_res = list(range(start, end))
        else:
            loop_res = []
    seed = int(vid.split('_')[1])
    vh_coords = random_coil(len(vh), seed)
    vl_coords = random_coil(len(vl), seed+1000)
    vl_coords += np.array([10.0, 0.0, 0.0])
    if loop_res:
        if chain_id == 'H':
            centroid = vh_coords[loop_res].mean(axis=0)
        else:
            centroid = vl_coords[loop_res].mean(axis=0)
    else:
        centroid = vh_coords.mean(axis=0)
    direction = np.array([1.0, 1.0, 0.0])
    direction /= np.linalg.norm(direction)
    pd1_coords = np.zeros((len(pd1_seq), 3))
    pd1_coords[0] = centroid + dist * direction
    perp = np.array([0.0, 0.0, 1.0])
    for i in range(1, len(pd1_seq)):
        pd1_coords[i] = pd1_coords[i-1] + 3.8 * perp
    pdb_path = f'/app/data/structures/{vid}.pdb'
    write_pdb(pdb_path, [('H', vh, vh_coords), ('L', vl, vl_coords), ('P', pd1_seq, pd1_coords)])
    u = mda.Universe(pdb_path)
    initial_positions = u.atoms.positions.copy()
    dcd_path = f'/app/data/trajectories/{vid}.dcd'
    with mda.Writer(dcd_path, u.atoms.n_atoms) as W:
        for frame in range(200):
            noise = np.random.RandomState(seed*1000 + frame).normal(0, 1.0, size=(u.atoms.n_atoms, 3))
            u.atoms.positions = initial_positions + noise
            W.write(u.atoms)

print("Data generation complete.")
