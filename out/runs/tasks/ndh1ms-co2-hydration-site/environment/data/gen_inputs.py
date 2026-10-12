import math, random

SEED = 20260803

def write_pdb(filename, atoms):
    with open(filename, 'w') as f:
        for atm in atoms:
            if atm['record'] == 'ATOM':
                f.write(f"ATOM  {atm['serial']:5d} {atm['name']:<4s} {atm['resname']:>3s} {atm['chain']}{atm['resseq']:4d}    "
                        f"{atm['x']:8.3f}{atm['y']:8.3f}{atm['z']:8.3f}{1.00:6.2f}{0.00:6.2f}          {atm['element']:>2s}\n")
            else:
                f.write(f"HETATM{atm['serial']:5d} {atm['name']:<4s} {atm['resname']:>3s} {atm['chain']}{atm['resseq']:4d}    "
                        f"{atm['x']:8.3f}{atm['y']:8.3f}{atm['z']:8.3f}{1.00:6.2f}{0.00:6.2f}          {atm['element']:>2s}\n")

AA = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

def gen_random_chain(chain_id, n_res, start, seed_offset):
    atoms = []
    serial = 1
    rng = random.Random(SEED + seed_offset)
    pos = list(start)
    for i in range(n_res):
        resseq = i + 1
        resname = rng.choice(AA)
        ca = tuple(pos)
        atoms.append({'record':'ATOM','serial':serial,'name':'N','resname':resname,'chain':chain_id,'resseq':resseq,
                      'x':ca[0]-1.0,'y':ca[1]+0.5,'z':ca[2]-0.5,'element':'N'})
        serial += 1
        atoms.append({'record':'ATOM','serial':serial,'name':'CA','resname':resname,'chain':chain_id,'resseq':resseq,
                      'x':ca[0],'y':ca[1],'z':ca[2],'element':'C'})
        serial += 1
        atoms.append({'record':'ATOM','serial':serial,'name':'C','resname':resname,'chain':chain_id,'resseq':resseq,
                      'x':ca[0]+1.0,'y':ca[1]+0.5,'z':ca[2]+0.5,'element':'C'})
        serial += 1
        atoms.append({'record':'ATOM','serial':serial,'name':'O','resname':resname,'chain':chain_id,'resseq':resseq,
                      'x':ca[0]+1.5,'y':ca[1]+1.2,'z':ca[2]+0.0,'element':'O'})
        serial += 1
        if resname != 'GLY':
            atoms.append({'record':'ATOM','serial':serial,'name':'CB','resname':resname,'chain':chain_id,'resseq':resseq,
                          'x':ca[0]+0.5,'y':ca[1]-1.0,'z':ca[2]-0.5,'element':'C'})
            serial += 1
        keep = True
        for atm in atoms[-6:]:
            d2_origin = atm['x']**2 + atm['y']**2 + atm['z']**2
            if d2_origin < 100.0:
                keep = False
                break
            if 0 < atm['x'] < 30:
                d2_axis = atm['y']**2 + atm['z']**2
                if d2_axis < 36.0:
                    keep = False
                    break
            if (atm['x']-22.0)**2 + atm['y']**2 + atm['z']**2 < 64.0:
                keep = False
                break
        if not keep:
            atoms = atoms[:- (5 if resname == 'GLY' else 6)]
            serial -= (5 if resname == 'GLY' else 6)
        theta = rng.uniform(0, 2*math.pi)
        phi = rng.uniform(0, math.pi)
        dx = 3.8 * math.sin(phi) * math.cos(theta)
        dy = 3.8 * math.sin(phi) * math.sin(theta)
        dz = 3.8 * math.cos(phi)
        pos[0] += dx
        pos[1] += dy
        pos[2] += dz
    return atoms, serial

def make_manual_residue(resname, chain, resseq, coords, start_serial):
    atoms = []
    serial = start_serial
    for name, (x,y,z) in coords.items():
        element = name[0]
        if name in ('CA','CB','C'): element = 'C'
        elif name == 'N': element = 'N'
        elif name in ('O','OG','OG1','OD1','OE1','OH'): element = 'O'
        elif name in ('NE2','ND1','NH1','NH2','NZ'): element = 'N'
        atoms.append({'record':'ATOM','serial':serial,'name':name,'resname':resname,'chain':chain,'resseq':resseq,
                      'x':x,'y':y,'z':z,'element':element})
        serial += 1
    return atoms, serial

def main():
    atoms_A, _ = gen_random_chain('A', 350, (0,0,0), 1)
    atoms_B, _ = gen_random_chain('B', 600, (0,0,0), 2)
    atoms_C, _ = gen_random_chain('C', 500, (0,0,0), 3)
    atoms_D, _ = gen_random_chain('D', 120, (40,10,10), 4)
    def remove_residues(atoms, resseqs):
        return [a for a in atoms if a['resseq'] not in resseqs]
    atoms_A = remove_residues(atoms_A, {45,78,102,156,201,234})
    atoms_B = remove_residues(atoms_B, {410,512,515,518,521,530})
    serial = max([a['serial'] for a in atoms_A] + [0]) + 1
    manual_A = []
    res, serial = make_manual_residue('GLY','A',45, {
        'N':(-1.0,4.5,-0.5), 'CA':(0,4.0,0), 'C':(1.0,4.5,0.5), 'O':(1.5,5.2,0.0)
    }, serial); manual_A.extend(res)
    res, serial = make_manual_residue('SER','A',78, {
        'N':(-1.0,-5.5,-0.5), 'CA':(0,-5.0,0), 'C':(1.0,-4.5,0.5), 'O':(1.5,-3.8,0.0),
        'CB':(0.5,-5.5,-0.5), 'OG':(0,-4.0,0)
    }, serial); manual_A.extend(res)
    res, serial = make_manual_residue('HIS','A',102, {
        'N':(-1.0,0.5,5.5), 'CA':(0,0,5.0), 'C':(1.0,0.5,5.5), 'O':(1.5,1.2,5.0),
        'CB':(0.5,-1.0,5.5), 'NE2':(0,0,4.0)
    }, serial); manual_A.extend(res)
    res, serial = make_manual_residue('ASP','A',156, {
        'N':(-1.0,0.5,-5.5), 'CA':(0,0,-5.0), 'C':(1.0,0.5,-4.5), 'O':(1.5,1.2,-5.0),
        'CB':(0.5,-1.0,-5.5), 'OD1':(0,0,-4.0)
    }, serial); manual_A.extend(res)
    res, serial = make_manual_residue('THR','A',201, {
        'N':(1.0,5.5,-0.5), 'CA':(2.0,5.0,0), 'C':(3.0,5.5,0.5), 'O':(3.5,6.2,0.0),
        'CB':(2.5,4.0,-0.5), 'OG1':(2.5,3.5,0)
    }, serial); manual_A.extend(res)
    res, serial = make_manual_residue('GLN','A',234, {
        'N':(-3.0,5.5,-0.5), 'CA':(-2.0,5.0,0), 'C':(-1.0,5.5,0.5), 'O':(-0.5,6.2,0.0),
        'CB':(-1.5,4.0,-0.5), 'OE1':(-2.5,3.5,0)
    }, serial); manual_A.extend(res)
    atoms_A.extend(manual_A)
    serial = max([a['serial'] for a in atoms_B] + [0]) + 1
    manual_B = []
    res, serial = make_manual_residue('HIS','B',410, {
        'N':(21.0,0.5,0.5), 'CA':(22.0,0,1.0), 'C':(23.0,0.5,1.5), 'O':(23.5,1.2,1.0),
        'CB':(22.5,-1.0,0.5), 'NE2':(22.0,0,0)
    }, serial); manual_B.extend(res)
    res, serial = make_manual_residue('PHE','B',512, {
        'N':(21.0,5.5,-0.5), 'CA':(22.0,5.0,0), 'C':(23.0,5.5,0.5), 'O':(23.5,6.2,0.0),
        'CB':(22.5,4.0,-0.5)
    }, serial); manual_B.extend(res)
    res, serial = make_manual_residue('TRP','B',515, {
        'N':(21.0,-5.5,-0.5), 'CA':(22.0,-5.0,0), 'C':(23.0,-4.5,0.5), 'O':(23.5,-3.8,0.0),
        'CB':(22.5,-6.0,-0.5)
    }, serial); manual_B.extend(res)
    res, serial = make_manual_residue('LEU','B',518, {
        'N':(21.0,0.5,5.5), 'CA':(22.0,0,5.0), 'C':(23.0,0.5,5.5), 'O':(23.5,1.2,5.0),
        'CB':(22.5,-1.0,5.5)
    }, serial); manual_B.extend(res)
    res, serial = make_manual_residue('ILE','B',521, {
        'N':(21.0,0.5,-5.5), 'CA':(22.0,0,-5.0), 'C':(23.0,0.5,-4.5), 'O':(23.5,1.2,-5.0),
        'CB':(22.5,-1.0,-5.5)
    }, serial); manual_B.extend(res)
    res, serial = make_manual_residue('MET','B',530, {
        'N':(30.0,0.5,-0.5), 'CA':(31.0,0,0), 'C':(32.0,0.5,0.5), 'O':(32.5,1.2,0.0),
        'CB':(31.5,-1.0,-0.5)
    }, serial); manual_B.extend(res)
    atoms_B.extend(manual_B)
    waters = []
    serial = 20000
    for i, x in enumerate([4.5, 8.0, 11.5, 15.0, 18.5], start=1):
        waters.append({'record':'HETATM','serial':serial,'name':'O','resname':'HOH','chain':'W','resseq':i,
                       'x':x,'y':0.0,'z':0.0,'element':'O'})
        serial += 1
    co2 = [
        {'record':'HETATM','serial':21000,'name':'C','resname':'CO2','chain':'A','resseq':1000,
         'x':0.0,'y':0.0,'z':0.0,'element':'C'},
        {'record':'HETATM','serial':21001,'name':'O1','resname':'CO2','chain':'A','resseq':1000,
         'x':1.5,'y':0.0,'z':0.0,'element':'O'},
        {'record':'HETATM','serial':21002,'name':'O2','resname':'CO2','chain':'A','resseq':1000,
         'x':-1.5,'y':0.0,'z':0.0,'element':'O'}
    ]
    def make_pq(cx, cy, cz, start_serial):
        atoms = []
        serial = start_serial
        coords = [
            (cx+1.4, cy, cz), (cx+0.7, cy+1.2, cz), (cx-0.7, cy+1.2, cz),
            (cx-1.4, cy, cz), (cx-0.7, cy-1.2, cz), (cx+0.7, cy-1.2, cz)
        ]
        for i, (x,y,z) in enumerate(coords, start=1):
            atoms.append({'record':'HETATM','serial':serial,'name':f'C{i}','resname':'PQ','chain':'P','resseq':2000,
                          'x':x,'y':y,'z':z,'element':'C'})
            serial += 1
        return atoms, serial
    pq_bound, _ = make_pq(40, 10, 10, 22000)
    pq_free, _ = make_pq(41.5, 10, 10, 22000)
    bound_atoms = atoms_A + atoms_B + atoms_C + atoms_D + waters + co2 + pq_bound
    free_atoms = atoms_A + atoms_B + atoms_C + waters + co2 + pq_free
    write_pdb('/app/data/ndh1ms_ndhv_bound.pdb', bound_atoms)
    write_pdb('/app/data/ndh1ms_ndhv_free.pdb', free_atoms)

if __name__ == '__main__':
    main()
