import os, sys, json, math
import numpy as np

def parse_pdb(path):
    structure = {}
    with open(path) as f:
        for line in f:
            if line.startswith('ATOM'):
                try:
                    atom_name = line[12:16].strip()
                    resname = line[17:20].strip()
                    chain = line[21].strip()
                    resnum = int(line[22:26].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                if chain not in structure:
                    structure[chain] = {}
                if resnum not in structure[chain]:
                    structure[chain][resnum] = {'resname': resname, 'atoms': {}}
                structure[chain][resnum]['atoms'][atom_name] = np.array([x,y,z])
    return structure

def get_loop_range(input_structure):
    h_res = sorted(input_structure.get('H', {}).keys())
    if not h_res:
        return None
    for i in range(len(h_res)-1):
        if h_res[i+1] - h_res[i] > 1:
            start = h_res[i] + 1
            end = h_res[i+1] - 1
            L = end - start + 1
            return start, end, L
    return None

def check_framework(input_structure, output_structure):
    loop_info = get_loop_range(input_structure)
    if not loop_info:
        return False, "No gap found in input chain H"
    start, end, L = loop_info
    for chain in ['L', 'A']:
        if chain not in output_structure:
            return False, f"Missing chain {chain} in output"
        for resnum, resdata in input_structure[chain].items():
            if resnum not in output_structure[chain]:
                return False, f"Missing residue {chain}{resnum} in output"
            for atom_name, coord in resdata['atoms'].items():
                if atom_name not in output_structure[chain][resnum]['atoms']:
                    return False, f"Missing atom {chain}{resnum} {atom_name}"
                out_coord = output_structure[chain][resnum]['atoms'][atom_name]
                if np.linalg.norm(out_coord - coord) > 0.01:
                    return False, f"Atom {chain}{resnum} {atom_name} moved"
    for resnum, resdata in input_structure['H'].items():
        if start <= resnum <= end:
            continue
        if resnum not in output_structure.get('H', {}):
            return False, f"Missing H{resnum} in output"
        for atom_name, coord in resdata['atoms'].items():
            if atom_name not in output_structure['H'][resnum]['atoms']:
                return False, f"Missing atom H{resnum} {atom_name}"
            out_coord = output_structure['H'][resnum]['atoms'][atom_name]
            if np.linalg.norm(out_coord - coord) > 0.01:
                return False, f"Atom H{resnum} {atom_name} moved"
    return True, "Framework OK"

def check_loop_presence(output_structure, start, end):
    if 'H' not in output_structure:
        return False, "No chain H in output"
    for resnum in range(start, end+1):
        if resnum not in output_structure['H']:
            return False, f"Missing loop residue H{resnum}"
        for atom in ['N','CA','C','O','CB']:
            if atom not in output_structure['H'][resnum]['atoms']:
                return False, f"Missing atom {atom} in H{resnum}"
    return True, "Loop present"

def check_sequence(output_structure, start, end, fragment_sequences):
    aa_map = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
    seq = []
    for resnum in range(start, end+1):
        resname = output_structure['H'][resnum]['resname']
        if resname not in aa_map:
            return False, f"Unknown residue {resname}"
        seq.append(aa_map[resname])
    seq_str = ''.join(seq)
    if seq_str not in fragment_sequences:
        return False, f"Sequence {seq_str} not in fragment library"
    return True, f"Sequence {seq_str} OK"

def check_closure(output_structure, input_structure, start, end):
    prev_res = start - 1
    next_res = end + 1
    if prev_res not in input_structure['H'] or next_res not in input_structure['H']:
        return False, "Anchor residues not in input"
    anchor_C = input_structure['H'][prev_res]['atoms'].get('C')
    anchor_N = input_structure['H'][next_res]['atoms'].get('N')
    if anchor_C is None or anchor_N is None:
        return False, "Anchor atoms missing"
    first_N = output_structure['H'][start]['atoms'].get('N')
    last_C = output_structure['H'][end]['atoms'].get('C')
    if first_N is None or last_C is None:
        return False, "Loop terminal atoms missing"
    d1 = np.linalg.norm(first_N - anchor_C)
    d2 = np.linalg.norm(last_C - anchor_N)
    if d1 > 1.5 or d2 > 1.5:
        return False, f"Closure distances: {d1:.2f}, {d2:.2f}"
    return True, f"Closure OK ({d1:.2f}, {d2:.2f})"

def check_geometry(output_structure, start, end):
    ideal = {'N-CA': 1.46, 'CA-C': 1.52, 'C-N': 1.33}
    tol = 0.1
    for resnum in range(start, end+1):
        atoms = output_structure['H'][resnum]['atoms']
        d = np.linalg.norm(atoms['N'] - atoms['CA'])
        if abs(d - ideal['N-CA']) > tol:
            return False, f"H{resnum} N-CA {d:.2f}"
        d = np.linalg.norm(atoms['CA'] - atoms['C'])
        if abs(d - ideal['CA-C']) > tol:
            return False, f"H{resnum} CA-C {d:.2f}"
        if resnum < end:
            next_atoms = output_structure['H'][resnum+1]['atoms']
            d = np.linalg.norm(atoms['C'] - next_atoms['N'])
            if abs(d - ideal['C-N']) > tol:
                return False, f"H{resnum} C-N {d:.2f}"
            d = np.linalg.norm(atoms['CA'] - next_atoms['CA'])
            if abs(d - 3.8) > 0.5:
                return False, f"H{resnum} CA-CA {d:.2f}"
    return True, "Geometry OK"

def check_clashes(output_structure, start, end):
    loop_atoms = []
    for resnum in range(start, end+1):
        for atom_name, coord in output_structure['H'][resnum]['atoms'].items():
            loop_atoms.append((resnum, atom_name, coord))
    for chain, res_dict in output_structure.items():
        for resnum, resdata in res_dict.items():
            if chain == 'H' and start <= resnum <= end:
                continue
            for atom_name, coord in resdata['atoms'].items():
                for l_resnum, l_atom_name, l_coord in loop_atoms:
                    if chain == 'H' and abs(l_resnum - resnum) <= 2:
                        continue
                    d = np.linalg.norm(l_coord - coord)
                    if d < 2.0:
                        return False, f"Clash between H{l_resnum} {l_atom_name} and {chain}{resnum} {atom_name}: {d:.2f}"
    return True, "No clashes"

def check_docking(output_structure, start, end):
    loop_atoms = []
    for resnum in range(start, end+1):
        for atom_name, coord in output_structure['H'][resnum]['atoms'].items():
            if atom_name in ['N','CA','C','O','CB']:
                loop_atoms.append(coord)
    antigen_atoms = []
    if 'A' in output_structure:
        for resnum, resdata in output_structure['A'].items():
            for atom_name, coord in resdata['atoms'].items():
                if atom_name in ['N','CA','C','O','CB']:
                    antigen_atoms.append(coord)
    contacts = 0
    for l in loop_atoms:
        for a in antigen_atoms:
            if np.linalg.norm(l - a) < 4.5:
                contacts += 1
    if contacts >= 5:
        return True, 1.0, f"Contacts {contacts}"
    elif contacts >= 3:
        return True, 0.5, f"Contacts {contacts}"
    else:
        return False, 0.0, f"Contacts {contacts}"

def main():
    input_dir = '/app/data/complexes'
    output_dir = '/app/output'
    frag_dir = '/app/data/fragments'
    fragment_sequences = set()
    if os.path.exists(frag_dir):
        for fname in os.listdir(frag_dir):
            if fname.endswith('.pdb'):
                struct = parse_pdb(os.path.join(frag_dir, fname))
                if 'A' in struct:
                    seq = []
                    aa_map = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
                    for resnum in sorted(struct['A'].keys()):
                        resname = struct['A'][resnum]['resname']
                        if resname in aa_map:
                            seq.append(aa_map[resname])
                    fragment_sequences.add(''.join(seq))
    complexes = sorted([f for f in os.listdir(input_dir) if f.endswith('.pdb')])
    total_score = 0.0
    max_score = 0.0
    report = []
    for comp_file in complexes:
        comp_id = comp_file.replace('.pdb', '')
        input_path = os.path.join(input_dir, comp_file)
        output_path = os.path.join(output_dir, f"{comp_id}_reconstructed.pdb")
        input_struct = parse_pdb(input_path)
        loop_info = get_loop_range(input_struct)
        if not loop_info:
            report.append(f"{comp_id}: No gap found")
            max_score += 7
            continue
        start, end, L = loop_info
        if not os.path.exists(output_path):
            report.append(f"{comp_id}: Output missing")
            max_score += 7
            continue
        output_struct = parse_pdb(output_path)
        comp_score = 0.0
        ok, msg = check_framework(input_struct, output_struct)
        report.append(f"{comp_id} framework: {ok} ({msg})")
        if ok: comp_score += 1.0
        ok, msg = check_loop_presence(output_struct, start, end)
        report.append(f"{comp_id} loop presence: {ok} ({msg})")
        if ok: comp_score += 1.0
        ok, msg = check_sequence(output_struct, start, end, fragment_sequences)
        report.append(f"{comp_id} sequence: {ok} ({msg})")
        if ok: comp_score += 1.0
        ok, msg = check_closure(output_struct, input_struct, start, end)
        report.append(f"{comp_id} closure: {ok} ({msg})")
        if ok: comp_score += 1.0
        ok, msg = check_geometry(output_struct, start, end)
        report.append(f"{comp_id} geometry: {ok} ({msg})")
        if ok: comp_score += 1.0
        ok, msg = check_clashes(output_struct, start, end)
        report.append(f"{comp_id} clashes: {ok} ({msg})")
        if ok: comp_score += 1.0
        ok, score_frac, msg = check_docking(output_struct, start, end)
        report.append(f"{comp_id} docking: {ok} ({msg})")
        comp_score += score_frac
        total_score += comp_score
        max_score += 7.0
        report.append(f"{comp_id} score: {comp_score}/7.0")
    final_score = total_score / max_score if max_score > 0 else 0.0
    report.append(f"Total score: {final_score:.4f}")
    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({"score": final_score, "reward": final_score}, f)
    with open('/logs/verifier/report.json', 'w') as f:
        json.dump({"report": report}, f)
    print('\n'.join(report))

if __name__ == '__main__':
    main()
