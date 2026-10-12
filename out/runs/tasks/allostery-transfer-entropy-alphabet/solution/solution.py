import os, json, math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

AA3to1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}
HYDROPHOBIC = set("AVILMFWY")

def parse_pdb(path):
    coords = []
    seq = []
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x,y,z])
                aa3 = line[17:20].strip()
                seq.append(AA3to1.get(aa3, 'X'))
    return np.array(coords), "".join(seq)

def compute_contacts(coords, cutoff=8.0, min_seq_sep=3):
    n = len(coords)
    contacts = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if abs(i-j) >= min_seq_sep:
                d = np.linalg.norm(coords[i] - coords[j])
                if d <= cutoff:
                    contacts[i].append(j)
                    contacts[j].append(i)
    return contacts

def compute_gnm(coords, contacts):
    n = len(coords)
    Gamma = np.zeros((n, n))
    for i in range(n):
        Gamma[i,i] = len(contacts[i])
        for j in contacts[i]:
            Gamma[i,j] = -1.0
    C = np.linalg.pinv(Gamma, rcond=1e-10)
    source = np.zeros(n)
    sink = np.zeros(n)
    net_te = np.zeros(n)
    for i in range(n):
        for j in contacts[i]:
            if i < j:
                rho = 1.0 / np.sqrt(Gamma[i,i] * Gamma[j,j])
                denom = C[i,i] + C[j,j] + 1e-8
                te_ij = rho * (C[i,i] - C[j,j]) / denom
                te_ji = rho * (C[j,j] - C[i,i]) / denom
                source[i] += te_ij
                sink[i] += te_ji
                source[j] += te_ji
                sink[j] += te_ij
                net_te[i] += te_ij - te_ji
                net_te[j] += te_ji - te_ij
    return source, sink, net_te

def get_word(i, contacts, coords, seq, cutoff=8.0, min_seq_sep=3):
    if len(contacts[i]) < 3:
        return None
    dists = [(j, np.linalg.norm(coords[i]-coords[j])) for j in contacts[i]]
    dists.sort(key=lambda x: (x[1], x[0]))
    top3 = [j for j, _ in dists[:3]]
    residues = [i] + top3
    residues.sort()
    aa_seq = "".join(seq[r] for r in residues)
    bits = []
    for a in range(4):
        for b in range(a+1, 4):
            ri, rj = residues[a], residues[b]
            if abs(ri-rj) >= min_seq_sep and np.linalg.norm(coords[ri]-coords[rj]) <= cutoff:
                bits.append('1')
            else:
                bits.append('0')
    return f"{aa_seq}|{''.join(bits)}"

def kabsch_rmsd(apo_coords, holo_coords):
    apo_cent = apo_coords.mean(axis=0)
    holo_cent = holo_coords.mean(axis=0)
    apo_c = apo_coords - apo_cent
    holo_c = holo_coords - holo_cent
    H = apo_c.T @ holo_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    holo_rot = (R @ holo_c.T).T + apo_cent
    rmsd = np.linalg.norm(holo_rot - apo_coords, axis=1)
    return rmsd

def main():
    background_dir = '/app/data/background'
    target_dir = '/app/data/targets'
    out_dir = '/app/results'
    os.makedirs(out_dir, exist_ok=True)
    alphabet = {}
    for fname in sorted(os.listdir(background_dir)):
        if not fname.endswith('.pdb'):
            continue
        path = os.path.join(background_dir, fname)
        coords, seq = parse_pdb(path)
        contacts = compute_contacts(coords)
        _, _, net_te = compute_gnm(coords, contacts)
        for i in range(len(coords)):
            w = get_word(i, contacts, coords, seq)
            if w is not None:
                alphabet.setdefault(w, []).append(net_te[i])
    rows = []
    for fname in sorted(os.listdir(target_dir)):
        if not fname.endswith('_apo.pdb'):
            continue
        protein = fname.replace('_apo.pdb', '')
        path = os.path.join(target_dir, fname)
        coords, seq = parse_pdb(path)
        contacts = compute_contacts(coords)
        source, sink, net_te = compute_gnm(coords, contacts)
        for i in range(len(coords)):
            w = get_word(i, contacts, coords, seq)
            if w is not None and w in alphabet and len(alphabet[w]) >= 2:
                sw = np.std(alphabet[w], ddof=0)
            else:
                sw = 0.0
            rows.append({
                'protein': protein,
                'residue_index': i+1,
                'residue_name': seq[i],
                'net_te': net_te[i],
                'source': source[i],
                'sink': sink[i],
                'switch': sw
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'allosteric_tracks.tsv'), sep='\t', index=False, float_format='%.6f')
    source_rhos = []
    sink_rhos = []
    for protein in sorted(df['protein'].unique()):
        apo_path = os.path.join(target_dir, f"{protein}_apo.pdb")
        holo_path = os.path.join(target_dir, f"{protein}_holo.pdb")
        apo_coords, _ = parse_pdb(apo_path)
        holo_coords, _ = parse_pdb(holo_path)
        rmsd = kabsch_rmsd(apo_coords, holo_coords)
        sub = df[df['protein'] == protein].sort_values('residue_index')
        rho_source, _ = spearmanr(sub['source'], rmsd)
        rho_sink, _ = spearmanr(sub['sink'], rmsd)
        source_rhos.append(rho_source)
        sink_rhos.append(rho_sink)
    metrics = {
        'source_rmsd_mean_rho': float(np.mean(source_rhos)),
        'source_rmsd_positive_count': int(np.sum(np.array(source_rhos) > 0)),
        'sink_rmsd_mean_rho': float(np.mean(sink_rhos)),
        'sink_rmsd_negative_count': int(np.sum(np.array(sink_rhos) < 0))
    }
    with open(os.path.join(out_dir, 'validation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    vocab_rows = []
    for w, vals in alphabet.items():
        if len(vals) >= 5:
            std = np.std(vals, ddof=0)
            seq_part = w.split('|')[0]
            hyd = sum(1 for aa in seq_part if aa in HYDROPHOBIC) / 4.0
            vocab_rows.append({'word': w, 'std_net_te': std, 'count': len(vals), 'hydrophobic_fraction': hyd})
    vdf = pd.DataFrame(vocab_rows)
    vdf = vdf.sort_values('std_net_te', ascending=False).head(20)
    vdf.to_csv(os.path.join(out_dir, 'switch_vocabulary.tsv'), sep='\t', index=False, float_format='%.6f')

if __name__ == '__main__':
    main()