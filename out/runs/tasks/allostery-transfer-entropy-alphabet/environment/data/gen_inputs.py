import os, random, math
import numpy as np

AA = list("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AVILMFWY")

def generate_sequence(length, rng):
    seq = []
    for _ in range(length):
        if rng.random() < 0.6:
            seq.append(rng.choice(list(HYDROPHOBIC)))
        else:
            seq.append(rng.choice([a for a in AA if a not in HYDROPHOBIC]))
    return "".join(seq)

def generate_coords(length, rng):
    coords = np.zeros((length, 3))
    for i in range(1, length):
        step = rng.normal(size=3)
        step = step / np.linalg.norm(step) * 3.8
        coords[i] = coords[i-1] + step
    for _ in range(200):
        centroid = coords.mean(axis=0)
        for i in range(length):
            coords[i] += 0.02 * (centroid - coords[i])
        for i in range(length):
            for j in range(i+1, length):
                d = np.linalg.norm(coords[i] - coords[j])
                if d < 4.0:
                    f = 0.5 * (4.0 - d) / (d + 1e-8)
                    coords[i] -= f * (coords[i] - coords[j])
                    coords[j] += f * (coords[i] - coords[j])
    return coords

def write_pdb(coords, seq, filename):
    nl = chr(10)
    with open(filename, 'w') as f:
        for i, (c, aa) in enumerate(zip(coords, seq)):
            f.write(f"ATOM  {i+1:5d}  N   {aa:3s} A{i+1:4d}    {c[0]-0.5:8.3f}{c[1]-0.5:8.3f}{c[2]-0.5:8.3f}  1.00  0.00           N" + nl)
            f.write(f"ATOM  {i+1:5d}  CA  {aa:3s} A{i+1:4d}    {c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00           C" + nl)
            f.write(f"ATOM  {i+1:5d}  C   {aa:3s} A{i+1:4d}    {c[0]+0.5:8.3f}{c[1]+0.5:8.3f}{c[2]+0.5:8.3f}  1.00  0.00           C" + nl)
            f.write(f"ATOM  {i+1:5d}  O   {aa:3s} A{i+1:4d}    {c[0]+0.8:8.3f}{c[1]+0.8:8.3f}{c[2]+0.8:8.3f}  1.00  0.00           O" + nl)
            f.write("TER" + nl)
        f.write("END" + nl)

def compute_gnm(coords, cutoff=8.0, min_seq_sep=3):
    n = len(coords)
    contacts = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if abs(i-j) >= min_seq_sep:
                d = np.linalg.norm(coords[i] - coords[j])
                if d <= cutoff:
                    contacts[i].append(j)
                    contacts[j].append(i)
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
                rho = 1.0 / math.sqrt(Gamma[i,i] * Gamma[j,j])
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

def main():
    class _Rng(np.random.Generator):
        """numpy Generator + the stdlib .randint spelling the rest of
        this file uses (preserves seed 42 and the intended ranges)."""
        def randint(self, a, b):
            return int(self.integers(a, b + 1))
    rng = _Rng(np.random.PCG64(42))
    np.random.seed(42)
    os.makedirs("background", exist_ok=True)
    os.makedirs("targets", exist_ok=True)
    bg_seqs = []
    for k in range(40):
        L = rng.randint(80, 160)
        seq = generate_sequence(L, rng)
        coords = generate_coords(L, rng)
        write_pdb(coords, seq, f"background/bg_{k:02d}.pdb")
        bg_seqs.append((f"bg_{k:02d}", seq))
    target_seqs = []
    for k in range(9):
        L = rng.randint(100, 150)
        seq = generate_sequence(L, rng)
        coords = generate_coords(L, rng)
        write_pdb(coords, seq, f"targets/target_{k:02d}_apo.pdb")
        source, sink, net_te = compute_gnm(coords)
        nt = net_te - net_te.mean()
        nt = nt / (np.std(nt) + 1e-8)
        holo_coords = coords.copy()
        for i in range(L):
            disp = nt[i] * 1.5
            v = np.random.normal(size=3)
            v = v / np.linalg.norm(v)
            holo_coords[i] += v * disp
        write_pdb(holo_coords, seq, f"targets/target_{k:02d}_holo.pdb")
        target_seqs.append((f"target_{k:02d}", seq))
    nl = chr(10)
    with open("sequences.fasta", "w") as f:
        for name, seq in bg_seqs + target_seqs:
            f.write(">" + name + nl + seq + nl)

if __name__ == "__main__":
    main()