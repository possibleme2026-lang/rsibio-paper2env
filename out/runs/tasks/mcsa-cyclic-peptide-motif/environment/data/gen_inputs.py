import csv, random

L = 18
M = 8
N = 1000
SEED = 20260728
CONSENSUS = "LWFPDGYA"
AA = "ACDEFGHIKLMNPQRSTVWY"

random.seed(SEED)
with open('/app/data/cyclic_peptides.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['peptide_id', 'sequence', 'cyclic'])
    for i in range(1, N+1):
        pid = f"PEP{i:04d}"
        motif = []
        for j in range(M):
            if random.random() < 0.90:
                motif.append(CONSENSUS[j])
            else:
                other = [aa for aa in AA if aa != CONSENSUS[j]]
                motif.append(random.choice(other))
        flank = [random.choice(AA) for _ in range(L-M)]
        canonical = ''.join(motif + flank)
        t = random.randint(0, L-1)
        input_seq = canonical[L-t:] + canonical[:L-t] if t > 0 else canonical
        writer.writerow([pid, input_seq, 'true'])
