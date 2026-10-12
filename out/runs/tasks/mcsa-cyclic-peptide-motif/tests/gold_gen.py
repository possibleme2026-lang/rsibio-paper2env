import random

L = 18
M = 8
N = 1000
SEED = 20260728
CONSENSUS = "LWFPDGYA"
AA = "ACDEFGHIKLMNPQRSTVWY"

def generate_gold():
    random.seed(SEED)
    gold_rotations = {}
    canonical_sequences = {}
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
        gold_rotations[pid] = t
        canonical_sequences[pid] = canonical
    return {
        "rotations": gold_rotations,
        "canonical_sequences": canonical_sequences,
        "consensus": CONSENSUS,
    }
