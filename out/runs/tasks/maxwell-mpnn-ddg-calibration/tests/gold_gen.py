#!/usr/bin/env python3
"""Hidden gold generator for the MAXWELL ProteinMPNN-calibration task.

Regenerates, deterministically, the held-out experimental ddG values that the agent
must predict. The noise seed used here (HIDDEN_TEST_NOISE_SEED) is never exposed in
the task environment, so the labels can neither be read nor recomputed by the agent.

The logit matrices, wild-type sequences and the train/test split are reproduced with
the same public seeds that environment/data/gen_inputs.py uses, so gold and inputs are
exactly consistent.
"""
import numpy as np

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_IDX = {a: i for i, a in enumerate(AA)}

PROTEINS = [
    ("P001", 128), ("P002", 145), ("P003", 162), ("P004", 178),
    ("P005", 191), ("P006", 205), ("P007", 134), ("P008", 156),
    ("P009", 230), ("P010", 168), ("P011", 142), ("P012", 199),
]
LANDSCAPE_PROTEINS = ["P009", "P001", "P004"]

MASTER_SEED = 20260729
SPLIT_SEED = 13579
AA_SEED = 777
ALPHA = 0.8
DELTA_SCALE = 1.0
NOISE_SIGMA = 1.0
N_TRAIN = 250
N_TEST = 70

# never shipped to the agent
HIDDEN_TEST_NOISE_SEED = 90210773


def make_logits(pid, L):
    rng = np.random.default_rng(MASTER_SEED * 1000 + int(pid[1:]))
    return rng.normal(0.0, 1.0, size=(L, 20))


def wildtype_index(logits):
    return np.argmax(logits, axis=1)


def aa_delta():
    rng = np.random.default_rng(AA_SEED)
    return rng.normal(0.0, DELTA_SCALE, size=20)


def split_mutations(pid, L, wt_idx):
    rng = np.random.default_rng(SPLIT_SEED * 1000 + int(pid[1:]))
    pairs = np.array([(i, m) for i in range(L) for m in range(20) if m != wt_idx[i]])
    pairs = pairs[rng.permutation(len(pairs))]
    return pairs[:N_TEST], pairs[N_TEST:N_TEST + N_TRAIN]


def wildtype_sequences():
    seqs = {}
    for pid, L in PROTEINS:
        wt = wildtype_index(make_logits(pid, L))
        seqs[pid] = "".join(AA[j] for j in wt)
    return seqs


def gold_test_labels():
    """Return [(protein_id, mutation, ddg), ...] in a fixed deterministic order."""
    delta = aa_delta()
    rng = np.random.default_rng(HIDDEN_TEST_NOISE_SEED)
    out = []
    for pid, L in PROTEINS:
        logits = make_logits(pid, L)
        wt = wildtype_index(logits)
        test_pairs, _ = split_mutations(pid, L, wt)
        for i, m in test_pairs:
            i, m = int(i), int(m)
            y = (ALPHA * (logits[i, m] - logits[i, wt[i]])
                 + delta[m] - delta[wt[i]]
                 + rng.normal(0.0, NOISE_SIGMA))
            out.append((pid, "%s%d%s" % (AA[wt[i]], i + 1, AA[m]), float(y)))
    return out


if __name__ == "__main__":
    rows = gold_test_labels()
    print(len(rows))
    for r in rows[:5]:
        print(r)
