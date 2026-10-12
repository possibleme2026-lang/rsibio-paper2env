#!/usr/bin/env python3
"""Deterministic builder for the MAXWELL calibration task inputs.

Emits:
  data/mpnn/<pid>.csv          ProteinMPNN-style L x 20 logit matrix per protein
  data/sequences.fasta         wild-type sequences
  data/train_ddg.csv           labelled training mutations (protein_id, mutation, ddg)
  data/test_mutations.csv      unlabelled test mutations (protein_id, mutation)
  data/landscape_proteins.txt  proteins for which a full landscape is required

Everything derives from fixed seeds, so re-running reproduces byte-identical data.
NOTE: this script is executed inside the image build and then deleted, so the agent
cannot read the ground-truth generative model at solve time.
"""
import os

import numpy as np
import pandas as pd

AA = list("ACDEFGHIKLMNPQRSTVWY")

PROTEINS = [
    ("P001", 128), ("P002", 145), ("P003", 162), ("P004", 178),
    ("P005", 191), ("P006", 205), ("P007", 134), ("P008", 156),
    ("P009", 230), ("P010", 168), ("P011", 142), ("P012", 199),
]
LANDSCAPE_PROTEINS = ["P009", "P001", "P004"]

MASTER_SEED = 20260729
SPLIT_SEED = 13579
AA_SEED = 777
TRAIN_NOISE_SEED = 4242
ALPHA = 0.8
DELTA_SCALE = 1.0
NOISE_SIGMA = 1.0
N_TRAIN = 250
N_TEST = 70


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


def main():
    out = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(out, "mpnn"), exist_ok=True)
    delta = aa_delta()
    noise = np.random.default_rng(TRAIN_NOISE_SEED)

    fasta, train_rows, test_rows = [], [], []
    for pid, L in PROTEINS:
        logits = make_logits(pid, L)
        wt = wildtype_index(logits)
        fasta.append(">%s\n%s\n" % (pid, "".join(AA[j] for j in wt)))

        df = pd.DataFrame(logits, columns=AA)
        df.insert(0, "wt", [AA[j] for j in wt])
        df.insert(0, "position", np.arange(1, L + 1))
        df.to_csv(os.path.join(out, "mpnn", "%s.csv" % pid), index=False, float_format="%.5f")

        test_pairs, train_pairs = split_mutations(pid, L, wt)
        for i, m in test_pairs:
            i, m = int(i), int(m)
            test_rows.append((pid, "%s%d%s" % (AA[wt[i]], i + 1, AA[m])))
        for i, m in train_pairs:
            i, m = int(i), int(m)
            y = (ALPHA * (logits[i, m] - logits[i, wt[i]])
                 + delta[m] - delta[wt[i]]
                 + noise.normal(0.0, NOISE_SIGMA))
            train_rows.append((pid, "%s%d%s" % (AA[wt[i]], i + 1, AA[m]), round(float(y), 4)))

    with open(os.path.join(out, "sequences.fasta"), "w") as fh:
        fh.writelines(fasta)
    pd.DataFrame(train_rows, columns=["protein_id", "mutation", "ddg"]).to_csv(
        os.path.join(out, "train_ddg.csv"), index=False)
    pd.DataFrame(test_rows, columns=["protein_id", "mutation"]).to_csv(
        os.path.join(out, "test_mutations.csv"), index=False)
    with open(os.path.join(out, "landscape_proteins.txt"), "w") as fh:
        fh.write("\n".join(LANDSCAPE_PROTEINS) + "\n")

    print("wrote inputs for %d proteins (%d train, %d test mutations) to %s"
          % (len(PROTEINS), len(train_rows), len(test_rows), out))


if __name__ == "__main__":
    main()
