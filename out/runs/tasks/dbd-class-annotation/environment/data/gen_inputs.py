#!/usr/bin/env python3
# Deterministic generator for the DBD-class annotation task inputs.
# Runs only in the builder stage; the generation core is mirrored in tests/gold_gen.py.
import argparse
import os
import random

BASES = ("A", "C", "G", "T")
CLASSES = ("bZIP", "bHLH", "Homeodomain", "POU", "Paired", "Forkhead", "T-box", "ETS",
           "Rel", "MADS", "HMG", "C2H2-ZF", "Nuclear-Receptor", "GATA-ZF", "DM-ZF",
           "Tryptophan-Cluster", "Cold-Shock", "Runt", "SAND", "Winged-Helix",
           "HTH-LysR", "STAT", "IRF")
BG = {"A": 0.295, "C": 0.205, "G": 0.205, "T": 0.295}
SEED_MODELS = 20241220
SEED_TEST = 771103
SEED_CAL = 550219
N_TEST = 520
N_CAL = 200
SEQ_LEN = 120
TAB = chr(9)
NL = chr(10)


def draw_base(rng, probs):
    r = rng.random()
    acc = 0.0
    for b in BASES:
        acc += probs[b]
        if r < acc:
            return b
    return BASES[-1]


def make_models(seed):
    rng = random.Random(seed)
    models = {}
    for name in CLASSES:
        width = 15 + rng.randrange(5)
        rows = []
        for _ in range(width):
            lam = 0.89 + 0.07 * rng.random()
            cons = BASES[rng.randrange(4)]
            probs = {}
            for b in BASES:
                probs[b] = (1.0 - lam) * 0.25 + (lam if b == cons else 0.0)
            probs = dict((b, probs[b] * (0.97 + 0.06 * rng.random())) for b in BASES)
            tot = sum(probs.values())
            rows.append(dict((b, probs[b] / tot) for b in BASES))
        models[name] = rows
    return models


def gc_background(gc):
    p = dict(BG)
    p["C"] = BG["C"] * gc / (BG["C"] + BG["G"])
    p["G"] = BG["G"] * gc / (BG["C"] + BG["G"])
    at = 1.0 - gc
    p["A"] = BG["A"] * at / (BG["A"] + BG["T"])
    p["T"] = BG["T"] * at / (BG["A"] + BG["T"])
    return p


def paint(seq, models, name, offset, rho, rng):
    for k, row in enumerate(models[name]):
        b = draw_base(rng, row)
        if rng.random() < rho:
            i = BASES.index(b)
            b = BASES[(i + 1 + rng.randrange(3)) % 4]
        seq[offset + k] = b


def generate_sequences(seed, n_seq, models, length):
    rng = random.Random(seed)
    n_motif = int(round(n_seq * 0.75))
    has = [True] * n_motif + [False] * (n_seq - n_motif)
    rng.shuffle(has)
    prim = []
    while len(prim) < n_motif:
        pool = list(CLASSES)
        rng.shuffle(pool)
        prim.extend(pool)
    prim = prim[:n_motif]
    rng.shuffle(prim)
    out = []
    labels = []
    k = 0
    for i, flag in enumerate(has):
        gc = 0.41 + rng.uniform(-0.03, 0.03)
        probs = gc_background(gc)
        seq = [draw_base(rng, probs) for _ in range(length)]
        if rng.random() < 0.20:
            unit = "".join(BASES[rng.randrange(4)] for _ in range(1 + rng.randrange(3)))
            patch = (unit * (6 + rng.randrange(10)))[:length]
            st = rng.randrange(0, length - len(patch) + 1)
            for j, ch in enumerate(patch):
                seq[st + j] = ch
        if flag:
            name = prim[k]
            k += 1
            w = len(models[name])
            st = rng.randrange(0, length - w + 1)
            paint(seq, models, name, st, rng.uniform(0.0, 0.03), rng)
            for _ in range((0, 0, 1, 1, 2)[rng.randrange(5)]):
                dname = CLASSES[rng.randrange(len(CLASSES))]
                if dname == name:
                    continue
                dw = len(models[dname])
                dst = None
                for _try in range(25):
                    cand = rng.randrange(0, length - dw + 1)
                    if cand + dw <= st or st + w <= cand:
                        dst = cand
                        break
                if dst is not None:
                    paint(seq, models, dname, dst, rng.uniform(0.05, 0.20), rng)
            labels.append(name)
        else:
            labels.append("None")
        out.append(("seq_%06d" % (i + 1), "".join(seq)))
    return out, labels


def write_all(outdir):
    os.makedirs(outdir, exist_ok=True)
    models = make_models(SEED_MODELS)
    with open(os.path.join(outdir, "dbd_classes.txt"), "w") as fh:
        fh.write("".join(name + NL for name in CLASSES))
    with open(os.path.join(outdir, "dbd_class_models.tsv"), "w") as fh:
        fh.write(TAB.join(["class_name", "position"] + list(BASES)) + NL)
        for name in CLASSES:
            for j, row in enumerate(models[name]):
                fh.write(TAB.join([name, str(j)] + ["%.6f" % row[b] for b in BASES]) + NL)
    with open(os.path.join(outdir, "background.tsv"), "w") as fh:
        fh.write(TAB.join(["base", "frequency"]) + NL)
        for b in BASES:
            fh.write(TAB.join([b, "%.6f" % BG[b]]) + NL)
    for fname, seed, n in (("test_sequences.fa", SEED_TEST, N_TEST),
                           ("calibration_sequences.fa", SEED_CAL, N_CAL)):
        seqs, labels = generate_sequences(seed, n, models, SEQ_LEN)
        with open(os.path.join(outdir, fname), "w") as fh:
            for sid, s in seqs:
                fh.write(">" + sid + NL)
                for i in range(0, len(s), 60):
                    fh.write(s[i:i + 60] + NL)
        if fname.startswith("calibration"):
            with open(os.path.join(outdir, "calibration_labels.tsv"), "w") as fh:
                fh.write(TAB.join(["seq_id", "dbd_class"]) + NL)
                for (sid, _), lab in zip(seqs, labels):
                    fh.write(TAB.join([sid, lab]) + NL)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/app/data")
    args = ap.parse_args()
    write_all(args.outdir)
