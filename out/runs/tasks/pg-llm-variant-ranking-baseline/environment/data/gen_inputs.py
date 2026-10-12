#!/usr/bin/env python3
"""Deterministic generator for the PG-LLM variant-ranking benchmark inputs.

Running this script recreates every file under data/ (manifest, wild-type
sequences, multiple sequence alignments and variant tables) from the fixed seed
below. It is executed once at image build time, so the shipped task contains
realistic-size assay data (~95 KB across six assays) without any pre-baked gold.

The measured dms_score column is synthesised from a hidden per-assay effect
model (random substitution cost matrix + per-position sensitivity + Gaussian
noise), NOT from the MSA-CONS-BLOSUM rule the agent must implement.
"""
import csv
import math
import os
import random

SEED = 20260727
ROOT = os.path.dirname(os.path.abspath(__file__))
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)
N_MSA = 64

ASSAYS = [
    dict(assay_id="TP53_HUMAN_Giacomelli_2018", uniprot="P04637",
         protein="Cellular tumor antigen p53", length=250, n_variants=170,
         split="proteinGym"),
    dict(assay_id="EGFR_HUMAN_Brenan_2016", uniprot="P00533",
         protein="Epidermal growth factor receptor", length=330, n_variants=190,
         split="proteinGym"),
    dict(assay_id="LYSC_CHICK_Peisajovich_2010", uniprot="P00698",
         protein="Lysozyme C", length=129, n_variants=150,
         split="proteinGym"),
    dict(assay_id="CAH2_HUMAN_Chiasson_2020", uniprot="P00918",
         protein="Carbonic anhydrase 2", length=260, n_variants=175,
         split="proteinGym"),
    dict(assay_id="SPIKE_SARS2_Starr_2020", uniprot="P0DTC2",
         protein="Spike glycoprotein (receptor binding domain)", length=300,
         n_variants=165, split="heldout"),
    dict(assay_id="GFP_AEQVI_Sarkisyan_2016", uniprot="P42212",
         protein="Green fluorescent protein", length=238, n_variants=160,
         split="heldout"),
]


def wrap(seq, width=60):
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def weighted_choice(rng, probs):
    r = rng.random()
    acc = 0.0
    for key, p in probs.items():
        acc += p
        if r <= acc:
            return key
    return next(iter(probs))


def build_assay(spec, rng):
    length = spec["length"]
    wt_chars = []
    columns = []
    for _ in range(length):
        w = rng.choice(AA)
        if rng.random() < 0.70:
            cons = w
        else:
            cons = rng.choice([x for x in AA if x != w])
        keep = rng.uniform(0.40, 0.95)
        others = rng.sample([x for x in AA if x != cons], k=rng.randint(3, 6))
        if w != cons and w not in others:
            others[0] = w
        probs = {cons: keep}
        share = (1.0 - keep) / len(others)
        for a in others:
            probs[a] = share
        columns.append(probs)
        wt_chars.append(w)
    wt = "".join(wt_chars)

    rows = [wt]
    for _ in range(N_MSA - 1):
        chars = []
        for probs in columns:
            u = rng.random()
            if u < 0.020:
                chars.append("-")
            elif u < 0.026:
                chars.append("X")
            else:
                chars.append(weighted_choice(rng, probs))
        rows.append("".join(chars))

    counts = [dict() for _ in range(length)]
    for seq in rows:
        for p, ch in enumerate(seq):
            if ch in AA_SET:
                counts[p][ch] = counts[p].get(ch, 0) + 1

    def log_ratio(p, a):
        col = counts[p]
        total = sum(col.values())
        qa = (col.get(a, 0) + 1.0) / (total + 20.0)
        qw = (col.get(wt[p], 0) + 1.0) / (total + 20.0)
        return math.log2(qa / qw)

    # hidden effect model used ONLY to synthesise the measured DMS values
    cost = {}
    for i, a in enumerate(AA):
        for j, b in enumerate(AA):
            if i == j:
                cost[(a, b)] = 0.0
            elif i < j:
                v = rng.uniform(0.2, 2.0)
                cost[(a, b)] = v
                cost[(b, a)] = v
    kappa = [max(0.2, rng.gauss(1.0, 0.25)) for _ in range(length)]
    imp = [max(0.2, rng.gauss(1.0, 0.35)) for _ in range(length)]

    variants = []
    seen = set()
    while len(variants) < spec["n_variants"]:
        k = rng.choices([1, 2, 3], weights=[0.70, 0.22, 0.08])[0]
        muts = []
        for p in rng.sample(range(length), k):
            w = wt[p]
            a = rng.choice([x for x in AA if x != w])
            muts.append("%s%d%s" % (w, p + 1, a))
        key = ":".join(muts)
        if key in seen:
            continue
        seen.add(key)
        variants.append(key)

    raw = {}
    for key in variants:
        effect = 0.0
        parts = key.split(":")
        for m in parts:
            w, p, a = m[0], int(m[1:-1]) - 1, m[-1]
            effect += kappa[p] * log_ratio(p, a)
            effect -= 0.35 * imp[p] * cost[(w, a)]
        effect += rng.gauss(0.0, 0.35 * len(parts))
        raw[key] = effect

    vals = list(raw.values())
    mu = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1))
    if sd <= 1e-9:
        sd = 1.0

    records = [("WT", "WT", 0.0)]
    for key in variants:
        records.append((key, key, round((raw[key] - mu) / sd, 4)))
    return wt, rows, records


def main():
    rng = random.Random(SEED)
    manifest_rows = []
    for spec in ASSAYS:
        wt, rows, records = build_assay(spec, rng)
        adir = os.path.join(ROOT, "assays", spec["assay_id"])
        os.makedirs(adir, exist_ok=True)

        header = ">sp|%s|%s %s" % (spec["uniprot"], spec["assay_id"], spec["protein"])
        with open(os.path.join(adir, "wt.fasta"), "w") as fh:
            fh.write("%s\n%s\n" % (header, wrap(wt)))

        with open(os.path.join(adir, "msa.fasta"), "w") as fh:
            fh.write("%s\n%s\n" % (header, wrap(rows[0])))
            for i, seq in enumerate(rows[1:], start=1):
                fh.write(">HOMOLOG_%03d\n%s\n" % (i, wrap(seq)))

        with open(os.path.join(adir, "variants.csv"), "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["variant_id", "mutations", "dms_score"])
            for vid, muts, dms in records:
                writer.writerow([vid, muts, "%.4f" % dms])

        manifest_rows.append([spec["assay_id"], spec["uniprot"], spec["protein"],
                              len(wt), len(records), spec["split"]])

    with open(os.path.join(ROOT, "assay_manifest.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["assay_id", "uniprot_id", "protein_name",
                         "sequence_length", "n_variants", "split"])
        writer.writerows(manifest_rows)

    print("generated %d assays in %s" % (len(ASSAYS), ROOT))


if __name__ == "__main__":
    main()
