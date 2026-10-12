"""Deterministic generator for the fish morphology -> life-history task.

The identical file is used twice:
  * as environment/data/generate_data.py inside a throw-away Docker build
    stage that materialises /app/data of the final image, and
  * as tests/gold_gen.py, imported by the verifier to rebuild the hidden
    held-out life histories and the reference diagnostics.
A single source guarantees that the visible inputs and the hidden gold are
mutually consistent.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

SEED = 20251112
N_MORPH = 290
N_TREE = 288
N_TRAIN = 238
N_TEST = N_TREE - N_TRAIN
N_MISSING_CELLS = 12

TRAITS = ["body_elongation", "relative_eye_size", "oral_gape_position",
          "caudal_fin_ratio", "pectoral_fin_ratio", "mouth_width_ratio",
          "head_length_ratio", "body_depth_ratio"]
LATERAL = ["body_elongation", "relative_eye_size", "oral_gape_position"]
CLADES = ["Perciformes", "Cypriniformes", "Salmoniformes",
          "Gadiformes", "Clupeiformes", "Pleuronectiformes"]
WATER = ["pelagic", "demersal", "benthic", "reef"]

GENERA = ["Sebastes", "Gadus", "Salmo", "Sparus", "Lutjanus", "Perca", "Oncorhynchus", "Cyprinus",
          "Thunnus", "Sardina", "Scomber", "Morone", "Oreochromis", "Danio", "Cottus", "Gasterosteus",
          "Fundulus", "Poecilia", "Xiphophorus", "Melanogrammus", "Micropterus", "Esox", "Silurus", "Clarias",
          "Acanthurus", "Pomacentrus", "Ctenopharyngodon", "Hypophthalmichthys", "Rutilus", "Alosa", "Clupea", "Engraulis",
          "Merluccius", "Solea", "Pleuronectes", "Limanda", "Hippoglossus", "Scophthalmus", "Raja", "Squalus",
          "Carcharhinus", "Scyliorhinus", "Mugil", "Dicentrarchus", "Atherina", "Belone", "Syngnathus", "Hippocampus",
          "Lophius", "Zeus", "Trachurus", "Seriola", "Caranx", "Sciaenops", "Cynoscion", "Ariopsis",
          "Ictalurus", "Ameiurus", "Lepomis", "Pomoxis", "Notropis", "Coregonus", "Salvelinus", "Thymallus",
          "Hucho", "Stenodus", "Lota", "Barbus", "Carassius", "Tinca", "Abramis", "Vimba",
          "Leuciscus", "Phoxinus", "Gobio", "Cobitis", "Misgurnus", "Anguilla", "Conger", "Muraena"]

EPITHETS = ["morhua", "trutta", "salar", "aurata", "campechanus", "fluviatilis", "mykiss", "carpio",
            "thynnus", "pilchardus", "scombrus", "saxatilis", "niloticus", "rerio", "gobio", "aculeatus",
            "heteroclitus", "reticulata", "maculatus", "aeglefinus", "salmoides", "lucius", "glanis", "gariepinus",
            "triostegus", "chrysurus", "idella", "molitrix", "rutilus", "fallax", "harengus", "encrasicolus",
            "merluccius", "solea", "platessa", "limanda", "hippoglossus", "maximus", "clavata", "acanthias",
            "plumbeus", "canicula", "cephalus", "labrax", "boyeri", "belone", "acus", "guttulatus",
            "piscatorius", "faber", "trachurus", "dumerili", "caranx", "ocellatus", "regalis", "felis",
            "punctatus", "nebulosus", "macrochirus", "nigromaculatus", "cornutus", "lavaretus", "alpinus", "thymallus",
            "hucho", "leucichthys", "lotus", "barbus", "carassius", "tinca", "brama", "vimba",
            "phoxinus", "taenia", "fossilis", "anguilla", "conger", "helena", "ruber", "melanops",
            "fuscus", "elegans", "vulgaris", "minor", "major", "brevis", "longus", "gracilis",
            "pallidus", "obscurus"]

_A = np.array([[0.90, -0.10, 0.15],
               [-0.75, 0.05, 0.20],
               [0.70, 0.25, 0.10],
               [-0.25, 0.80, 0.10],
               [0.15, 0.70, -0.10],
               [0.10, 0.25, 0.80],
               [-0.30, 0.20, 0.65],
               [0.20, -0.20, 0.05]], float)

_B = np.array([[0.30, 0.10, -0.20],
               [0.20, 0.15, 0.10],
               [-0.15, 0.25, 0.20],
               [0.25, 0.35, -0.15],
               [-0.20, 0.30, 0.10],
               [0.15, -0.20, 0.35],
               [0.25, -0.10, 0.30],
               [0.20, 0.15, 0.25]], float)


def norm(name):
    return str(name).strip().replace(" ", "_")


def _make_names(rng):
    total = len(GENERA) * len(EPITHETS)
    picked = rng.choice(total, size=N_MORPH, replace=False)
    out = []
    for k in picked:
        k = int(k)
        out.append(GENERA[k // len(EPITHETS)] + "_" + EPITHETS[k % len(EPITHETS)])
    return out


def _simulate_tree(rng, tip_names):
    tips = [{"name": nm, "height": 0.0, "blen": 0.0,
             "left": None, "right": None, "parent": None} for nm in tip_names]
    lineages = list(tips)
    t = 0.0
    while len(lineages) > 1:
        k = len(lineages)
        t += float(rng.exponential(2.0 / (k * (k - 1))))
        pick = rng.choice(k, size=2, replace=False)
        a = lineages[int(pick[0])]
        b = lineages[int(pick[1])]
        a["blen"] = t - a["height"]
        b["blen"] = t - b["height"]
        node = {"name": None, "height": t, "blen": 0.0,
                "left": a, "right": b, "parent": None}
        a["parent"] = node
        b["parent"] = node
        for pos in sorted([int(pick[0]), int(pick[1])], reverse=True):
            lineages.pop(pos)
        lineages.append(node)
    return lineages[0], tips


def _newick(node):
    if node["left"] is None:
        return "%s:%.6f" % (node["name"], node["blen"])
    return "(%s,%s):%.6f" % (_newick(node["left"]), _newick(node["right"]), node["blen"])


def _newick_tree(root):
    return "(%s,%s);" % (_newick(root["left"]), _newick(root["right"]))


def _vcv(tips, total_height):
    anc = []
    for tip in tips:
        seen = {}
        node = tip
        while node is not None:
            seen[id(node)] = node["height"]
            node = node["parent"]
        anc.append(seen)
    n = len(tips)
    C = np.zeros((n, n))
    for i in range(n):
        C[i, i] = total_height
        for j in range(i + 1, n):
            h = 0.0
            for key, val in anc[i].items():
                if val > h and key in anc[j]:
                    h = val
            C[i, j] = C[j, i] = total_height - h
    return C


def _to_traits(Z):
    X = np.empty_like(Z, dtype=float)
    X[:, 0] = np.exp(0.55 + 0.42 * Z[:, 0])
    X[:, 1] = 0.16 * np.exp(0.32 * Z[:, 1])
    X[:, 2] = np.clip(8.0 + 14.0 * Z[:, 2], -30.0, 55.0)
    X[:, 3] = 0.24 * np.exp(0.28 * Z[:, 3])
    X[:, 4] = 0.17 * np.exp(0.25 * Z[:, 4])
    X[:, 5] = 0.21 * np.exp(0.26 * Z[:, 5])
    X[:, 6] = 0.29 * np.exp(0.17 * Z[:, 6])
    X[:, 7] = np.exp(-0.45 + 0.33 * Z[:, 7])
    return X


def generate(seed=SEED):
    rng = np.random.default_rng(seed)

    all_names = _make_names(rng)
    perm = rng.permutation(N_MORPH)
    tree_idx = np.sort(perm[:N_TREE])
    extra_idx = np.sort(perm[N_TREE:])
    tree_names = [all_names[int(i)] for i in tree_idx]

    root, tips = _simulate_tree(rng, tree_names)
    total_height = float(root["height"])
    C = _vcv(tips, total_height)
    newick = _newick_tree(root)

    L = np.linalg.cholesky(C)
    Zf = rng.standard_normal((N_TREE, 3))
    Ze = rng.standard_normal((N_TREE, 3))
    Xp = L @ (Zf @ _A.T + Ze @ _B.T)
    Xp = Xp / Xp.std(axis=0, ddof=1)
    Xp = Xp + 0.65 * rng.standard_normal((N_TREE, 8))
    Xp = (Xp - Xp.mean(axis=0)) / Xp.std(axis=0, ddof=1)
    Xex = rng.standard_normal((N_MORPH - N_TREE, 8))
    Zfull = np.empty((N_MORPH, 8))
    Zfull[tree_idx] = Xp
    Zfull[extra_idx] = Xex
    M_true = _to_traits(Zfull)

    clade = [CLADES[int(rng.integers(len(CLADES)))] for _ in range(N_MORPH)]
    water = [WATER[int(rng.integers(len(WATER)))] for _ in range(N_MORPH)]

    Zm_all = (M_true - M_true.mean(0)) / M_true.std(0, ddof=1)
    Zm = Zm_all[tree_idx]
    wc = np.array([water[int(i)] for i in tree_idx])
    cl = np.array([clade[int(i)] for i in tree_idx])

    order = rng.permutation(N_TREE)
    train_idx = np.sort(order[:N_TRAIN])
    test_idx = np.sort(order[N_TRAIN:])

    pel = (wc == "pelagic").astype(float)
    dem = (wc == "demersal").astype(float)
    ben = (wc == "benthic").astype(float)

    lat1 = (0.85 * Zm[:, 0] - 0.65 * Zm[:, 1] + 0.55 * Zm[:, 2]
            + 0.25 * Zm[:, 3] - 0.20 * Zm[:, 6]
            + pel * (1.10 * Zm[:, 0] + 0.35 * Zm[:, 1])
            + ben * (-0.90 * Zm[:, 0] - 0.40 * Zm[:, 2])
            + dem * (-0.50 * Zm[:, 0]))
    gt = np.exp(1.05 + 0.42 * lat1 + 0.38 * rng.standard_normal(N_TREE))

    lat2 = (-0.55 * Zm[:, 0] + 0.45 * Zm[:, 1] + 0.65 * Zm[:, 2]
            + 0.25 * Zm[:, 4] + 0.35 * Zm[:, 5])
    inter = ((cl == "Perciformes") * (0.80 * Zm[:, 2] - 0.40 * Zm[:, 1])
             + (cl == "Cypriniformes") * (0.60 * Zm[:, 0] + 0.50 * Zm[:, 5])
             + (cl == "Salmoniformes") * (-0.55 * Zm[:, 0] + 0.30 * Zm[:, 4])
             + (cl == "Gadiformes") * (0.50 * Zm[:, 1] - 0.30 * Zm[:, 6])
             + (cl == "Clupeiformes") * (0.45 * Zm[:, 0] - 0.25 * Zm[:, 3]))
    repro = np.exp(0.35 + 0.45 * (lat2 + inter) + 0.36 * rng.standard_normal(N_TREE))

    morph_csv = np.round(M_true, 4).astype(float)
    missing = set()
    while len(missing) < N_MISSING_CELLS:
        missing.add((int(rng.integers(N_MORPH)), int(rng.integers(len(TRAITS)))))
    for r, c in missing:
        morph_csv[r, c] = np.nan

    return {
        "seed": seed,
        "all_names": all_names,
        "names_display": [n.replace("_", " ") for n in all_names],
        "clade": clade,
        "water": water,
        "morph_true": M_true,
        "morph_csv": morph_csv,
        "tree_names": tree_names,
        "tree_newick": newick,
        "C": C,
        "gt": gt,
        "rep": repro,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


def observable(d):
    tree_names = d["tree_names"]
    return {
        "names_display": d["names_display"],
        "clade": d["clade"],
        "water": d["water"],
        "morph": d["morph_csv"],
        "tree_names": tree_names,
        "tree_newick": d["tree_newick"],
        "C": d["C"],
        "train_names": [tree_names[int(k)].replace("_", " ") for k in d["train_idx"]],
        "test_names": [tree_names[int(k)].replace("_", " ") for k in d["test_idx"]],
        "train_gt": {tree_names[int(k)].replace("_", " "): float(d["gt"][int(k)])
                     for k in d["train_idx"]},
        "train_rep": {tree_names[int(k)].replace("_", " "): float(d["rep"][int(k)])
                      for k in d["train_idx"]},
    }


def hidden_truth(d):
    tree_names = d["tree_names"]
    return {norm(tree_names[int(k)]): (float(d["gt"][int(k)]), float(d["rep"][int(k)]))
            for k in d["test_idx"]}


def reference_pipeline(obs):
    """Recompute the reference diagnostics from the observable data only."""
    names_display = obs["names_display"]
    tree_names = obs["tree_names"]
    idx_of = {norm(n): i for i, n in enumerate(names_display)}
    rows = [idx_of[norm(t)] for t in tree_names]
    M = obs["morph"][rows].astype(float).copy()
    clade = [obs["clade"][i] for i in rows]
    n = len(tree_names)

    for c in sorted(set(clade)):
        sel = [i for i in range(n) if clade[i] == c]
        for j in range(M.shape[1]):
            col = M[sel, j]
            med = float(np.nanmedian(col))
            col[np.isnan(col)] = med
            M[sel, j] = col

    X = (M - M.mean(0)) / M.std(0, ddof=1)
    C = np.asarray(obs["C"], float)
    w, V = np.linalg.eigh(C)
    Cinv_half = (V * (1.0 / np.sqrt(w))) @ V.T
    Xw = Cinv_half @ X
    Xw = Xw - Xw.mean(0)
    _, s, Vt = np.linalg.svd(Xw, full_matrices=False)
    ev = s ** 2 / (n - 1)
    var_expl = float(ev[0] / ev.sum())
    load = Vt[0].copy()
    if load[TRAITS.index("body_elongation")] < 0:
        load = -load

    train_key = {norm(s) for s in obs["train_names"]}
    tri = [i for i in range(n) if norm(tree_names[i]) in train_key]
    Mtr = M[tri]
    Ztr = (Mtr - Mtr.mean(0)) / Mtr.std(0, ddof=1)
    y = np.array([np.log(obs["train_gt"][tree_names[i].replace("_", " ")]) for i in tri])
    Ctr = C[np.ix_(tri, tri)]
    Ci = np.linalg.inv(Ctr)
    cols = [TRAITS.index(t) for t in LATERAL]
    Xd = np.column_stack([np.ones(len(tri))] + [Ztr[:, c] for c in cols])
    beta = np.linalg.solve(Xd.T @ Ci @ Xd, Xd.T @ Ci @ y)

    return {
        "n_species": int(n),
        "n_train": int(len(tri)),
        "var_expl": float(var_expl),
        "loadings": [float(v) for v in load],
        "slopes": [float(b) for b in beta[1:]],
    }


def write_files(d, outdir):
    os.makedirs(outdir, exist_ok=True)
    disp = d["names_display"]

    with open(os.path.join(outdir, "morphology.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "clade", "water_column"] + TRAITS)
        for i in sorted(range(N_MORPH), key=lambda k: disp[k]):
            row = [disp[i], d["clade"][i], d["water"][i]]
            for j in range(len(TRAITS)):
                val = d["morph_csv"][i, j]
                row.append("" if np.isnan(val) else "%.4f" % val)
            w.writerow(row)

    with open(os.path.join(outdir, "life_history_train.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "generation_turnover_years", "reproductive_output_index"])
        for k in sorted(d["train_idx"], key=lambda q: d["tree_names"][int(q)]):
            k = int(k)
            w.writerow([d["tree_names"][k].replace("_", " "),
                        "%.4f" % d["gt"][k], "%.4f" % d["rep"][k]])

    with open(os.path.join(outdir, "test_species.txt"), "w") as fh:
        for k in sorted(d["test_idx"], key=lambda q: d["tree_names"][int(q)]):
            print(d["tree_names"][int(k)].replace("_", " "), file=fh)

    with open(os.path.join(outdir, "phylogeny.nwk"), "w") as fh:
        print(d["tree_newick"], file=fh)


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    write_files(generate(), outdir)
    print("wrote benchmark inputs to", outdir)
