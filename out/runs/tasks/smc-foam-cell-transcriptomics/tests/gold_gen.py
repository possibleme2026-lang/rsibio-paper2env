#!/usr/bin/env python3
import argparse, json, os
import numpy as np
import pandas as pd

SIG_GENES = ["CFH", "SERPINE1", "C3", "C4A", "C4B", "CFB", "CFD", "CFI", "F2", "F3", "F5", "F7", "F9", "F10", "F11", "F12", "F13A1", "PLAT", "PLAU", "PLG", "PROS1", "PROC", "PROCR", "THBD", "VEGFA", "ANGPT2", "ANGPTL4", "FLT1", "KDR", "PECAM1", "VWF", "COL1A1", "COL3A1", "COL5A1", "COL5A2", "COL6A1", "COL6A2", "COL6A3", "FN1", "TIMP1", "TIMP2", "MMP2", "MMP14", "SPARC", "FBLN1", "FBLN2", "FBLN5", "FBN1", "TGFB1", "TGFB2", "TGFB3"]
CHOL_GENES = ["IL6", "TNF", "IL1B", "CXCL8", "CCL2", "CCL5", "NFKB1", "RELA", "STAT1", "STAT3"]

def build_genes():
    markers = ["MYH11","ACTA2","TAGLN","CNN1","MYOCD","LMOD1","DES","CALD1","CD68","CD14","LYZ","AIF1","ITGAM","PTPRC","C1QA","C1QB","C1QC","APOE","ABCA1","ABCG1","NR1H3","CD3D","CD3E","IL7R","DCN","LUM","RGS5","PDGFRB","ACTA1","MYL9","MT-ND1","MT-ND2","MT-CO1","MT-CO2","MT-ATP6","MT-CYB"]
    rand = [f"GENE{i:04d}" for i in range(1, 700)]
    return list(dict.fromkeys(markers + SIG_GENES + CHOL_GENES + rand))

def get_hallmark_sets():
    return {
      "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": ["FN1","COL1A1","COL3A1","COL5A1","COL5A2","COL6A1","COL6A2","COL6A3","COL8A1","COL8A2","FBN1","FBLN1","FBLN2","FBLN5","SPARC","TIMP1","TIMP2","TIMP3","MMP2","MMP14","SERPINE1","TGFB1","TGFB2","TGFB3"],
      "HALLMARK_COAGULATION": ["SERPINE1","F2","F3","F5","F7","F8","F9","F10","F11","F12","F13A1","PLAT","PLAU","PLG","PROS1","PROC","PROCR","THBD","VWF"],
      "HALLMARK_COMPLEMENT": ["CFH","C1QA","C1QB","C1QC","C2","C3","C4A","C4B","C5","C6","C7","C8A","C8B","C8G","C9","CFB","CFD","CFI","CFHR1"],
      "HALLMARK_ANGIOGENESIS": ["VEGFA","VEGFB","VEGFC","FLT1","KDR","FLT4","ANGPT1","ANGPT2","ANGPTL4","PECAM1","VWF","CDH5","TEK","TIE1","PDGFB","PDGFRB"],
      "HALLMARK_INFLAMMATORY_RESPONSE": ["IL6","TNF","IL1B","IL1A","CXCL8","CXCL10","CCL2","CCL5","NFKB1","NFKB2","RELA","RELB","STAT1","STAT3"],
      "HALLMARK_OXIDATIVE_PHOSPHORYLATION": ["NDUFA1","NDUFA2","NDUFB1","NDUFB2","SDHA","SDHB","UQCRC1","UQCRC2","CYC1","COX4I1","COX5A","ATP5F1A","ATP5F1B"],
      "HALLMARK_FATTY_ACID_METABOLISM": ["ACADM","ACADL","ACADVL","CPT1A","CPT1B","CPT2","HADHA","HADHB"],
      "HALLMARK_CHOLESTEROL_HOMEOSTASIS": ["HMGCR","HMGCS1","LDLR","SREBF2","INSIG1","INSIG2","SCAP","MBTPS1"],
      "HALLMARK_APOPTOSIS": ["BAX","BAK1","CASP3","CASP7","CASP8","CASP9","BCL2","BCL2L1","MCL1"],
      "HALLMARK_TNFA_SIGNALING_VIA_NFKB": ["IL6","TNF","IL1B","CXCL8","NFKB1","NFKB2","RELA","RELB","STAT1","STAT3","CCL2","CCL5"]
    }

def generate_dataset(outdir, seed=42, n_cells_per_type=80):
    rng = np.random.default_rng(seed)
    genes = build_genes()
    n_genes = len(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    cell_types = ["SMC", "SMC_foam", "Mac", "Mac_foam", "T", "Endo", "Fibro", "Pericyte"]
    cluster_ids = rng.permutation(len(cell_types))
    type_to_cluster = {ct: int(cluster_ids[i]) for i, ct in enumerate(cell_types)}
    base = {ct: np.full(n_genes, 0.2) for ct in cell_types}
    def set_expr(ct, gene, val):
        if gene in gene_to_idx:
            base[ct][gene_to_idx[gene]] = val
    for g in ["MYH11","ACTA2","TAGLN","CNN1","MYOCD","LMOD1","DES","CALD1"]:
        set_expr("SMC", g, 3.5)
    for g in ["MYH11","ACTA2","TAGLN"]:
        set_expr("SMC_foam", g, 2.0)
    for g in SIG_GENES:
        set_expr("SMC_foam", g, 3.5)
    for g in ["CD68","LYZ","CD14","AIF1","ITGAM","PTPRC","C1QA","C1QB","C1QC"]:
        set_expr("Mac", g, 3.5)
    for g in ["CD68","LYZ","CD14","AIF1","PTPRC"]:
        set_expr("Mac_foam", g, 3.5)
    for g in ["APOE","ABCA1","ABCG1","NR1H3","IL6","TNF","IL1B","CXCL8","CCL2"]:
        set_expr("Mac_foam", g, 3.0)
    for g in ["CD3D","CD3E","IL7R"]:
        set_expr("T", g, 3.5)
    for g in ["PECAM1","VWF","CDH5"]:
        set_expr("Endo", g, 3.5)
    for g in ["DCN","LUM","COL1A1","COL3A1","FN1"]:
        set_expr("Fibro", g, 2.5)
    for g in ["RGS5","PDGFRB","ACTA2"]:
        set_expr("Pericyte", g, 2.5)
    cells = []
    clusters = []
    cell_type_list = []
    for ct in cell_types:
        for i in range(n_cells_per_type):
            cells.append(f"{ct}_{i:03d}")
            clusters.append(type_to_cluster[ct])
            cell_type_list.append(ct)
    perm = rng.permutation(len(cells))
    cells = [cells[i] for i in perm]
    clusters = [clusters[i] for i in perm]
    cell_type_list = [cell_type_list[i] for i in perm]
    counts = np.zeros((n_genes, len(cells)), dtype=int)
    mt_idx = [gene_to_idx[g] for g in genes if g.startswith("MT-")]
    for j, ct in enumerate(cell_type_list):
        total = int(rng.lognormal(mean=9.0, sigma=0.3))
        if rng.random() < 0.05:
            total = 100
        for i in range(n_genes):
            log1p_mean = base[ct][i]
            mean_cp10k = np.expm1(log1p_mean)
            lam = mean_cp10k * total / 10000.0
            if lam > 0:
                theta = 5.0
                p = theta / (theta + lam)
                counts[i, j] = rng.negative_binomial(theta, p)
            else:
                counts[i, j] = 0
        if rng.random() < 0.10:
            for idx in mt_idx:
                counts[idx, j] = int(counts[idx, j] * 5)
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(counts, index=genes, columns=cells).to_csv(os.path.join(outdir, "sc_counts.csv"))
    with open(os.path.join(outdir, "sc_genes.txt"), "w") as f:
        f.write("\n".join(genes) + "\n")
    with open(os.path.join(outdir, "sc_cells.txt"), "w") as f:
        f.write("\n".join(cells) + "\n")
    pd.DataFrame({"cell_id": cells, "cluster": clusters}).to_csv(os.path.join(outdir, "cell_metadata.csv"), index=False)
    samples = []
    conditions = []
    for cond in ["ctrl", "agLDL", "chol"]:
        for rep in range(3):
            samples.append(f"{cond}_{rep+1}")
            conditions.append(cond)
    bulk_counts = np.zeros((n_genes, len(samples)), dtype=int)
    base_cpm = rng.uniform(5, 50, size=n_genes)
    for j, cond in enumerate(conditions):
        total = 1_000_000
        for i, g in enumerate(genes):
            cpm = base_cpm[i]
            if cond == "agLDL" and g in SIG_GENES:
                cpm = 100.0
            elif cond == "chol" and g in CHOL_GENES:
                cpm = 100.0
            elif cond == "ctrl" and g in SIG_GENES:
                cpm = 1.0
            elif cond == "ctrl" and g in CHOL_GENES:
                cpm = 1.0
            lam = cpm * total / 1e6
            bulk_counts[i, j] = rng.poisson(lam)
    pd.DataFrame(bulk_counts, index=genes, columns=samples).to_csv(os.path.join(outdir, "bulk_counts.csv"))
    pd.DataFrame({"sample_id": samples, "condition": conditions}).to_csv(os.path.join(outdir, "bulk_samples.csv"), index=False)
    with open(os.path.join(outdir, "hallmark_sets.json"), "w") as f:
        json.dump(get_hallmark_sets(), f, indent=2)
    return {"type_to_cluster": type_to_cluster}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.outdir, args.seed)
