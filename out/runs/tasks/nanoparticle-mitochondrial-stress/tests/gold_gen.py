#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd

def generate(outdir, seed=20260729):
    rng = np.random.default_rng(seed)
    genes = [
        "NFE2L2","HMOX1","NQO1","GCLM","TXN","SOD1","CAT","GPX1",
        "PPARGC1A","TFAM","NRF1","POLG","TWNK",
        "OMA1","DELE1","ATF4","DDIT3","CHAC1",
        "NFKB1","TNF","IL6","NLRP3","IL1B","CXCL8",
        "MT-ND1","NDUFA1","SDHA","UQCRC1","COX5A","ATP5F1A"
    ]
    donors = [f"D{i:02d}" for i in range(1,13)]
    particles = ["UFPM", "BC", "PS_NP"]
    times = [6, 24, 72]
    samples = []
    for d in donors:
        for p in particles:
            for t in times:
                samples.append({
                    "sample_id": f"{d}_{p}_{t}h",
                    "donor": d,
                    "particle": p,
                    "time_h": t
                })
    meta = pd.DataFrame(samples)
    n = len(meta)
    g = len(genes)
    base = rng.normal(5.0, 1.0, size=g)
    donor_effect = rng.normal(0.0, 0.2, size=(len(donors), g))
    donor_idx = {d: i for i, d in enumerate(donors)}

    oxi_genes = ["NFE2L2","HMOX1","NQO1","GCLM","TXN","SOD1","CAT","GPX1"]
    adapt_genes = ["PPARGC1A","TFAM","NRF1","POLG","TWNK"]
    isr_genes = ["OMA1","DELE1","ATF4","DDIT3","CHAC1"]
    inflam_genes = ["NFKB1","TNF","IL6","NLRP3","IL1B","CXCL8"]
    complex_genes = ["MT-ND1","NDUFA1","SDHA","UQCRC1","COX5A","ATP5F1A"]

    O = {
        "UFPM": {6: 2.5, 24: 1.3, 72: 0.5},
        "BC": {6: 2.0, 24: 2.2, 72: 2.4},
        "PS_NP": {6: 0.5, 24: 0.7, 72: 0.9}
    }
    A = {
        "UFPM": {6: 1.2, 24: 1.8, 72: 1.5},
        "BC": {6: 0.3, 24: 0.1, 72: -0.5},
        "PS_NP": {6: 0.8, 24: 1.4, 72: 2.0}
    }
    I = {
        "UFPM": {6: 0.2, 24: 0.3, 72: 0.2},
        "BC": {6: 1.5, 24: 2.0, 72: 2.5},
        "PS_NP": {6: 0.8, 24: 1.2, 72: 1.6}
    }
    F = {
        "UFPM": {6: 0.5, 24: 0.4, 72: 0.2},
        "BC": {6: 1.0, 24: 1.6, 72: 2.2},
        "PS_NP": {6: 0.3, 24: 0.4, 72: 0.6}
    }
    C = {
        "UFPM": {6: 0.2, 24: 0.3, 72: 0.2},
        "BC": {6: -0.3, 24: -0.6, 72: -1.2},
        "PS_NP": {6: 0.2, 24: 0.3, 72: 0.4}
    }
    S = {
        "UFPM": {6: 0.8, 24: 1.2, 72: 1.5},
        "BC": {6: 1.5, 24: 2.2, 72: 3.0},
        "PS_NP": {6: 0.9, 24: 1.4, 72: 2.0}
    }

    common_load = {gene: rng.uniform(0.6, 1.0) for gene in genes}
    common_load["OMA1"] = 0.9
    common_load["DELE1"] = 0.9
    oxi_load = {gene: rng.uniform(0.7, 1.2) for gene in oxi_genes}
    adapt_load = {gene: rng.uniform(0.7, 1.2) for gene in adapt_genes}
    isr_load = {gene: rng.uniform(0.7, 1.2) for gene in isr_genes}
    inflam_load = {gene: rng.uniform(0.7, 1.2) for gene in inflam_genes}
    complex_load = {gene: rng.uniform(0.7, 1.2) for gene in complex_genes}

    expr = np.zeros((n, g))
    for i, row in meta.iterrows():
        d = row["donor"]
        p = row["particle"]
        t = row["time_h"]
        de = donor_effect[donor_idx[d]]
        for j, gene in enumerate(genes):
            val = base[j] + de[j]
            val += S[p][t] * common_load[gene]
            if gene in oxi_genes:
                val += O[p][t] * oxi_load[gene]
            if gene in adapt_genes:
                val += A[p][t] * adapt_load[gene]
            if gene in isr_genes:
                val += I[p][t] * isr_load[gene]
            if gene in inflam_genes:
                val += F[p][t] * inflam_load[gene]
            if gene in complex_genes:
                val += C[p][t] * complex_load[gene]
            val += rng.normal(0.0, 0.15)
            expr[i, j] = val

    expr_df = pd.DataFrame(expr, columns=genes)
    expr_df.insert(0, "sample_id", meta["sample_id"])
    os.makedirs(outdir, exist_ok=True)
    expr_df.to_csv(os.path.join(outdir, "expression_matrix.csv"), index=False)
    meta.to_csv(os.path.join(outdir, "sample_metadata.csv"), index=False)

if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    generate(outdir)
