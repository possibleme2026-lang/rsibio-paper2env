#!/usr/bin/env python3
import json, os, sys, tempfile, subprocess
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import gseapy

sys.path.insert(0, os.path.dirname(__file__))
import gold_gen

def analyze(data_dir):
    counts = pd.read_csv(os.path.join(data_dir, "sc_counts.csv"), index_col=0)
    genes = counts.index.tolist()
    cells = counts.columns.tolist()
    meta = pd.read_csv(os.path.join(data_dir, "cell_metadata.csv"))
    meta = meta.set_index("cell_id")
    total_counts = counts.sum(axis=0)
    mt_genes = [g for g in genes if g.startswith("MT-")]
    mt_counts = counts.loc[mt_genes].sum(axis=0) if mt_genes else pd.Series(0, index=cells)
    mito_frac = mt_counts / total_counts
    detected_genes = (counts > 0).sum(axis=0)
    keep_cells = (detected_genes >= 200) & (mito_frac < 0.20)
    counts_f = counts.loc[:, keep_cells]
    detected_genes_per_gene = (counts_f > 0).sum(axis=1)
    keep_genes = detected_genes_per_gene >= 3
    counts_f = counts_f.loc[keep_genes]
    n_cells = counts_f.shape[1]
    n_genes = counts_f.shape[0]
    totals = counts_f.sum(axis=0)
    cp10k = counts_f.div(totals, axis=1) * 10000
    log1p = np.log1p(cp10k)
    bulk = pd.read_csv(os.path.join(data_dir, "bulk_counts.csv"), index_col=0)
    samples = pd.read_csv(os.path.join(data_dir, "bulk_samples.csv"))
    bulk_totals = bulk.sum(axis=0)
    bulk_cpm = bulk.div(bulk_totals, axis=1) * 1e6
    bulk_log2cpm = np.log2(bulk_cpm + 1)
    ctrl_samples = samples[samples["condition"] == "ctrl"]["sample_id"].tolist()
    agldl_samples = samples[samples["condition"] == "agLDL"]["sample_id"].tolist()
    log2fc_bulk = bulk_log2cpm[agldl_samples].mean(axis=1) - bulk_log2cpm[ctrl_samples].mean(axis=1)
    pvals = []
    for g in bulk.index:
        a = bulk_log2cpm.loc[g, agldl_samples].values
        c = bulk_log2cpm.loc[g, ctrl_samples].values
        t, p = stats.ttest_ind(a, c, equal_var=False)
        pvals.append(p)
    padj = multipletests(pvals, method="fdr_bh")[1]
    sig_df = pd.DataFrame({"log2fc": log2fc_bulk, "padj": padj}, index=bulk.index)
    signature = sig_df[(sig_df["log2fc"] > 0.5) & (sig_df["padj"] < 0.05)].index.tolist()
    cluster_ids = meta.loc[counts_f.columns, "cluster"].values
    sig_genes_present = [g for g in signature if g in log1p.index]
    sig_expr = log1p.loc[sig_genes_present].mean(axis=0) if sig_genes_present else pd.Series(0, index=log1p.columns)
    myh11 = log1p.loc["MYH11"] if "MYH11" in log1p.index else pd.Series(0, index=log1p.columns)
    cd68 = log1p.loc["CD68"] if "CD68" in log1p.index else pd.Series(0, index=log1p.columns)
    apoe = log1p.loc["APOE"] if "APOE" in log1p.index else pd.Series(0, index=log1p.columns)
    cluster_stats = pd.DataFrame({"cluster": cluster_ids, "sig": sig_expr.values, "myh11": myh11.values, "cd68": cd68.values, "apoe": apoe.values}).groupby("cluster").mean()
    smc_candidates = cluster_stats[(cluster_stats["myh11"] > 0.5) & (cluster_stats["cd68"] < 0.5)]
    smc_foam_cluster = int(smc_candidates["sig"].idxmax()) if len(smc_candidates) > 0 else -1
    mac_candidates = cluster_stats[cluster_stats["cd68"] > 1.0]
    mac_foam_cluster = int(mac_candidates["apoe"].idxmax()) if len(mac_candidates) > 0 else -1
    smc_cells = meta.index[meta["cluster"] == smc_foam_cluster]
    mac_cells = meta.index[meta["cluster"] == mac_foam_cluster]
    smc_cells = [c for c in smc_cells if c in log1p.columns]
    mac_cells = [c for c in mac_cells if c in log1p.columns]
    mean_smc = cp10k[smc_cells].mean(axis=1)
    mean_mac = cp10k[mac_cells].mean(axis=1)
    log2fc_de = np.log2((mean_smc + 1) / (mean_mac + 1))
    pvals_de = []
    for g in log1p.index:
        a = log1p.loc[g, smc_cells].values
        b = log1p.loc[g, mac_cells].values
        t, p = stats.ttest_ind(a, b, equal_var=False)
        pvals_de.append(p)
    padj_de = multipletests(pvals_de, method="fdr_bh")[1]
    de_df = pd.DataFrame({"log2fc": log2fc_de, "padj": padj_de}, index=log1p.index).reset_index().rename(columns={"index": "gene"})
    de_up = de_df[de_df["log2fc"] > 0].sort_values(["log2fc", "padj", "gene"], ascending=[False, True, True])
    top20 = de_up.head(20)[["gene", "log2fc", "padj"]].to_dict(orient="records")
    hallmark = json.load(open(os.path.join(data_dir, "hallmark_sets.json")))
    rnk = de_df.set_index("gene")["log2fc"].sort_values(ascending=False)
    gsea_res = gseapy.prerank(rnk=rnk, gene_sets=hallmark, min_size=5, max_size=500, permutation_num=200, seed=42, threads=1, outdir=None)
    res2d = gsea_res.res2d
    res2d = res2d[res2d["NES"] > 0].sort_values("NES", ascending=False)
    top5 = res2d.head(5)[["Term", "NES", "FDR q-val"]].rename(columns={"Term": "pathway", "NES": "nes", "FDR q-val": "fdr"}).to_dict(orient="records")
    marker_val = {}
    for g in ["CFH", "SERPINE1"]:
        if g in de_df["gene"].values:
            row = de_df[de_df["gene"] == g].iloc[0]
            marker_val[g] = bool(row["log2fc"] > 1.0 and row["padj"] < 0.01)
        else:
            marker_val[g] = False
    return {
        "qc": {"n_cells_retained": int(n_cells), "n_genes_retained": int(n_genes)},
        "invitro_signature": signature,
        "smc_foam_cluster": smc_foam_cluster,
        "macrophage_foam_cluster": mac_foam_cluster,
        "top_upregulated": top20,
        "gsea_top": top5,
        "marker_validation": marker_val,
    }

def compare_results(agent, ref):
    results = {}
    required = ["qc", "invitro_signature", "smc_foam_cluster", "macrophage_foam_cluster", "top_upregulated", "gsea_top", "marker_validation"]
    results["format"] = all(k in agent for k in required)
    try:
        results["qc"] = (agent["qc"]["n_cells_retained"] == ref["qc"]["n_cells_retained"] and agent["qc"]["n_genes_retained"] == ref["qc"]["n_genes_retained"])
    except Exception:
        results["qc"] = False
    try:
        a_sig = set(agent["invitro_signature"])
        r_sig = set(ref["invitro_signature"])
        inter = len(a_sig & r_sig)
        union = len(a_sig | r_sig)
        jaccard = inter / union if union > 0 else 0
        results["signature"] = ("CFH" in a_sig and "SERPINE1" in a_sig and jaccard >= 0.8)
    except Exception:
        results["signature"] = False
    try:
        results["smc_cluster"] = (agent["smc_foam_cluster"] == ref["smc_foam_cluster"])
    except Exception:
        results["smc_cluster"] = False
    try:
        results["mac_cluster"] = (agent["macrophage_foam_cluster"] == ref["macrophage_foam_cluster"])
    except Exception:
        results["mac_cluster"] = False
    try:
        a_top = agent["top_upregulated"]
        r_top = ref["top_upregulated"]
        a_genes = [x["gene"] for x in a_top]
        r_genes = [x["gene"] for x in r_top]
        overlap = len(set(a_genes) & set(r_genes))
        cfh_ok = any(x["gene"] == "CFH" and x["log2fc"] > 1.0 and x["padj"] < 0.01 for x in a_top)
        serp_ok = any(x["gene"] == "SERPINE1" and x["log2fc"] > 1.0 and x["padj"] < 0.01 for x in a_top)
        results["top20"] = (overlap >= 15 and cfh_ok and serp_ok)
    except Exception:
        results["top20"] = False
    try:
        a_gsea = agent["gsea_top"]
        r_gsea = ref["gsea_top"]
        a_paths = [x["pathway"] for x in a_gsea]
        r_paths = [x["pathway"] for x in r_gsea]
        overlap = len(set(a_paths) & set(r_paths))
        a_top10 = a_paths[:10]
        comp_ok = "HALLMARK_COMPLEMENT" in a_top10 and any(x["pathway"] == "HALLMARK_COMPLEMENT" and x["nes"] > 0 for x in a_gsea)
        coag_ok = "HALLMARK_COAGULATION" in a_top10 and any(x["pathway"] == "HALLMARK_COAGULATION" and x["nes"] > 0 for x in a_gsea)
        results["gsea"] = (overlap >= 3 and comp_ok and coag_ok)
    except Exception:
        results["gsea"] = False
    try:
        mv = agent["marker_validation"]
        results["marker_val"] = (mv.get("CFH") is True and mv.get("SERPINE1") is True)
    except Exception:
        results["marker_val"] = False
    return results

def main():
    app_dir = "/app"
    results_path = os.path.join(app_dir, "results.json")
    pipeline_path = os.path.join(app_dir, "pipeline.py")
    data_dir = os.path.join(app_dir, "data")
    agent_res = {}
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                agent_res = json.load(f)
        except Exception as e:
            print(f"Error loading results.json: {e}")
    ref_res = analyze(data_dir)
    constraint_results = compare_results(agent_res, ref_res)
    heldout_ok = False
    if os.path.exists(pipeline_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            secret_seed = 987654321
            gold_gen.generate_dataset(tmpdir, seed=secret_seed)
            heldout_out = os.path.join(tmpdir, "heldout_results.json")
            cmd = [sys.executable, pipeline_path, "--data_dir", tmpdir, "--out", heldout_out]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode == 0 and os.path.exists(heldout_out):
                try:
                    with open(heldout_out) as f:
                        heldout_agent = json.load(f)
                    heldout_ref = analyze(tmpdir)
                    hc = compare_results(heldout_agent, heldout_ref)
                    keys = ["qc", "signature", "smc_cluster", "mac_cluster", "top20", "gsea", "marker_val"]
                    heldout_ok = sum(1 for k in keys if hc.get(k, False)) >= 6
                except Exception as e:
                    print(f"Held-out comparison error: {e}")
            else:
                print(f"Held-out pipeline failed: {proc.stderr}")
    constraint_results["heldout"] = heldout_ok
    all_constraints = ["format", "qc", "signature", "smc_cluster", "mac_cluster", "top20", "gsea", "marker_val", "heldout"]
    score = sum(1 for c in all_constraints if constraint_results.get(c, False)) / len(all_constraints)
    os.makedirs("/logs/verifier", exist_ok=True)
    with open("/logs/verifier/reward.json", "w") as f:
        json.dump({"score": score, "reward": score}, f)
    print(json.dumps(constraint_results, indent=2))
    print(f"Score: {score}")

if __name__ == "__main__":
    main()
