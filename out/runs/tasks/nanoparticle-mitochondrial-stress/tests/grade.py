#!/usr/bin/env python3
import os, sys, json, tempfile, subprocess, shutil
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

AGENT_OUTPUT = "/app/output/results.json"
REPORT_PATH = "/logs/verifier/report.json"
REWARD_PATH = "/logs/verifier/reward.json"
GOLD_GEN = "/tests/gold_gen.py"

GENES = [
    "NFE2L2","HMOX1","NQO1","GCLM","TXN","SOD1","CAT","GPX1",
    "PPARGC1A","TFAM","NRF1","POLG","TWNK",
    "OMA1","DELE1","ATF4","DDIT3","CHAC1",
    "NFKB1","TNF","IL6","NLRP3","IL1B","CXCL8",
    "MT-ND1","NDUFA1","SDHA","UQCRC1","COX5A","ATP5F1A"
]
OXI_GENES = ["NFE2L2","HMOX1","NQO1","GCLM","TXN","SOD1","CAT","GPX1"]
ADAPT_GENES = ["PPARGC1A","TFAM","NRF1","POLG","TWNK"]
ISR_GENES = ["OMA1","DELE1","ATF4","DDIT3","CHAC1"]
INFLAM_GENES = ["NFKB1","TNF","IL6","NLRP3","IL1B","CXCL8"]
COMPLEX_GENES = ["MT-ND1","NDUFA1","SDHA","UQCRC1","COX5A","ATP5F1A"]

def run_gold_gen(tmpdir):
    subprocess.check_call([sys.executable, GOLD_GEN, tmpdir])

def load_data(data_dir):
    expr = pd.read_csv(os.path.join(data_dir, "expression_matrix.csv"))
    meta = pd.read_csv(os.path.join(data_dir, "sample_metadata.csv"))
    expr = expr.set_index("sample_id")
    meta = meta.set_index("sample_id")
    expr = expr.loc[meta.index]
    return expr, meta

def zscore_df(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

def compute_pca(z):
    pca = PCA(n_components=5, svd_solver="full", random_state=0)
    pcs = pca.fit_transform(z)
    return pca, pcs

def top_loadings(pca, comp, genes, k=5):
    loadings = pca.components_[comp]
    abs_load = np.abs(loadings)
    order = sorted(range(len(genes)), key=lambda i: (-abs_load[i], genes[i]))
    return [genes[i] for i in order[:k]]

def hclust(z, pcs):
    dist = 1 - np.corrcoef(z.values)
    dist = np.clip(dist, 0, 2)
    Z = linkage(dist, method="average")
    clusters = fcluster(Z, t=3, criterion="maxclust")
    cluster_means = {c: pcs[clusters == c, 0].mean() for c in np.unique(clusters)}
    sorted_clusters = sorted(cluster_means, key=lambda c: cluster_means[c])
    mapping = {old: new for new, old in enumerate(sorted_clusters)}
    labels = np.array([mapping[c] for c in clusters])
    return {sid: int(lab) for sid, lab in zip(z.index, labels)}

def gene_corr_matrix(z):
    return np.corrcoef(z.values.T)

def hub_genes(corr, genes, threshold=0.7, k=5):
    adj = np.abs(corr) >= threshold
    np.fill_diagonal(adj, False)
    degrees = adj.sum(axis=1)
    order = sorted(range(len(genes)), key=lambda i: (-degrees[i], genes[i]))
    return [genes[i] for i in order[:k]]

def conserved_edges(expr, meta, genes, threshold=0.7):
    particles = meta["particle"].unique()
    edge_sets = []
    for p in particles:
        sub = expr[meta["particle"] == p]
        corr = np.corrcoef(sub.values.T)
        adj = np.abs(corr) >= threshold
        np.fill_diagonal(adj, False)
        edges = set()
        for i in range(len(genes)):
            for j in range(i+1, len(genes)):
                if adj[i, j]:
                    e = (genes[i], genes[j]) if genes[i] < genes[j] else (genes[j], genes[i])
                    edges.add(e)
        edge_sets.append(edges)
    conserved = edge_sets[0].intersection(*edge_sets[1:])
    return conserved

def top_conserved_edges(expr, meta, genes, conserved, k=5):
    particles = meta["particle"].unique()
    r_means = {}
    for edge in conserved:
        r_vals = []
        for p in particles:
            sub = expr[meta["particle"] == p]
            i = genes.index(edge[0])
            j = genes.index(edge[1])
            r = np.corrcoef(sub.values[:, i], sub.values[:, j])[0, 1]
            r_vals.append(abs(r))
        r_means[edge] = float(np.mean(r_vals))
    order = sorted(conserved, key=lambda e: (-r_means[e], e[0], e[1]))
    return [f"{e[0]}-{e[1]}" for e in order[:k]]

def oma1_dele1_corr(expr):
    i = expr.columns.get_loc("OMA1")
    j = expr.columns.get_loc("DELE1")
    return float(np.corrcoef(expr.iloc[:, i], expr.iloc[:, j])[0, 1])

def module_scores(z, meta):
    modules = {
        "oxidative": OXI_GENES,
        "adaptation": ADAPT_GENES,
        "isr": ISR_GENES,
        "inflammation": INFLAM_GENES,
        "complex": COMPLEX_GENES
    }
    scores = pd.DataFrame(index=z.index)
    for name, gs in modules.items():
        scores[name] = z[gs].mean(axis=1)
    scores = scores.join(meta[["particle", "time_h"]])
    grouped = scores.groupby(["particle", "time_h"]).mean()
    return grouped

def classify_particles(grouped):
    result = {}
    for p in grouped.index.get_level_values(0).unique():
        sub = grouped.loc[p]
        O = sub["oxidative"]
        A = sub["adaptation"]
        I = sub["isr"]
        F = sub["inflammation"]
        C = sub["complex"]
        peak_time = O.idxmax()
        if peak_time == 6 and O[72] < 0.5 * O[6]:
            kinetics = "acute"
        elif peak_time == 72 and O[72] >= 0.8 * max(O[6], O[24]):
            kinetics = "chronic"
        else:
            kinetics = "mixed"
        if A[72] - A[6] >= 0.3 and I[72] < 0.0 and C[72] > 0.0:
            mito_state = "adaptation"
        elif I[72] > 0.5 or C[72] < -0.5:
            mito_state = "dysfunction"
        elif A[72] - A[6] >= 0.5 and I[72] > 0.0:
            mito_state = "biogenesis"
        else:
            mito_state = "other"
        isr_axis = "active" if I.mean() > 0.0 else "weak"
        inflammation = "high" if F[72] > 0.0 else "low"
        result[p] = {
            "kinetics": kinetics,
            "mito_state": mito_state,
            "isr_axis": isr_axis,
            "inflammation": inflammation
        }
    return result

def compute_gold(data_dir):
    expr, meta = load_data(data_dir)
    z = zscore_df(expr)
    pca, pcs = compute_pca(z)
    gold = {}
    gold["pca_explained_variance"] = [float(x) for x in pca.explained_variance_ratio_[:3]]
    gold["pc1_top_genes"] = top_loadings(pca, 0, list(z.columns), 5)
    gold["pc2_top_genes"] = top_loadings(pca, 1, list(z.columns), 5)
    gold["hierarchical_clusters"] = hclust(z, pcs)
    corr = gene_corr_matrix(z)
    gold["hub_genes"] = hub_genes(corr, list(z.columns), 0.7, 5)
    gold["oma1_dele1_correlation"] = oma1_dele1_corr(expr)
    conserved = conserved_edges(expr, meta, list(z.columns), 0.7)
    gold["conserved_network_edge_count"] = len(conserved)
    gold["conserved_network_top_edges"] = top_conserved_edges(expr, meta, list(z.columns), conserved, 5)
    grouped = module_scores(z, meta)
    gold["particle_response"] = classify_particles(grouped)
    conserved_genes = set()
    for e in conserved:
        conserved_genes.add(e[0])
        conserved_genes.add(e[1])
    gold["biomarkers"] = sorted([g for g in gold["hub_genes"] if g in conserved_genes])
    return gold

def load_agent_output():
    if not os.path.exists(AGENT_OUTPUT):
        return None
    with open(AGENT_OUTPUT) as f:
        return json.load(f)

def check_constraints(gold, agent):
    results = {}
    weights = {}
    if agent is None:
        results["output_exists"] = False
        weights["output_exists"] = 1
        return results, weights
    results["output_exists"] = True
    weights["output_exists"] = 1

    def close(a, b, tol):
        return abs(a - b) <= tol

    try:
        agent_pca = agent.get("pca_explained_variance", [])
        gold_pca = gold["pca_explained_variance"]
        ok = len(agent_pca) == 3 and all(close(agent_pca[i], gold_pca[i], 0.002) for i in range(3))
    except Exception:
        ok = False
    results["pca_variance"] = ok
    weights["pca_variance"] = 1

    try:
        ok = set(agent.get("pc1_top_genes", [])) == set(gold["pc1_top_genes"])
    except Exception:
        ok = False
    results["pc1_top_genes"] = ok
    weights["pc1_top_genes"] = 1

    try:
        ok = set(agent.get("pc2_top_genes", [])) == set(gold["pc2_top_genes"])
    except Exception:
        ok = False
    results["pc2_top_genes"] = ok
    weights["pc2_top_genes"] = 1

    try:
        agent_clusters = agent.get("hierarchical_clusters", {})
        gold_clusters = gold["hierarchical_clusters"]
        ok = len(agent_clusters) == len(gold_clusters) and all(agent_clusters.get(sid) == lab for sid, lab in gold_clusters.items())
    except Exception:
        ok = False
    results["hierarchical_clusters"] = ok
    weights["hierarchical_clusters"] = 1

    try:
        ok = agent.get("hub_genes", []) == gold["hub_genes"]
    except Exception:
        ok = False
    results["hub_genes"] = ok
    weights["hub_genes"] = 1

    try:
        ok = close(agent.get("oma1_dele1_correlation", 0.0), gold["oma1_dele1_correlation"], 0.01)
    except Exception:
        ok = False
    results["oma1_dele1_correlation"] = ok
    weights["oma1_dele1_correlation"] = 1

    try:
        ok = agent.get("conserved_network_edge_count") == gold["conserved_network_edge_count"]
    except Exception:
        ok = False
    results["conserved_network_edge_count"] = ok
    weights["conserved_network_edge_count"] = 1

    try:
        ok = set(agent.get("conserved_network_top_edges", [])) == set(gold["conserved_network_top_edges"])
    except Exception:
        ok = False
    results["conserved_network_top_edges"] = ok
    weights["conserved_network_top_edges"] = 1

    try:
        agent_pr = agent.get("particle_response", {})
        gold_pr = gold["particle_response"]
        ok = True
        for p in gold_pr:
            for k in gold_pr[p]:
                if agent_pr.get(p, {}).get(k) != gold_pr[p][k]:
                    ok = False
    except Exception:
        ok = False
    results["particle_response"] = ok
    weights["particle_response"] = 1

    try:
        ok = set(agent.get("biomarkers", [])) == set(gold["biomarkers"])
    except Exception:
        ok = False
    results["biomarkers"] = ok
    weights["biomarkers"] = 1

    return results, weights

def main():
    tmpdir = tempfile.mkdtemp()
    try:
        run_gold_gen(tmpdir)
        gold = compute_gold(tmpdir)
    finally:
        shutil.rmtree(tmpdir)
    agent = load_agent_output()
    results, weights = check_constraints(gold, agent)
    total_weight = sum(weights.values())
    passed_weight = sum(weights[k] for k in results if results[k])
    score = passed_weight / total_weight if total_weight > 0 else 0.0
    os.makedirs("/logs/verifier", exist_ok=True)
    with open(REWARD_PATH, "w") as f:
        json.dump({"score": score, "reward": score}, f)
    report = {
        "score": score,
        "passed": passed_weight,
        "total": total_weight,
        "constraints": results,
        "weights": weights
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
