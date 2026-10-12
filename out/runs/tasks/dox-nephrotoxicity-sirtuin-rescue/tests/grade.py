import json
import os
import sys
import math
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

DATA_DIR = '/app/data'
OUTPUT_FILE = '/app/output/results.json'
REWARD_FILE = '/logs/verifier/reward.json'
REPORT_FILE = '/logs/verifier/report.json'

def load_agent_output():
    if not os.path.exists(OUTPUT_FILE):
        return None
    try:
        with open(OUTPUT_FILE) as f:
            return json.load(f)
    except Exception:
        return None

def compute_gold():
    rna = pd.read_csv(os.path.join(DATA_DIR, 'rna_counts.csv'), index_col=0)
    prot = pd.read_csv(os.path.join(DATA_DIR, 'proteomics.csv'), index_col=0)
    seahorse = pd.read_csv(os.path.join(DATA_DIR, 'seahorse_ocr.csv'))
    sim = pd.read_csv(os.path.join(DATA_DIR, 'sim_morphology.csv'))
    sod2 = pd.read_csv(os.path.join(DATA_DIR, 'sod2_acetyl.csv'))
    with open(os.path.join(DATA_DIR, 'gene_sets.json')) as f:
        gene_sets = json.load(f)
    samples = rna.columns.tolist()
    groups = [s.split('_')[0] for s in samples]
    group_indices = {g: [i for i, s in enumerate(samples) if s.startswith(g)] for g in set(groups)}
    group_names = ['Control', 'DOX', 'DOX+RSV', 'DOX+NMN']
    total_counts = rna.sum(axis=0)
    cpm = rna.div(total_counts, axis=1) * 1e6
    keep_genes = (cpm >= 1).sum(axis=1) >= 3
    rna_filtered = rna.loc[keep_genes]
    cpm_filtered = cpm.loc[keep_genes]
    rna_genes_after_filter = int(rna_filtered.shape[0])
    keep_prot = (prot > 0).sum(axis=1) >= 3
    prot_filtered = prot.loc[keep_prot]
    proteins_after_filter = int(prot_filtered.shape[0])
    log_cpm = np.log2(cpm_filtered + 1)
    dox_idx = group_indices['DOX']
    ctrl_idx = group_indices['Control']
    log2fc = log_cpm.iloc[:, dox_idx].mean(axis=1) - log_cpm.iloc[:, ctrl_idx].mean(axis=1)
    pvals = []
    for gene in log_cpm.index:
        dox_vals = log_cpm.loc[gene, log_cpm.columns[dox_idx]]
        ctrl_vals = log_cpm.loc[gene, log_cpm.columns[ctrl_idx]]
        t, p = stats.ttest_ind(dox_vals, ctrl_vals, equal_var=False)
        pvals.append(p)
    pvals = np.array(pvals)
    _, fdr, _, _ = multipletests(pvals, method='fdr_bh')
    sig = (fdr < 0.05) & (np.abs(log2fc) > 1.0)
    sig_genes = log2fc[sig].index.tolist()
    oxphos_set = set(gene_sets['OXPHOS'])
    inflam_set = set(gene_sets['Inflammation'])
    down_oxphos = [g for g in sig_genes if g in oxphos_set and log2fc[g] < 0]
    up_inflam = [g for g in sig_genes if g in inflam_set and log2fc[g] > 0]
    num_down_OXPHOS = len(down_oxphos)
    num_up_inflammation = len(up_inflam)
    sig_log2fc = log2fc[sig]
    top10_down = sig_log2fc.sort_values(ascending=True).head(10).index.tolist()
    top10_up = sig_log2fc.sort_values(ascending=False).head(10).index.tolist()
    z_scores = log_cpm.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0, axis=1)
    pathway_scores = {}
    for pathway in ['OXPHOS', 'FAO']:
        genes = [g for g in gene_sets[pathway] if g in z_scores.index]
        if len(genes) == 0:
            pathway_scores[pathway] = {g: 0.0 for g in group_names}
            continue
        sample_scores = z_scores.loc[genes].mean(axis=0)
        grp_means = {}
        for g in group_names:
            idx = group_indices[g]
            grp_means[g] = float(sample_scores.iloc[idx].mean())
        pathway_scores[pathway] = grp_means
    basal_window = (seahorse['time'] >= 0) & (seahorse['time'] <= 18)
    oligo_window = (seahorse['time'] >= 21) & (seahorse['time'] <= 38)
    fccp_window = (seahorse['time'] >= 42) & (seahorse['time'] <= 58)
    non_mito_window = (seahorse['time'] >= 63) & (seahorse['time'] <= 90)
    well_metrics = {}
    for well in seahorse['well'].unique():
        well_data = seahorse[seahorse['well'] == well]
        non_mito = well_data[non_mito_window]['OCR'].mean()
        basal = well_data[basal_window]['OCR'].mean() - non_mito
        oligo = well_data[oligo_window]['OCR'].mean() - non_mito
        maximal = well_data[fccp_window]['OCR'].mean() - non_mito
        spare = maximal - basal
        atp = basal - oligo
        well_metrics[well] = {'basal': float(basal), 'maximal': float(maximal), 'spare': float(spare), 'ATP': float(atp), 'non_mito': float(non_mito)}
    seahorse_metrics = {}
    for metric in ['basal', 'maximal', 'spare', 'ATP', 'non_mito']:
        seahorse_metrics[metric] = {}
        for g in group_names:
            wells_g = seahorse[seahorse['group'] == g]['well'].unique()
            vals = [well_metrics[w][metric] for w in wells_g]
            seahorse_metrics[metric][g] = float(np.mean(vals))
    sim['fragmented'] = (sim['aspect_ratio'] < 2.0) & (sim['form_factor'] < 0.5)
    sim_percent = {}
    for g in group_names:
        grp_data = sim[sim['group'] == g]
        sim_percent[g] = float(100 * grp_data['fragmented'].mean())
    sod2['ratio'] = sod2['acetylated_SOD2'] / sod2['total_SOD2']
    sod2_ratio = {}
    for g in group_names:
        grp_data = sod2[sod2['group'] == g]
        sod2_ratio[g] = float(grp_data['ratio'].mean())
    metrics = {
        'OXPHOS': (pathway_scores['OXPHOS'], True),
        'FAO': (pathway_scores['FAO'], True),
        'basal': (seahorse_metrics['basal'], True),
        'maximal': (seahorse_metrics['maximal'], True),
        'spare': (seahorse_metrics['spare'], True),
        'ATP': (seahorse_metrics['ATP'], True),
        'percent_non_fragmented': ({g: 100 - sim_percent[g] for g in group_names}, True),
        'sod2_acetylation': (sod2_ratio, False)
    }
    rescue_scores = {}
    for treatment in ['DOX+RSV', 'DOX+NMN']:
        score = 0.0
        for name, (values, higher_better) in metrics.items():
            ctrl = values['Control']
            dox = values['DOX']
            treat = values[treatment]
            if higher_better:
                if abs(ctrl - dox) > 1e-9:
                    score += (treat - dox) / (ctrl - dox)
            else:
                if abs(dox - ctrl) > 1e-9:
                    score += (dox - treat) / (dox - ctrl)
        rescue_scores[treatment] = float(score)
    gold = {
        'qc': {'rna_genes_after_filter': rna_genes_after_filter, 'proteins_after_filter': proteins_after_filter},
        'differential_expression': {'num_down_OXPHOS': num_down_OXPHOS, 'num_up_inflammation': num_up_inflammation, 'top10_down': top10_down, 'top10_up': top10_up},
        'pathway_scores': {'OXPHOS': pathway_scores['OXPHOS'], 'FAO': pathway_scores['FAO']},
        'seahorse': seahorse_metrics,
        'sim_morphology': {'percent_fragmented': sim_percent},
        'sod2_acetylation': {'ratio': sod2_ratio},
        'rescue_scores': rescue_scores
    }
    return gold

def compare(agent, gold):
    report = []
    if agent is None:
        return 0.0, ['Agent output missing or invalid JSON']
    constraints = []
    try:
        rna_ok = agent['qc']['rna_genes_after_filter'] == gold['qc']['rna_genes_after_filter']
        prot_ok = agent['qc']['proteins_after_filter'] == gold['qc']['proteins_after_filter']
        if rna_ok and prot_ok:
            constraints.append(1.0)
            report.append('QC counts: PASS')
        else:
            constraints.append(0.0)
            report.append(f"QC counts: FAIL (agent rna={agent['qc']['rna_genes_after_filter']}, gold={gold['qc']['rna_genes_after_filter']}; agent prot={agent['qc']['proteins_after_filter']}, gold={gold['qc']['proteins_after_filter']})")
    except (KeyError, TypeError) as e:
        constraints.append(0.0)
        report.append(f'QC counts: FAIL (missing key: {e})')
    try:
        de = agent['differential_expression']
        num_ok = de['num_down_OXPHOS'] == gold['differential_expression']['num_down_OXPHOS'] and de['num_up_inflammation'] == gold['differential_expression']['num_up_inflammation']
        if num_ok:
            constraints.append(1.0)
            report.append('DE counts: PASS')
        else:
            constraints.append(0.0)
            report.append(f"DE counts: FAIL (agent down={de.get('num_down_OXPHOS')}, gold={gold['differential_expression']['num_down_OXPHOS']}; agent up={de.get('num_up_inflammation')}, gold={gold['differential_expression']['num_up_inflammation']})")
    except (KeyError, TypeError) as e:
        constraints.append(0.0)
        report.append(f'DE counts: FAIL (missing key: {e})')
    try:
        de = agent['differential_expression']
        down_ok = de['top10_down'] == gold['differential_expression']['top10_down']
        up_ok = de['top10_up'] == gold['differential_expression']['top10_up']
        if down_ok and up_ok:
            constraints.append(1.0)
            report.append('Top10 lists: PASS')
        else:
            constraints.append(0.0)
            report.append(f"Top10 lists: FAIL (agent down={de.get('top10_down')}, gold={gold['differential_expression']['top10_down']}; agent up={de.get('top10_up')}, gold={gold['differential_expression']['top10_up']})")
    except (KeyError, TypeError) as e:
        constraints.append(0.0)
        report.append(f'Top10 lists: FAIL (missing key: {e})')
    try:
        ok = True
        for pathway in ['OXPHOS', 'FAO']:
            for g in gold['pathway_scores'][pathway]:
                a = agent['pathway_scores'][pathway][g]
                b = gold['pathway_scores'][pathway][g]
                if abs(a - b) > 0.05:
                    ok = False
                    report.append(f'Pathway {pathway} {g}: agent={a:.4f}, gold={b:.4f}')
        if ok:
            constraints.append(1.0)
            report.append('Pathway scores: PASS')
        else:
            constraints.append(0.0)
            report.append('Pathway scores: FAIL')
    except (KeyError, TypeError) as e:
        constraints.append(0.0)
        report.append(f'Pathway scores: FAIL (missing key: {e})')
    try:
        ok = True
        for metric in ['basal', 'maximal', 'spare', 'ATP', 'non_mito']:
            for g in gold['seahorse'][metric]:
                a = agent['seahorse'][metric][g]
                b = gold['seahorse'][metric][g]
                if abs(a - b) > 5.0:
                    ok = False
                    report.append(f'Seahorse {metric} {g}: agent={a:.2f}, gold={b:.2f}')
        if ok:
            constraints.append(1.0)
            report.append('Seahorse metrics: PASS')
        else:
            constraints.append(0.0)
            report.append('Seahorse metrics: FAIL')
    except (KeyError, TypeError) as e:
        constraints.append(0.0)
        report.append(f'Seahorse metrics: FAIL (missing key: {e})')
    try:
        ok = True
        for g in gold['sim_morphology']['percent_fragmented']:
            a = agent['sim_morphology']['percent_fragmented'][g]
            b = gold['sim_morphology']['percent_fragmented'][g]
            if abs(a - b) > 3.0:
                ok = False
                report.append(f'SIM {g}: agent={a:.2f}, gold={b:.2f}')
        if ok:
            constraints.append(1.0)
            report.append('SIM percent fragmented: PASS')
        else:
            constraints.append(0.0)
            report.append('SIM percent fragmented: FAIL')
    except (KeyError, TypeError) as e:
        constraints.append(0.0)
        report.append(f'SIM percent fragmented: FAIL (missing key: {e})')
    try:
        ok = True
        for g in gold['sod2_acetylation']['ratio']:
            a = agent['sod2_acetylation']['ratio'][g]
            b = gold['sod2_acetylation']['ratio'][g]
            if abs(a - b) > 0.05:
                ok = False
                report.append(f'SOD2 {g}: agent={a:.4f}, gold={b:.4f}')
        if ok:
            constraints.append(1.0)
            report.append('SOD2 acetylation ratio: PASS')
        else:
            constraints.append(0.0)
            report.append('SOD2 acetylation ratio: FAIL')
    except (KeyError, TypeError) as e:
        constraints.append(0.0)
        report.append(f'SOD2 acetylation ratio: FAIL (missing key: {e})')
    try:
        ok = True
        for t in ['DOX+RSV', 'DOX+NMN']:
            a = agent['rescue_scores'][t]
            b = gold['rescue_scores'][t]
            if abs(a - b) > 0.1:
                ok = False
                report.append(f'Rescue {t}: agent={a:.4f}, gold={b:.4f}')
        if ok:
            constraints.append(1.0)
            report.append('Rescue scores: PASS')
        else:
            constraints.append(0.0)
            report.append('Rescue scores: FAIL')
    except (KeyError, TypeError) as e:
        constraints.append(0.0)
        report.append(f'Rescue scores: FAIL (missing key: {e})')
    score = sum(constraints) / len(constraints)
    return score, report

def main():
    agent = load_agent_output()
    gold = compute_gold()
    score, report = compare(agent, gold)
    os.makedirs(os.path.dirname(REWARD_FILE), exist_ok=True)
    with open(REWARD_FILE, 'w') as f:
        json.dump({'score': score, 'reward': score}, f)
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, 'w') as f:
        json.dump({'score': score, 'report': report}, f, indent=2)
    print('\n'.join(report))
    print(f'Final score: {score}')

if __name__ == '__main__':
    main()
