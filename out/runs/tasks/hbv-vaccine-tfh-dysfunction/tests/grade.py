import json, os, sys, math
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, '/tests')
import gold_gen

def load_agent_output():
    ap = '/app/results/analysis.json'
    dp = '/app/results/de_genes.csv'
    if not os.path.exists(ap) or not os.path.exists(dp):
        return None, None
    with open(ap) as f:
        analysis = json.load(f)
    de = pd.read_csv(dp)
    return analysis, de

def check_structure(analysis):
    if not isinstance(analysis, dict):
        return False, 'analysis.json is not a dict'
    if 'donors' not in analysis or 'correlations' not in analysis:
        return False, 'missing top-level keys'
    donors = analysis['donors']
    if len(donors) != 10:
        return False, f'expected 10 donors, got {len(donors)}'
    required = ['responder_group', 'end_of_study_titer_mIU_mL', 'n_tfh', 'n_gng4_tfh',
                'gng4_tfh_signature_score', 'inflammatory_score', 'bcl6_regulon_activity',
                'bcl6_regulon_specificity', 'dz_score', 'lz_score', 'myc_score', 'mtorc1_score',
                'mean_shm', 'clonotype_diversity', 'public_clonotypes']
    for d, entry in donors.items():
        for k in required:
            if k not in entry:
                return False, f'donor {d} missing {k}'
    for k in ['gng4_score_vs_titer', 'shm_vs_titer', 'inflammatory_vs_titer']:
        if k not in analysis['correlations']:
            return False, f'correlations missing {k}'
    return True, 'structure OK'

def check_donor_field(analysis, ref, field, rtol=0.02, atol=0.01):
    for d in ref['donors']:
        a = analysis['donors'].get(d, {}).get(field)
        r = ref['donors'][d][field]
        if a is None:
            return False, f'donor {d} missing {field}'
        if not math.isclose(a, r, rel_tol=rtol, abs_tol=atol):
            return False, f'donor {d} {field}: got {a}, expected {r}'
    return True, f'{field} matches reference'

def check_donor_fields(analysis, ref, fields, rtol=0.02, atol=0.01):
    for d in ref['donors']:
        for f in fields:
            a = analysis['donors'].get(d, {}).get(f)
            r = ref['donors'][d][f]
            if a is None:
                return False, f'donor {d} missing {f}'
            if not math.isclose(a, r, rel_tol=rtol, abs_tol=atol):
                return False, f'donor {d} {f}: got {a}, expected {r}'
    return True, f'{fields} match reference'

def check_public_clonotypes(analysis, ref):
    for d in ref['donors']:
        a = analysis['donors'].get(d, {}).get('public_clonotypes')
        r = ref['donors'][d]['public_clonotypes']
        if a is None:
            return False, f'donor {d} missing public_clonotypes'
        if abs(a - r) > 1:
            return False, f'donor {d} public_clonotypes: got {a}, expected {r}'
    return True, 'public_clonotypes within ±1'

def check_de_genes(de, ref):
    cols = ['gene', 'mean_high', 'mean_low', 'log2_fold_change', 'p_value', 'padj', 'direction']
    if not all(c in de.columns for c in cols):
        return False, 'de_genes.csv missing columns'
    if len(de) < 250:
        return False, f'de_genes.csv has only {len(de)} genes'
    ref_de = pd.DataFrame(ref['de_genes'])
    ref_up = set(ref_de[ref_de['direction'] == 'up_in_high'].sort_values('p_value').head(20)['gene'])
    agent_up = set(de[de['direction'] == 'up_in_high'].sort_values('p_value').head(20)['gene'])
    inter = ref_up & agent_up
    f1 = 2 * len(inter) / (len(ref_up) + len(agent_up)) if (len(ref_up) + len(agent_up)) > 0 else 0.0
    if f1 < 0.7:
        return False, f'top 20 up F1={f1:.2f} (ref: {len(ref_up)}, agent: {len(agent_up)}, inter: {len(inter)})'
    return True, f'top 20 up F1={f1:.2f}'

def check_correlations(analysis, ref):
    for k in ['gng4_score_vs_titer', 'shm_vs_titer', 'inflammatory_vs_titer']:
        a = analysis['correlations'].get(k)
        r = ref['correlations'][k]
        if a is None:
            return False, f'correlations missing {k}'
        if not math.isclose(a, r, rel_tol=0.1, abs_tol=0.05):
            return False, f'{k}: got {a}, expected {r} (tol 0.05)'
    if analysis['correlations']['gng4_score_vs_titer'] <= 0.5:
        return False, f'gng4_score_vs_titer too low: {analysis["correlations"]["gng4_score_vs_titer"]}'
    if analysis['correlations']['inflammatory_vs_titer'] >= -0.4:
        return False, f'inflammatory_vs_titer not negative enough: {analysis["correlations"]["inflammatory_vs_titer"]}'
    return True, 'correlations match and have expected signs'

def check_group_differences(analysis):
    donors = analysis['donors']
    high = [donors[d] for d in donors if donors[d]['responder_group'] == 'High']
    low = [donors[d] for d in donors if donors[d]['responder_group'] == 'Low']
    if len(high) < 2 or len(low) < 2:
        return False, 'not enough donors per group'
    def mwu(field):
        h = [x[field] for x in high]
        l = [x[field] for x in low]
        stat, p = mannwhitneyu(h, l, alternative='two-sided')
        return p
    if not (np.mean([x['gng4_tfh_signature_score'] for x in high]) > np.mean([x['gng4_tfh_signature_score'] for x in low])):
        return False, 'GNG4 score not higher in High'
    if mwu('gng4_tfh_signature_score') >= 0.05:
        return False, f'GNG4 score MWU p={mwu("gng4_tfh_signature_score"):.3f} not <0.05'
    if not (np.mean([x['inflammatory_score'] for x in low]) > np.mean([x['inflammatory_score'] for x in high])):
        return False, 'inflammatory score not higher in Low'
    if mwu('inflammatory_score') >= 0.05:
        return False, f'inflammatory score MWU p={mwu("inflammatory_score"):.3f} not <0.05'
    if not (np.mean([x['bcl6_regulon_specificity'] for x in high]) > np.mean([x['bcl6_regulon_specificity'] for x in low])):
        return False, 'BCL6 specificity not higher in High'
    for f in ['dz_score', 'lz_score', 'myc_score', 'mtorc1_score']:
        if not (np.mean([x[f] for x in high]) > np.mean([x[f] for x in low])):
            return False, f'{f} not higher in High'
        if mwu(f) >= 0.05:
            return False, f'{f} MWU p={mwu(f):.3f} not <0.05'
    return True, 'all expected group differences present'

def main():
    tempdir = '/tmp/gold_data'
    os.makedirs(tempdir, exist_ok=True)
    gold_gen.generate_data(tempdir, seed=42)
    ref = gold_gen.reference_analysis(tempdir)
    analysis, de = load_agent_output()
    if analysis is None or de is None:
        result = {'score': 0.0, 'reward': 0.0}
        os.makedirs('/logs/verifier', exist_ok=True)
        with open('/logs/verifier/reward.json', 'w') as f:
            json.dump(result, f)
        print('FAIL: missing output files')
        return
    checks = []
    checks.append(check_structure(analysis))
    checks.append(check_donor_field(analysis, ref, 'gng4_tfh_signature_score', rtol=0.02, atol=0.01))
    checks.append(check_donor_field(analysis, ref, 'inflammatory_score', rtol=0.02, atol=0.01))
    checks.append(check_donor_field(analysis, ref, 'bcl6_regulon_specificity', rtol=0.05, atol=0.02))
    checks.append(check_donor_fields(analysis, ref, ['dz_score', 'lz_score', 'myc_score', 'mtorc1_score'], rtol=0.02, atol=0.01))
    checks.append(check_donor_fields(analysis, ref, ['mean_shm', 'clonotype_diversity'], rtol=0.05, atol=0.01))
    checks.append(check_public_clonotypes(analysis, ref))
    checks.append(check_de_genes(de, ref))
    checks.append(check_correlations(analysis, ref))
    checks.append(check_group_differences(analysis))
    score = sum(1 for ok, msg in checks if ok) / len(checks)
    result = {'score': score, 'reward': score}
    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump(result, f)
    report = {f'check_{i+1}': {'passed': ok, 'message': msg} for i, (ok, msg) in enumerate(checks)}
    with open('/logs/verifier/report.json', 'w') as f:
        json.dump(report, f, indent=2)
    for i, (ok, msg) in enumerate(checks):
        print(f"Check {i+1}: {'PASS' if ok else 'FAIL'} - {msg}")
    print(f'Score: {score}')

if __name__ == '__main__':
    main()
