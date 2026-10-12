import sys, os, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
sys.path.insert(0, '/tests')
from gold_gen import generate_gold

def main():
    true_image_df, true_ril_df, true_qtl_list, marker_df, geno_df = generate_gold()
    output_dir = '/app/output'
    constraints = []
    def add_constraint(name, passed, msg):
        constraints.append({'name': name, 'passed': bool(passed), 'message': msg})
    # Check traits.csv
    traits_path = os.path.join(output_dir, 'traits.csv')
    if os.path.exists(traits_path):
        try:
            agent_traits = pd.read_csv(traits_path)
            required_cols = ['image_id', 'ril_id', 'stomatal_density', 'stomatal_size', 'vein_density', 'vein_thickness', 'nnd_mean', 'nnd_cv']
            if all(c in agent_traits.columns for c in required_cols):
                add_constraint('traits_format', True, 'OK')
            else:
                add_constraint('traits_format', False, f'Missing columns. Found: {list(agent_traits.columns)}')
        except Exception as e:
            add_constraint('traits_format', False, f'Failed to read: {e}')
            agent_traits = None
    else:
        add_constraint('traits_format', False, 'File not found')
        agent_traits = None
    # Check ril_traits.csv
    ril_path = os.path.join(output_dir, 'ril_traits.csv')
    if os.path.exists(ril_path):
        try:
            agent_ril = pd.read_csv(ril_path)
            required_cols = ['ril_id', 'stomatal_density', 'stomatal_size', 'vein_density', 'vein_thickness', 'nnd_mean', 'nnd_cv']
            if all(c in agent_ril.columns for c in required_cols):
                add_constraint('ril_traits_format', True, 'OK')
            else:
                add_constraint('ril_traits_format', False, f'Missing columns. Found: {list(agent_ril.columns)}')
        except Exception as e:
            add_constraint('ril_traits_format', False, f'Failed to read: {e}')
            agent_ril = None
    else:
        add_constraint('ril_traits_format', False, 'File not found')
        agent_ril = None
    # Check qtl_results.json
    qtl_path = os.path.join(output_dir, 'qtl_results.json')
    if os.path.exists(qtl_path):
        try:
            with open(qtl_path) as f:
                agent_qtl = json.load(f)
            if isinstance(agent_qtl, list) and all(isinstance(x, dict) and 'trait' in x and 'marker_id' in x and 'chromosome' in x and 'position' in x and 'lod' in x for x in agent_qtl):
                add_constraint('qtl_format', True, 'OK')
            else:
                add_constraint('qtl_format', False, 'Invalid JSON structure')
        except Exception as e:
            add_constraint('qtl_format', False, f'Failed to read: {e}')
            agent_qtl = None
    else:
        add_constraint('qtl_format', False, 'File not found')
        agent_qtl = None

    if agent_traits is not None:
        merged = pd.merge(agent_traits, true_image_df, on='image_id', suffixes=('_pred', '_true'))
        if len(merged) == 0:
            for name in ['stomatal_density_mape', 'vein_density_mape', 'nnd_cv_mape', 'stomatal_size_mape', 'vein_thickness_mape', 'nnd_mean_mape']:
                add_constraint(name, False, 'No matching image_id')
        else:
            def mape(pred, true):
                return float(np.mean(np.abs(pred - true) / np.maximum(np.abs(true), 1e-9)) * 100)
            sd_mape = mape(merged['stomatal_density_pred'], merged['stomatal_density_true'])
            add_constraint('stomatal_density_mape', sd_mape < 15.0, f'MAPE = {sd_mape:.2f}%')
            vd_mape = mape(merged['vein_density_pred'], merged['vein_density_true'])
            add_constraint('vein_density_mape', vd_mape < 10.0, f'MAPE = {vd_mape:.2f}%')
            nnd_cv_mape = mape(merged['nnd_cv_pred'], merged['nnd_cv_true'])
            add_constraint('nnd_cv_mape', nnd_cv_mape < 20.0, f'MAPE = {nnd_cv_mape:.2f}%')
            ss_mape = mape(merged['stomatal_size_pred'], merged['stomatal_size_true'])
            add_constraint('stomatal_size_mape', ss_mape < 15.0, f'MAPE = {ss_mape:.2f}%')
            vt_mape = mape(merged['vein_thickness_pred'], merged['vein_thickness_true'])
            add_constraint('vein_thickness_mape', vt_mape < 15.0, f'MAPE = {vt_mape:.2f}%')
            nnd_mean_mape = mape(merged['nnd_mean_pred'], merged['nnd_mean_true'])
            add_constraint('nnd_mean_mape', nnd_mean_mape < 15.0, f'MAPE = {nnd_mean_mape:.2f}%')
    else:
        for name in ['stomatal_density_mape', 'vein_density_mape', 'nnd_cv_mape', 'stomatal_size_mape', 'vein_thickness_mape', 'nnd_mean_mape']:
            add_constraint(name, False, 'traits.csv missing or invalid')

    if agent_ril is not None:
        merged_ril = pd.merge(agent_ril, true_ril_df, on='ril_id', suffixes=('_pred', '_true'))
        if len(merged_ril) == 0:
            add_constraint('ril_correlation', False, 'No matching ril_id')
        else:
            traits = ['stomatal_density', 'stomatal_size', 'vein_density', 'vein_thickness', 'nnd_mean', 'nnd_cv']
            cors = {}
            for t in traits:
                if t + '_pred' in merged_ril.columns and t + '_true' in merged_ril.columns:
                    pred = merged_ril[t + '_pred'].values
                    true = merged_ril[t + '_true'].values
                    if len(pred) > 1 and np.std(pred) > 0 and np.std(true) > 0:
                        r, _ = pearsonr(pred, true)
                        cors[t] = r
                    else:
                        cors[t] = 0.0
                else:
                    cors[t] = 0.0
            min_cor = min(cors.values()) if cors else 0.0
            add_constraint('ril_correlation', min_cor > 0.8, f'Min correlation = {min_cor:.3f} across {cors}')
    else:
        add_constraint('ril_correlation', False, 'ril_traits.csv missing or invalid')

    if agent_qtl is not None:
        true_by_trait = {}
        for q in true_qtl_list:
            true_by_trait.setdefault(q['trait'], []).append(q)
        for trait in ['stomatal_density', 'vein_density', 'nnd_cv']:
            true_qs = true_by_trait.get(trait, [])
            agent_qs = [q for q in agent_qtl if q.get('trait') == trait and q.get('lod', 0) > 3.0]
            found = False
            for aq in agent_qs:
                for tq in true_qs:
                    if aq.get('chromosome') == tq['chromosome'] and abs(aq.get('position', 0) - tq['position']) < 20.0:
                        found = True
                        break
                if found:
                    break
            add_constraint(f'qtl_{trait}', found, f'Found={found}, agent QTLs: {len(agent_qs)}, true: {len(true_qs)}')
    else:
        for trait in ['stomatal_density', 'vein_density', 'nnd_cv']:
            add_constraint(f'qtl_{trait}', False, 'qtl_results.json missing or invalid')

    passed = sum(1 for c in constraints if c['passed'])
    total = len(constraints)
    score = passed / total if total > 0 else 0.0
    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/report.json', 'w') as f:
        json.dump({'score': score, 'constraints': constraints}, f, indent=2)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({'score': score, 'reward': score}, f)

if __name__ == '__main__':
    main()
