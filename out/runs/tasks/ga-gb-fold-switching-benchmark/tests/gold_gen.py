import numpy as np
import pandas as pd

def generate_data():
    np.random.seed(42)
    aa_props = {
        'A': {'hydrophobicity': 1.8, 'volume': 88.6, 'helix': 1.42, 'beta': 0.83},
        'R': {'hydrophobicity': -4.5, 'volume': 173.4, 'helix': 0.98, 'beta': 0.93},
        'N': {'hydrophobicity': -3.5, 'volume': 114.1, 'helix': 0.67, 'beta': 0.89},
        'D': {'hydrophobicity': -3.5, 'volume': 111.1, 'helix': 1.01, 'beta': 0.54},
        'C': {'hydrophobicity': 2.5, 'volume': 108.5, 'helix': 0.70, 'beta': 1.19},
        'Q': {'hydrophobicity': -3.5, 'volume': 143.8, 'helix': 1.11, 'beta': 1.10},
        'E': {'hydrophobicity': -3.5, 'volume': 138.4, 'helix': 1.51, 'beta': 0.37},
        'G': {'hydrophobicity': -0.4, 'volume': 60.1, 'helix': 0.57, 'beta': 0.75},
        'H': {'hydrophobicity': -3.2, 'volume': 153.2, 'helix': 1.00, 'beta': 0.87},
        'I': {'hydrophobicity': 4.5, 'volume': 166.7, 'helix': 1.08, 'beta': 1.60},
        'L': {'hydrophobicity': 3.8, 'volume': 166.7, 'helix': 1.21, 'beta': 1.30},
        'K': {'hydrophobicity': -3.9, 'volume': 168.6, 'helix': 1.16, 'beta': 0.74},
        'M': {'hydrophobicity': 1.9, 'volume': 162.9, 'helix': 1.45, 'beta': 1.05},
        'F': {'hydrophobicity': 2.8, 'volume': 189.9, 'helix': 1.13, 'beta': 1.38},
        'P': {'hydrophobicity': -1.6, 'volume': 112.7, 'helix': 0.57, 'beta': 0.55},
        'S': {'hydrophobicity': -0.8, 'volume': 89.0, 'helix': 0.77, 'beta': 0.75},
        'T': {'hydrophobicity': -0.7, 'volume': 116.1, 'helix': 0.83, 'beta': 1.19},
        'W': {'hydrophobicity': -0.9, 'volume': 227.8, 'helix': 1.08, 'beta': 1.37},
        'Y': {'hydrophobicity': -1.3, 'volume': 193.6, 'helix': 0.69, 'beta': 1.47},
        'V': {'hydrophobicity': 4.2, 'volume': 140.0, 'helix': 1.06, 'beta': 1.70},
    }
    wt_seq = list(''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), size=56)))
    wt_seq[25] = 'L'
    wt_seq[32] = 'A'
    wt_seq[44] = 'V'
    wt_seq = ''.join(wt_seq)
    positions = [26, 33, 45]
    other_positions = [i for i in range(1, 57) if i not in positions]
    variants = []
    for pos in positions:
        wt_aa = wt_seq[pos-1]
        for mut_aa in 'ACDEFGHIKLMNPQRSTVWY':
            if mut_aa == wt_aa:
                continue
            variants.append({'position': pos, 'wt_aa': wt_aa, 'mut_aa': mut_aa})
    for pos in other_positions:
        wt_aa = wt_seq[pos-1]
        for _ in range(3):
            mut_aa = np.random.choice([aa for aa in 'ACDEFGHIKLMNPQRSTVWY' if aa != wt_aa])
            variants.append({'position': pos, 'wt_aa': wt_aa, 'mut_aa': mut_aa})
    while len(variants) < 300:
        pos = np.random.choice(other_positions)
        wt_aa = wt_seq[pos-1]
        mut_aa = np.random.choice([aa for aa in 'ACDEFGHIKLMNPQRSTVWY' if aa != wt_aa])
        variants.append({'position': pos, 'wt_aa': wt_aa, 'mut_aa': mut_aa})
    np.random.shuffle(variants)
    def get_props(aa):
        return aa_props[aa]
    data = []
    for i, v in enumerate(variants):
        pos = v['position']
        wt = v['wt_aa']
        mut = v['mut_aa']
        hp_wt = get_props(wt)['hydrophobicity']
        hp_mut = get_props(mut)['hydrophobicity']
        vol_wt = get_props(wt)['volume']
        vol_mut = get_props(mut)['volume']
        helix_wt = get_props(wt)['helix']
        helix_mut = get_props(mut)['helix']
        beta_wt = get_props(wt)['beta']
        beta_mut = get_props(mut)['beta']
        pos_eff = {26: 0.5, 33: -0.3, 45: 0.8}.get(pos, 0.0)
        s3 = 1.0*(helix_mut - helix_wt) + 0.5*(hp_mut - hp_wt) - 0.2*(beta_mut - beta_wt) + 0.3*pos_eff
        s4 = 1.0*(beta_mut - beta_wt) - 0.3*(hp_mut - hp_wt) + 0.2*(vol_mut - vol_wt)/100.0 - 0.4*pos_eff
        sm = 0.5*s3 + 0.5*s4 + 0.2*np.sin(pos) + 0.1*np.random.normal()
        su = -0.8*(s3+s4) + 0.5*abs(hp_mut - hp_wt) + 0.3*np.random.normal()
        scores = np.array([s3, s4, sm, su])
        scores = scores - scores.max()
        exp_scores = np.exp(scores / 0.8)
        probs = exp_scores / exp_scores.sum()
        probs = probs + np.random.normal(0, 0.03, 4)
        probs = np.clip(probs, 1e-6, None)
        probs = probs / probs.sum()
        row = {
            'variant_id': f'var_{i+1:04d}',
            'position': pos,
            'wt_aa': wt,
            'mut_aa': mut,
            'measured_p_3': probs[0],
            'measured_p_4b_plus': probs[1],
            'measured_p_mixed': probs[2],
            'measured_p_unfolded': probs[3],
        }
        methods = ['alphafold2', 'esmfold', 'rosetta', 'foldx', 'evfold', 'sparks']
        method_params = {
            'alphafold2': {'bias': [0.0, 0.0, 0.0, 0.0], 'noise': 0.05, 'pos_bias': {26: 0.0, 33: 0.0, 45: 0.0}, 'pos_noise_mult': {26: 1.0, 33: 1.5, 45: 0.8}},
            'esmfold': {'bias': [0.02, -0.02, 0.0, 0.0], 'noise': 0.07, 'pos_bias': {26: -0.02, 33: 0.03, 45: -0.01}, 'pos_noise_mult': {26: 0.5, 33: 1.0, 45: 1.2}},
            'rosetta': {'bias': [-0.03, 0.03, 0.0, 0.0], 'noise': 0.09, 'pos_bias': {26: 0.01, 33: -0.02, 45: 0.02}, 'pos_noise_mult': {26: 1.2, 33: 0.7, 45: 1.0}},
            'foldx': {'bias': [0.01, 0.01, -0.02, 0.0], 'noise': 0.10, 'pos_bias': {26: 0.0, 33: 0.0, 45: 0.0}, 'pos_noise_mult': {26: 1.0, 33: 1.0, 45: 1.0}},
            'evfold': {'bias': [-0.01, 0.0, 0.01, 0.0], 'noise': 0.11, 'pos_bias': {26: 0.02, 33: 0.01, 45: -0.02}, 'pos_noise_mult': {26: 0.8, 33: 1.3, 45: 0.9}},
            'sparks': {'bias': [0.0, -0.01, 0.02, -0.01], 'noise': 0.12, 'pos_bias': {26: -0.01, 33: -0.01, 45: 0.01}, 'pos_noise_mult': {26: 1.5, 33: 0.8, 45: 1.4}},
        }
        for method in methods:
            params = method_params[method]
            pred = probs.copy()
            pred += params['bias']
            pred += params['pos_bias'].get(pos, 0.0) * np.array([1, -1, 0.5, -0.5])
            noise_scale = params['noise'] * params['pos_noise_mult'].get(pos, 1.0)
            pred += np.random.normal(0, noise_scale, 4)
            pred = np.clip(pred, 1e-6, None)
            pred = pred / pred.sum()
            if np.random.random() < 0.05:
                pred = np.array([np.nan, np.nan, np.nan, np.nan])
            row[f'{method}_p_3'] = pred[0]
            row[f'{method}_p_4b_plus'] = pred[1]
            row[f'{method}_p_mixed'] = pred[2]
            row[f'{method}_p_unfolded'] = pred[3]
        data.append(row)
    df = pd.DataFrame(data)
    return df
