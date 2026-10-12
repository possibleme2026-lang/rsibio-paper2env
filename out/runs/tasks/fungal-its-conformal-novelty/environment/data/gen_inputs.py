import numpy as np
import pandas as pd
import os

def generate_all(seed=20260729):
    rng = np.random.default_rng(seed)
    regions = ['ITS1', 'ITS2']
    methods = ['alignment', 'kmer', 'neural_cnn', 'neural_mlp']
    method_params = {
        'alignment': (50, 2),
        'kmer': (30, 3),
        'neural_cnn': (20, 3),
        'neural_mlp': (15, 4),
    }
    novel_params = {
        'alignment': (12, 8),
        'kmer': (10, 6),
        'neural_cnn': (8, 6),
        'neural_mlp': (7, 7),
    }
    n_cal = 500
    n_val = 500
    n_query = 2000
    n_genera = 100
    genera = [f'G{i:03d}' for i in range(n_genera)]
    novel_genera = [f'N{i:03d}' for i in range(40)]
    
    ref_rows = []
    for g in genera:
        for region in regions:
            seq = ''.join(rng.choice(list('ACGT'), size=500))
            ref_rows.append({'ref_id': f'{g}_{region}', 'genus': g, 'region': region, 'sequence': seq})
    ref_df = pd.DataFrame(ref_rows)
    
    cal_rows = []
    for region in regions:
        for method in methods:
            a, b = method_params[method]
            scores = rng.beta(a, b, size=n_cal)
            for i, s in enumerate(scores):
                cal_rows.append({'cal_id': f'CAL_{region}_{method}_{i:04d}', 'region': region, 'method': method, 'score': s, 'genus': rng.choice(genera)})
    cal_df = pd.DataFrame(cal_rows)
    
    val_rows = []
    for region in regions:
        for method in methods:
            a, b = method_params[method]
            scores = rng.beta(a, b, size=n_val)
            for i, s in enumerate(scores):
                val_rows.append({'val_id': f'VAL_{region}_{method}_{i:04d}', 'region': region, 'method': method, 'score': s, 'genus': rng.choice(genera)})
    val_df = pd.DataFrame(val_rows)
    
    query_labels = {}
    for i in range(n_query):
        qid = f'Q{i:04d}'
        if rng.random() < 0.8:
            query_labels[qid] = rng.choice(genera)
        else:
            query_labels[qid] = rng.choice(novel_genera)
    
    query_rows = []
    for qid, true_genus in query_labels.items():
        is_novel = true_genus.startswith('N')
        for region in regions:
            for method in methods:
                if is_novel:
                    a, b = novel_params[method]
                else:
                    a, b = method_params[method]
                s = rng.beta(a, b)
                query_rows.append({'query_id': qid, 'region': region, 'method': method, 'score': s})
    query_df = pd.DataFrame(query_rows)
    
    soil_rows = []
    for i in range(500):
        asv_id = f'ASV{i:04d}'
        is_novel = rng.random() < 0.2
        if is_novel:
            a, b = novel_params['alignment']
            abundance = int(np.round(np.exp(rng.normal(2, 1.5))))
        else:
            a, b = method_params['alignment']
            abundance = int(np.round(np.exp(rng.normal(3, 1.5))))
        s = rng.beta(a, b)
        soil_rows.append({'asv_id': asv_id, 'region': 'ITS1', 'method': 'alignment', 'score': s, 'abundance': abundance})
    soil_df = pd.DataFrame(soil_rows)
    
    return ref_df, cal_df, val_df, query_df, query_labels, soil_df

if __name__ == '__main__':
    ref_df, cal_df, val_df, query_df, query_labels, soil_df = generate_all(seed=20260729)
    os.makedirs('/app/data', exist_ok=True)
    ref_df.to_csv('/app/data/reference_db.tsv', sep='\t', index=False)
    cal_df.to_csv('/app/data/calibration_scores.tsv', sep='\t', index=False)
    val_df.to_csv('/app/data/validation_scores.tsv', sep='\t', index=False)
    query_df.to_csv('/app/data/query_scores.tsv', sep='\t', index=False)
    soil_df.to_csv('/app/data/soil_asvs.tsv', sep='\t', index=False)
    print('Data generated.')
