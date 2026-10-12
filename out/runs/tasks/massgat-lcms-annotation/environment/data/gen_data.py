import argparse
import csv
import math
import random

ADDUCTS = {
    'M+H': 1.007276,
    'M+Na': 22.989218,
    'M+NH4': 18.033823,
    'M+K': 38.963158,
}
ISOTOPES = {
    'iso0': 0.0,
    'iso1': 1.003355,
    'iso2': 2.006710,
}

def generate(seed, n_compounds, n_noise, out_features, out_labels):
    rng = random.Random(seed)
    features = []
    labels = []
    fid = 1
    for cid in range(1, n_compounds + 1):
        M = rng.uniform(100.0, 800.0)
        rt = rng.uniform(0.5, 12.0)
        n_adducts = rng.randint(1, 4)
        adduct_names = rng.sample(list(ADDUCTS.keys()), n_adducts)
        base_intensity = rng.lognormvariate(math.log(1e5), 0.5)
        for ad in adduct_names:
            for iso, shift in ISOTOPES.items():
                if iso == 'iso0':
                    factor = 1.0
                elif iso == 'iso1':
                    factor = 0.15
                else:
                    factor = 0.02
                ad_factor = {'M+H': 1.0, 'M+Na': 0.6, 'M+NH4': 0.3, 'M+K': 0.2}[ad]
                intensity = base_intensity * ad_factor * factor
                intensity *= rng.lognormvariate(0, 0.1)
                mz = M + ADDUCTS[ad] + shift
                mz += rng.gauss(0, 1e-5)
                rt_jitter = rt + rng.gauss(0, 0.005)
                features.append({
                    'feature_id': fid,
                    'mz': round(mz, 6),
                    'rt': round(rt_jitter, 4),
                    'intensity': round(intensity, 2)
                })
                ion_type = f'{ad}_{iso}'
                labels.append({
                    'feature_id': fid,
                    'is_valid': 1,
                    'compound_id': cid,
                    'ion_type': ion_type
                })
                fid += 1
    for _ in range(n_noise):
        mz = rng.uniform(100.0, 1000.0)
        rt = rng.uniform(0.0, 15.0)
        intensity = rng.lognormvariate(math.log(1e4), 1.0)
        features.append({
            'feature_id': fid,
            'mz': round(mz, 6),
            'rt': round(rt, 4),
            'intensity': round(intensity, 2)
        })
        labels.append({
            'feature_id': fid,
            'is_valid': 0,
            'compound_id': -1,
            'ion_type': 'noise'
        })
        fid += 1
    combined = list(zip(features, labels))
    rng.shuffle(combined)
    features, labels = zip(*combined)
    with open(out_features, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['feature_id', 'mz', 'rt', 'intensity'])
        writer.writeheader()
        for row in features:
            writer.writerow(row)
    with open(out_labels, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['feature_id', 'is_valid', 'compound_id', 'ion_type'])
        writer.writeheader()
        for row in labels:
            writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--n_compounds', type=int, default=30)
    parser.add_argument('--n_noise', type=int, default=100)
    parser.add_argument('--out_features', required=True)
    parser.add_argument('--out_labels', required=True)
    args = parser.parse_args()
    generate(args.seed, args.n_compounds, args.n_noise, args.out_features, args.out_labels)
