import numpy as np
import pandas as pd
import os

def generate_data(seed=42, n_train=400, n_test=200):
    np.random.seed(seed)
    categories = ['dog', 'rain', 'car', 'speech', 'music']
    distortions = ['none', 'lowpass', 'clipping', 'time_reverse']
    snr_levels = [0, 10, 20, 30]
    
    def make_samples(n, start_id):
        rows = []
        for i in range(n):
            clip_id = f"clip_{start_id + i:04d}"
            category = np.random.choice(categories)
            distortion = np.random.choice(distortions)
            n_sources = np.random.choice([1, 2, 3])
            snr_db = np.random.choice(snr_levels)
            
            cat_effect = {'dog': 0.5, 'rain': -0.5, 'car': 0.0, 'speech': 0.8, 'music': -0.8}
            dist_effect = {'none': 0.0, 'lowpass': -0.5, 'clipping': -0.2, 'time_reverse': -0.8}
            
            f1 = np.random.normal(cat_effect[category], 0.8)
            f2 = np.random.normal(-cat_effect[category] * 0.5, 0.8)
            f3 = np.random.normal(dist_effect[distortion], 0.8)
            f4 = np.random.normal(0.01 * snr_db, 0.8)
            f5 = np.random.normal(-0.1 * (n_sources - 1), 0.8)
            f6 = np.random.normal(0, 1.0)
            f7 = np.random.normal(0, 1.0)
            f8 = np.random.normal(0, 1.0)
            f9 = np.random.normal(0, 1.0)
            f10 = np.random.normal(0, 1.0)
            f11 = np.random.normal(0, 1.0)
            f12 = np.random.normal(0, 1.0)
            
            base = 0.85
            cat_acc = {'dog': 0.05, 'rain': -0.10, 'car': -0.05, 'speech': 0.10, 'music': -0.15}
            dist_acc = {'none': 0.0, 'lowpass': -0.15, 'clipping': -0.08, 'time_reverse': -0.25}
            nsrc_acc = -0.08 * (n_sources - 1)
            snr_acc = 0.008 * (snr_db - 15)
            
            feat_acc = (0.08 * f1 + -0.06 * f2 + 0.05 * f3 + 0.04 * f4 + -0.04 * f5 + 
                        0.02 * f6 + -0.02 * f7 + 0.01 * f8 + 0.01 * f9)
            
            noise = np.random.normal(0, 0.04)
            accuracy = base + cat_acc[category] + dist_acc[distortion] + nsrc_acc + snr_acc + feat_acc + noise
            accuracy = np.clip(accuracy, 0.0, 1.0)
            
            row = {
                'clip_id': clip_id,
                'category': category,
                'distortion': distortion,
                'n_sources': n_sources,
                'snr_db': snr_db,
                'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4, 'f5': f5, 'f6': f6,
                'f7': f7, 'f8': f8, 'f9': f9, 'f10': f10, 'f11': f11, 'f12': f12,
                'human_accuracy': accuracy
            }
            rows.append(row)
        return pd.DataFrame(rows)
    
    train = make_samples(n_train, 0)
    test = make_samples(n_test, n_train)
    test_no_target = test.drop(columns=['human_accuracy'])
    return train, test, test_no_target

if __name__ == "__main__":
    train, test, test_no_target = generate_data()
    os.makedirs('/app/data', exist_ok=True)
    train.to_csv('/app/data/train.csv', index=False)
    test_no_target.to_csv('/app/data/test.csv', index=False)
