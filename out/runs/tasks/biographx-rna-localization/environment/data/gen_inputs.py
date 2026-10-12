import json, random, os
import numpy as np

SEED = 20260223
np.random.seed(SEED)
random.seed(SEED)

os.makedirs('/app/data', exist_ok=True)

n_per_type = 100
types = ['mRNA', 'miRNA', 'lncRNA']
lengths = {'mRNA': (400, 800), 'miRNA': (20, 24), 'lncRNA': (200, 500)}
seqs = []
for t in types:
    for i in range(n_per_type):
        L = random.randint(*lengths[t])
        gc = random.uniform(0.35, 0.65)
        seq_chars = []
        for _ in range(L):
            r = random.random()
            if r < gc/2:
                seq_chars.append('G')
            elif r < gc:
                seq_chars.append('C')
            elif r < gc + (1-gc)/2:
                seq_chars.append('A')
            else:
                seq_chars.append('U')
        seq = ''.join(seq_chars)
        seqs.append((f"{t}_{i:03d}", t, seq))

with open('/app/data/sequences.fasta', 'w') as f:
    for sid, t, seq in seqs:
        f.write(f">{sid}|{t}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + '\n')

emb_dim = 128
with open('/app/data/embeddings.csv', 'w') as f:
    header = ['seq_id'] + [f'emb_{i}' for i in range(emb_dim)]
    f.write(','.join(header) + '\n')
    for sid, t, seq in seqs:
        mean = {'mRNA': 0.0, 'miRNA': 0.5, 'lncRNA': -0.5}[t]
        emb = np.random.normal(mean, 1.0, emb_dim)
        f.write(sid + ',' + ','.join(f'{x:.6f}' for x in emb) + '\n')

F = 12
graph_features = ['length','gc_content','purine_content','mfe_norm','unpaired_fraction','basepair_count','backbone_edges','stacking_edges','physchem_edges','avg_degree','clustering_coeff','max_degree']
classes = ['nucleus','cytoplasm','exosome']
graph_weights = np.random.normal(0, 0.5, (3, F)).tolist()
seq_weights = np.random.normal(0, 0.1, (3, emb_dim)).tolist()
gate_weights = np.random.normal(0, 0.3, F).tolist()
gate_bias = float(np.random.normal(0, 0.1))
class_bias = np.random.normal(0, 0.1, 3).tolist()
weights = {
    'graph_features': graph_features,
    'classes': classes,
    'graph_weights': graph_weights,
    'seq_weights': seq_weights,
    'gate_weights': gate_weights,
    'gate_bias': gate_bias,
    'class_bias': class_bias
}
with open('/app/data/model_weights.json', 'w') as f:
    json.dump(weights, f, indent=2)
