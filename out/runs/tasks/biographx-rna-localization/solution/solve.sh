#!/bin/bash
set -euo pipefail
cat > /app/solve.py <<'PY'
import csv, json, math, os, subprocess, tempfile
from collections import defaultdict

DATA_DIR = '/app/data'
OUT_DIR = '/app/output'
os.makedirs(OUT_DIR, exist_ok=True)

def parse_fasta(path):
    seqs = []
    with open(path) as f:
        sid = None
        seq = []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                if sid is not None:
                    seqs.append((sid, ''.join(seq)))
                sid = line[1:].split('|')[0]
                seq = []
            else:
                seq.append(line)
        if sid is not None:
            seqs.append((sid, ''.join(seq)))
    return seqs

seqs = parse_fasta(os.path.join(DATA_DIR, 'sequences.fasta'))
seq_ids = [s[0] for s in seqs]

with tempfile.NamedTemporaryFile('w', suffix='.fasta', delete=False) as tmp:
    for sid, seq in seqs:
        tmp.write(f'>{sid}\n{seq}\n')
    tmp_path = tmp.name

result = subprocess.run(['RNAfold', '--noPS', tmp_path], capture_output=True, text=True, check=True)
os.unlink(tmp_path)
lines = result.stdout.strip().split('\n')
structures = {}
i = 0
while i < len(lines):
    if lines[i].startswith('>'):
        sid = lines[i][1:].strip()
        db_line = lines[i+2].strip()
        parts = db_line.split()
        db = parts[0]
        mfe = float(parts[1].strip('()'))
        structures[sid] = (db, mfe)
        i += 3
    else:
        i += 1

def physchem_score(a, b):
    pair = tuple(sorted((a, b)))
    table = {('A','U'):1.0, ('G','C'):1.0, ('G','U'):0.7, ('A','G'):0.5, ('C','U'):0.5}
    return table.get(pair, 0.0)

def compute_features(seq, db, mfe):
    L = len(seq)
    gc = sum(1 for c in seq if c in 'GC') / L
    purine = sum(1 for c in seq if c in 'AG') / L
    unpaired = db.count('.') / L
    bp_count = db.count('(')
    adj = defaultdict(set)
    for i in range(1, L):
        adj[i].add(i+1)
        adj[i+1].add(i)
    stack = []
    pairs = []
    for i, c in enumerate(db, start=1):
        if c == '(':
            stack.append(i)
        elif c == ')':
            j = stack.pop()
            pairs.append((j, i))
            adj[j].add(i)
            adj[i].add(j)
    pair_set = set(pairs)
    stacking_edges = set()
    for (i, j) in pairs:
        if (i+1, j-1) in pair_set:
            u, v = i, j-1
            if u != v:
                stacking_edges.add((min(u,v), max(u,v)))
    for u, v in stacking_edges:
        adj[u].add(v)
        adj[v].add(u)
    physchem_edges = set()
    for i in range(1, L+1):
        for j in range(i+4, min(L, i+50)+1):
            if physchem_score(seq[i-1], seq[j-1]) >= 0.5:
                u, v = i, j
                physchem_edges.add((min(u,v), max(u,v)))
    for u, v in physchem_edges:
        adj[u].add(v)
        adj[v].add(u)
    length = L
    mfe_norm = mfe / L
    backbone_edges = L - 1
    stacking_edges_count = len(stacking_edges)
    physchem_edges_count = len(physchem_edges)
    degrees = {i: len(adj[i]) for i in range(1, L+1)}
    avg_degree = sum(degrees.values()) / L
    max_degree = max(degrees.values()) if L > 0 else 0
    clustering_sum = 0.0
    count_nodes = 0
    for i in range(1, L+1):
        neigh = adj[i]
        k = len(neigh)
        if k >= 2:
            e = 0
            neigh_list = list(neigh)
            for a in range(len(neigh_list)):
                for b in range(a+1, len(neigh_list)):
                    if neigh_list[b] in adj[neigh_list[a]]:
                        e += 1
            C_i = 2 * e / (k * (k - 1))
            clustering_sum += C_i
            count_nodes += 1
    clustering_coeff = clustering_sum / count_nodes if count_nodes > 0 else 0.0
    return [length, gc, purine, mfe_norm, unpaired, bp_count, backbone_edges, stacking_edges_count, physchem_edges_count, avg_degree, clustering_coeff, max_degree]

features = {}
for sid, seq in seqs:
    db, mfe = structures[sid]
    features[sid] = compute_features(seq, db, mfe)

embeddings = {}
with open(os.path.join(DATA_DIR, 'embeddings.csv')) as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        sid = row[0]
        embeddings[sid] = [float(x) for x in row[1:]]

with open(os.path.join(DATA_DIR, 'model_weights.json')) as f:
    W = json.load(f)
graph_features = W['graph_features']
classes = W['classes']
graph_weights = W['graph_weights']
seq_weights = W['seq_weights']
gate_weights = W['gate_weights']
gate_bias = W['gate_bias']
class_bias = W['class_bias']

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def softmax(x):
    m = max(x)
    ex = [math.exp(v - m) for v in x]
    s = sum(ex)
    return [v / s for v in ex]

with open(os.path.join(OUT_DIR, 'graph_features.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['seq_id'] + graph_features)
    for sid, _ in seqs:
        writer.writerow([sid] + features[sid])

with open(os.path.join(OUT_DIR, 'gates.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['seq_id', 'gate'])
    for sid, _ in seqs:
        feat = features[sid]
        gate = sigmoid(sum(feat[i] * gate_weights[i] for i in range(len(feat))) + gate_bias)
        writer.writerow([sid, gate])

with open(os.path.join(OUT_DIR, 'predictions.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['seq_id'] + classes)
    for sid, _ in seqs:
        feat = features[sid]
        emb = embeddings[sid]
        gate = sigmoid(sum(feat[i] * gate_weights[i] for i in range(len(feat))) + gate_bias)
        logits = []
        for c in range(len(classes)):
            seq_logit = sum(emb[j] * seq_weights[c][j] for j in range(len(emb)))
            graph_logit = sum(feat[i] * graph_weights[c][i] for i in range(len(feat)))
            fused = gate * seq_logit + (1 - gate) * graph_logit + class_bias[c]
            logits.append(fused)
        probs = softmax(logits)
        writer.writerow([sid] + probs)
PY
python3 /app/solve.py
