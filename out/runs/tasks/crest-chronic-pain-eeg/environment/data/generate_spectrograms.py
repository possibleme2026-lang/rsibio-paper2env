import csv, json
N=125; R=56
BANDS=['delta','theta','alpha','beta','gamma']
BW=[0.15,-0.10,0.55,0.05,-0.05]
GW=0.5
def pseudo(x):
    return ((x * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff * 2 - 1
def rw(r):
    return round(((r * 37) % 101 - 50) / 100.0, 4)
def bp(s,r,b):
    v = pseudo(s * 1000003 + r * 10007 + b * 101)
    if b == 2:
        v += 0.6 * pseudo(s * 9973 + 17)
    return round(v, 4)
model={
    'bands': BANDS,
    'band_weights': BW,
    'region_weights': [rw(r) for r in range(R)],
    'graph_mode': 'ring',
    'graph_weight': GW
}
with open('/app/data/crest_model.json', 'w') as f:
    json.dump(model, f, indent=2)
cols=['subject_id'] + ['r%d_%s' % (r, b) for r in range(R) for b in BANDS]
with open('/app/data/spectrograms.csv', 'w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for s in range(N):
        row={'subject_id': 'sub-%03d' % s}
        for r in range(R):
            for b in range(5):
                row['r%d_%s' % (r, BANDS[b])] = bp(s,r,b)
        w.writerow(row)
