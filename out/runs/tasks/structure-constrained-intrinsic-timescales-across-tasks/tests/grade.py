import json

GOLD = {
    'c0': 'task',
    'c1': 'task',
    'c2': 'task',
    'c3': 'rest',
    'c4': 'task',
    'c5': 'task',
    'c6': 'task',
    'c7': 'task'
}

def normalize(v):
    return str(v).strip().lower()

def get_labels(obj):
    if isinstance(obj, dict):
        if 'labels' in obj and isinstance(obj['labels'], dict):
            return obj['labels']
        if 'rest_condition' in obj:
            rest = str(obj['rest_condition'])
            return {k: ('rest' if k == rest else 'task') for k in GOLD}
        if all(k in obj for k in GOLD):
            return obj
    return {}

def main():
    try:
        with open('/app/output.json') as f:
            obj = json.load(f)
    except Exception:
        obj = {}
    labels = get_labels(obj)
    tp = fp = fn = 0
    for cond, gold in GOLD.items():
        pred = normalize(labels.get(cond, 'task'))
        if pred == 'rest':
            if gold == 'rest':
                tp += 1
            else:
                fp += 1
        elif gold == 'rest':
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    score = max(0.0, min(1.0, f1))
    with open('/logs/verifier/reward.json', 'w') as out:
        json.dump({'score': score}, out)

if __name__ == '__main__':
    main()
