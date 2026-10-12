#!/usr/bin/env python3
import json
import math

GOLD = {
    'TEST1_P1': 0,
    'TEST1_P2': 1,
    'TEST1_P3': 0,
    'TEST2_P1': 1,
    'TEST2_P2': 0,
    'TEST2_P3': 0,
    'TEST3_P1': 0,
    'TEST3_P2': 0,
    'TEST3_P3': 1,
    'TEST4_P1': 1,
    'TEST4_P2': 0,
    'TEST4_P3': 0
}

def auroc(labels, scores):
    npos = sum(labels)
    nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    pos_rank_sum = 0.0
    for k in range(len(pairs)):
        if pairs[k][1] == 1:
            pos_rank_sum += ranks[k]
    return (pos_rank_sum - npos * (npos + 1) / 2.0) / (npos * nneg)

def main():
    score = 0.0
    try:
        obj = json.load(open('/app/predictions.json'))
        preds = obj.get('predictions', [])
        pred_map = {}
        for item in preds:
            pid = str(item.get('participant_id', ''))
            prob = float(item.get('nonfriend_prob', 0.5))
            if prob < 0.0:
                prob = 0.0
            elif prob > 1.0:
                prob = 1.0
            pred_map[pid] = prob
        labels = []
        scores = []
        for pid, label in GOLD.items():
            labels.append(int(label))
            scores.append(pred_map.get(pid, 0.5))
        score = auroc(labels, scores)
    except Exception:
        import traceback as _tb, sys as _sys
        _err = _tb.format_exc()
        try:
            import os as _os
            _os.makedirs('/logs/verifier', exist_ok=True)
            open('/logs/verifier/grader_error.txt','w').write(_err)
        except Exception:
            pass
        print('GRADER_EXCEPTION\n' + _err, file=_sys.stderr, flush=True)
        score = 0.0
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({'score': round(score, 4)}, f)

if __name__ == '__main__':
    main()
