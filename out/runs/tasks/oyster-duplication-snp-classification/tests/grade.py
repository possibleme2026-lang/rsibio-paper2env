import csv
import json

GOLD = {
    'v001': 'CN2',
    'v002': 'CN2',
    'v003': 'CN2',
    'v004': 'CN2',
    'v005': 'CN2',
    'v006': 'CN2',
    'v007': 'CN2',
    'v008': 'CN2',
    'v009': 'CN2',
    'v010': 'CN2',
    'v011': 'CN3',
    'v012': 'CN3',
    'v013': 'CN3',
    'v014': 'CN3',
    'v015': 'CN3',
    'v016': 'CN3',
    'v017': 'CN3',
    'v018': 'CN3',
    'v019': 'CN3',
    'v020': 'CN3',
    'v021': 'CN4',
    'v022': 'CN4',
    'v023': 'CN4',
    'v024': 'CN4',
    'v025': 'CN4',
    'v026': 'CN4',
    'v027': 'CN4',
    'v028': 'CN4',
    'v029': 'CN4',
    'v030': 'CN4'
}

def macro_f1(gold, pred):
    classes = ['CN2', 'CN3', 'CN4']
    scores = []
    for cls in classes:
        tp = sum(1 for k, v in gold.items() if v == cls and pred.get(k) == cls)
        fp = sum(1 for k, v in pred.items() if v == cls and gold.get(k) != cls)
        fn = sum(1 for k, v in gold.items() if v == cls and pred.get(k) != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return sum(scores) / len(scores)

score = 0.0
try:
    pred = {}
    with open('/app/output/predictions.tsv') as f:
        reader = csv.DictReader(f, delimiter=chr(9))
        for row in reader:
            vid = row.get('variant_id')
            lab = row.get('predicted_class')
            if vid and lab:
                pred[vid] = lab.strip().upper()
    score = macro_f1(GOLD, pred)
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
    json.dump({'score': round(score, 6)}, f)
