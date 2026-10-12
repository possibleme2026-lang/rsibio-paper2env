#!/usr/bin/env python3
'''Grader: EOLA1-dependent acC salvage (LC-MS quantification + dipyridamole sensitivity).

Eight independent constraints, score = fraction passed.
Gold is embedded (hidden from the agent) and is cross-checked by recomputation
where the shipped data allows it.
'''
import csv, json, math, os, re, traceback

OUT = '/app/output'
LOGDIR = '/logs/verifier'

# ----------------------------------------------------------- atomic masses
AM = {'C': 12.0, 'H': 1.00782503207, 'N': 14.0030740048, 'O': 15.9949146196,
      'F': 18.99840322, 'Na': 22.9897692809, 'P': 30.97376163, 'S': 31.97207100}
PROTON = 1.007276466621
ELECTRON = 0.000548579909
ADDUCT_SHIFT = {'M+H': PROTON, 'M+NA': AM['Na'] - ELECTRON,
                'M+NH4': AM['N'] + 4 * AM['H'] - ELECTRON}

COMPOUNDS = {'cytidine': 'C9H13N3O5', 'uridine': 'C9H12N2O6',
             'pseudouridine': 'C9H12N2O6', '5-methylcytidine': 'C10H15N3O5',
             '5-fluorocytidine': 'C9H12FN3O5', 'adenosine': 'C10H13N5O4',
             'inosine': 'C10H12N4O5', 'guanosine': 'C10H13N5O5',
             'N6-methyladenosine': 'C11H15N5O4',
             '1-methyladenosine': 'C11H15N5O4',
             'N4-acetylcytidine': 'C11H15N3O6', "2'-deoxycytidine": 'C9H13N3O4'}

CHANNEL_MZ = ['244.0928', '245.0768', '258.1084', '262.0834',
              '268.1040', '282.1197', '286.1034', '312.1444']
# gold mapping: channel -> (compound, adduct) or None for unassignable
GOLD_CHANNEL = {'244.0928': ('cytidine', 'M+H'),
                '245.0768': ('uridine', 'M+H'),
                '258.1084': ('5-methylcytidine', 'M+H'),
                '262.0834': ('5-fluorocytidine', 'M+H'),
                '268.1040': ('adenosine', 'M+H'),
                '282.1197': ('N6-methyladenosine', 'M+H'),
                '286.1034': ('N4-acetylcytidine', 'M+H'),
                '312.1444': None}

GOLD_MEANS = {
    'WT': {'cytidine': 42.167, 'uridine': 26.033, '5-methylcytidine': 8.200,
           'N6-methyladenosine': 2.100, 'N4-acetylcytidine': 1.200},
    'EOLA1-KO1': {'cytidine': 44.333, 'uridine': 25.333,
                  '5-methylcytidine': 8.467, 'N6-methyladenosine': 2.150,
                  'N4-acetylcytidine': 5.583},
    'EOLA1-KO2': {'cytidine': 43.167, 'uridine': 25.933,
                  '5-methylcytidine': 8.100, 'N6-methyladenosine': 2.133,
                  'N4-acetylcytidine': 5.117},
    'KO1+EOLA1': {'cytidine': 41.000, 'uridine': 25.333,
                  '5-methylcytidine': 8.267, 'N6-methyladenosine': 2.117,
                  'N4-acetylcytidine': 1.350},
}

GOLD_IC50 = [('dose_response', 'WT', False, 12.0),
             ('dose_response', 'EOLA1-KO1', False, 2.5),
             ('dose_response', 'EOLA1-KO2', False, 3.0),
             ('dose_response', 'KO1+EOLA1', False, 11.0),
             ('cytidine_rescue', 'EOLA1-KO1', False, 2.8),
             ('cytidine_rescue', 'EOLA1-KO1', True, 9.5)]

SUMMARY_KEYS = ['n_channels_annotated', 'n_channels_unassigned',
                'free_acC_fold_change_KO1_vs_WT',
                'free_acC_fold_change_KO2_vs_WT',
                'free_acC_restored_by_reexpression',
                'other_nucleosides_unchanged', 'ic50_uM_WT',
                'ic50_uM_EOLA1_KO1', 'ic50_uM_EOLA1_KO2',
                'ic50_uM_KO1_reexpressed', 'ic50_uM_KO1_dipyridamole',
                'ic50_uM_KO1_dipyridamole_cytidine', 'cytidine_rescue_index',
                'cytidine_rescues_viability',
                'EOLA1_loss_sensitis_to_dipyridamole_placeholder']

# helpers ---------------------------------------------------------------
def parse_formula(f):
    out = {}
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', f):
        if not el:
            continue
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out

def mono(formula):
    return sum(AM[e] * n for e, n in parse_formula(formula).items())

def theo_mz(formula, adduct):
    return mono(formula) + ADDUCT_SHIFT[adduct]

def read_csv(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))

def norm(s):
    return re.sub(r'\s+', '', str(s if s is not None else '')).lower()

def norm_adduct(s):
    t = norm(s).strip('[]')
    t = t.replace('-', '')
    return t.upper()

def to_float(s):
    if s is None:
        return None
    try:
        return float(str(s).strip())
    except Exception:
        return None

def truthy(s):
    return norm(s) in ('true', '1', 'yes', 't')

def close(a, b, tol):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(abs(b), 1e-12)

# --------------------------------------------------------------- checks
def check_annotation():
    path = os.path.join(OUT, 'channel_annotation.csv')
    rows = read_csv(path)
    index = {}
    for r in rows:
        mz = to_float(r.get('channel_mz'))
        if mz is None:
            continue
        index[round(mz, 4)] = r
    bad = []
    for mz_s in CHANNEL_MZ:
        mz = round(float(mz_s), 4)
        r = index.get(mz)
        if r is None:
            bad.append('channel %s missing' % mz_s)
            continue
        gold = GOLD_CHANNEL[mz_s]
        comp = norm(r.get('compound'))
        if gold is None:
            if comp != 'unassigned':
                bad.append('channel %s should be unassigned (got %s)' % (mz_s, r.get('compound')))
            continue
        if comp != norm(gold[0]):
            bad.append('channel %s assigned %s, expected %s' % (mz_s, r.get('compound'), gold[0]))
            continue
        if norm_adduct(r.get('adduct')) != norm_adduct(gold[1]):
            bad.append('channel %s adduct %s, expected %s' % (mz_s, r.get('adduct'), gold[1]))
            continue
        ppm = to_float(r.get('ppm_error'))
        if ppm is None:
            ppm = 0.0
        if abs(ppm) > 10.0:
            bad.append('channel %s ppm_error %.2f out of range' % (mz_s, ppm))
    return (not bad), ('all 8 channels annotated within 10 ppm' if not bad else '; '.join(bad[:5]))

def check_quantification():
    rows = read_csv(os.path.join(OUT, 'nucleoside_quantification.csv'))
    index = {(norm(r.get('cell_line')), norm(r.get('analyte'))): r for r in rows}
    bad = []
    for line, analytes in GOLD_MEANS.items():
        for ana, gold in analytes.items():
            r = index.get((norm(line), norm(ana)))
            if r is None:
                bad.append('missing %s/%s' % (line, ana))
                continue
            v = to_float(r.get('mean_pmol_per_mg'))
            if not close(v, gold, 0.15):
                bad.append('%s/%s mean %s vs %s' % (line, ana, v, gold))
    return (not bad), ('20/20 quantified values within 15%%' if not bad else '; '.join(bad[:5]))

def check_statistics():
    rows = read_csv(os.path.join(OUT, 'nucleoside_quantification.csv'))
    index = {(norm(r.get('cell_line')), norm(r.get('analyte'))): r for r in rows}
    bad = []
    for line, analytes in GOLD_MEANS.items():
        for ana in analytes:
            r = index.get((norm(line), norm(ana)))
            if r is None:
                bad.append('missing %s/%s' % (line, ana))
                continue
            want = (line != 'WT') and ana == 'N4-acetylcytidine' and line in ('EOLA1-KO1', 'EOLA1-KO2')
            got = truthy(r.get('significant'))
            if got != want:
                bad.append('%s/%s significant=%s expected %s' % (line, ana, got, want))
                continue
            p = to_float(r.get('ttest_p'))
            if p is None:
                bad.append('%s/%s missing p-value' % (line, ana))
            elif want and p > 0.01:
                bad.append('%s/%s p=%.4g too large for a significant call' % (line, ana, p))
            elif (not want) and line != 'WT' and p < 0.02:
                bad.append('%s/%s p=%.4g too small for a non-significant call' % (line, ana, p))
    return (not bad), ('significance calls match the BH rule' if not bad else '; '.join(bad[:5]))

def _ic50_index():
    rows = read_csv(os.path.join(OUT, 'ic50_summary.csv'))
    index = {}
    for r in rows:
        key = (norm(r.get('dataset')), norm(r.get('cell_line')),
               'cytidine' in norm(r.get('treatment')))
        index[key] = r
    return index

def check_ic50():
    index = _ic50_index()
    bad = []
    for dataset, line, cyt, gold in GOLD_IC50:
        r = index.get((norm(dataset), norm(line), cyt))
        if r is None:
            bad.append('missing curve %s/%s/%s' % (dataset, line, cyt))
            continue
        ic = to_float(r.get('ic50_uM'))
        hill = to_float(r.get('hill_slope'))
        r2 = to_float(r.get('r_squared'))
        if not close(ic, gold, 0.25):
            bad.append('%s/%s IC50 %s vs %s' % (dataset, line, ic, gold))
        if hill is None or abs(hill - 1.0) > 0.35:
            bad.append('%s/%s hill=%s' % (dataset, line, hill))
        if r2 is None or r2 < 0.97:
            bad.append('%s/%s R2=%s' % (dataset, line, r2))
    return (not bad), ('6/6 4PL fits within tolerance' if not bad else '; '.join(bad[:5]))

def check_phenotype():
    index = _ic50_index()
    bad = []
    wt = to_float(index.get(('doseresponse', 'wt', False), {}).get('ic50_uM'))
    ko1 = to_float(index.get(('doseresponse', 'eola1-ko1', False), {}).get('ic50_uM'))
    ko2 = to_float(index.get(('doseresponse', 'eola1-ko2', False), {}).get('ic50_uM'))
    resc = to_float(index.get(('doseresponse', 'ko1+eola1', False), {}).get('ic50_uM'))
    if wt is None or ko1 is None or ko2 is None or resc is None:
        return False, 'dose_response curves incomplete'
    if not (wt / ko1 >= 1.8):
        bad.append('KO1 IC50 shift %.2f < 2' % (wt / ko1))
    if not (wt / ko2 >= 1.8):
        bad.append('KO2 IC50 shift %.2f < 2' % (wt / ko2))
    if not (0.7 <= resc / wt <= 1.5):
        bad.append('re-expressed IC50 ratio %.2f outside [0.7,1.5]' % (resc / wt))
    try:
        sj = json.load(open(os.path.join(OUT, 'summary.json')))
    except Exception as exc:
        return False, 'summary.json unreadable (%s)' % exc
    if not truthy(sj.get('EOLA1_loss_sensitis_to_dipyridamole_placeholder', sj.get('EOLA1_loss_sensitises_to_dipyridamole'))):
        bad.append('summary call EOLA1_loss_sensitis... is not true')
    if not truthy(sj.get('free_acC_restored_by_reexpression')):
        bad.append('summary call free_acC_restored_by_reexpression is not true')
    return (not bad), ('sensitivity + restoration phenotype reproduced' if not bad else '; '.join(bad))

def check_rescue():
    index = _ic50_index()
    dip = to_float(index.get(('cytidinerescue', 'eola1-ko1', False), {}).get('ic50_uM'))
    dipc = to_float(index.get(('cytidinerescue', 'eola1-ko1', True), {}).get('ic50_uM'))
    bad = []
    if dip is None or dipc is None or dip <= 0:
        return False, 'cytidine_rescue curves incomplete'
    ratio = dipc / dip
    if ratio < 1.8:
        bad.append('rescue index %.2f < 2' % ratio)
    try:
        sj = json.load(open(os.path.join(OUT, 'summary.json')))
    except Exception as exc:
        return False, 'summary.json unreadable (%s)' % exc
    if not truthy(sj.get('cytidine_rescues_viability')):
        bad.append('summary call cytidine_rescues_viability is not true')
    idx = to_float(sj.get('cytidine_rescue_index'))
    if not close(idx, ratio, 0.20):
        bad.append('summary rescue index %s vs fitted %.2f' % (idx, ratio))
    return (not bad), ('cytidine rescue reproduced (index %.2f)' % ratio if not bad else '; '.join(bad))

def check_format():
    bad = []
    spec = {'channel_annotation.csv': ['channel_mz', 'rt_min', 'compound', 'adduct', 'ppm_error'],
            'nucleoside_quantification.csv': ['cell_line', 'analyte', 'mean_pmol_per_mg',
                                              'sd_pmol_per_mg', 'fold_change_vs_WT', 'ttest_p',
                                              'significant'],
            'ic50_summary.csv': ['dataset', 'cell_line', 'treatment', 'ic50_uM',
                                 'hill_slope', 'r_squared']}
    for fname, cols in spec.items():
        path = os.path.join(OUT, fname)
        if not os.path.exists(path):
            bad.append('%s missing' % fname)
            continue
        rows = read_csv(path)
        if not rows:
            bad.append('%s empty' % fname)
            continue
        header = list(rows[0].keys())
        if [h.strip() for h in header] != cols:
            bad.append('%s header %s' % (fname, header))
    q = os.path.join(OUT, 'nucleoside_quantification.csv')
    if os.path.exists(q):
        rows = read_csv(q)
        if len(rows) != 20:
            bad.append('quantification has %d rows (expected 20)' % len(rows))
        for r in rows:
            sd = to_float(r.get('sd_pmol_per_mg'))
            if sd is None or sd < 0:
                bad.append('bad sd in row %s/%s' % (r.get('cell_line'), r.get('analyte')))
                break
    i = os.path.join(OUT, 'ic50_summary.csv')
    if os.path.exists(i):
        rows = read_csv(i)
        if len(rows) != 6:
            bad.append('ic50_summary has %d rows (expected 6)' % len(rows))
        for r in rows:
            ic = to_float(r.get('ic50_uM'))
            if ic is None or ic <= 0:
                bad.append('non-positive IC50 for %s' % r.get('cell_line'))
                break
    j = os.path.join(OUT, 'summary.json')
    if not os.path.exists(j):
        bad.append('summary.json missing')
    else:
        try:
            sj = json.load(open(j))
            for k in ['n_channels_annotated', 'n_channels_unassigned', 'cytidine_rescue_index']:
                if k not in sj:
                    bad.append('summary.json missing key %s' % k)
        except Exception as exc:
            bad.append('summary.json invalid (%s)' % exc)
    return (not bad), ('all four output files well formed' if not bad else '; '.join(bad[:5]))

def check_consistency():
    bad = []
    try:
        sj = json.load(open(os.path.join(OUT, 'summary.json')))
    except Exception as exc:
        return False, 'summary.json unreadable (%s)' % exc
    rows = read_csv(os.path.join(OUT, 'nucleoside_quantification.csv'))
    index = {(norm(r.get('cell_line')), norm(r.get('analyte'))): r for r in rows}
    for line, key in (('EOLA1-KO1', 'free_acC_fold_change_KO1_vs_WT'),
                      ('EOLA1-KO2', 'free_acC_fold_change_KO2_vs_WT')):
        r = index.get((norm(line), norm('N4-acetylcytidine')))
        fc = to_float(r.get('fold_change_vs_WT')) if r else None
        if not close(to_float(sj.get(key)), fc, 0.10):
            bad.append('%s = %s but CSV fold change %s' % (key, sj.get(key), fc))
    n_ann = to_float(sj.get('n_channels_annotated'))
    n_un = to_float(sj.get('n_channels_unassigned'))
    if n_ann is None or n_un is None or n_ann + n_un != 8:
        bad.append('channel counts do not sum to 8 (%s + %s)' % (n_ann, n_un))
    idx = _ic50_index()
    pairs = [('ic50_uM_WT', ('doseresponse', 'wt', False)),
             ('ic50_uM_EOLA1_KO1', ('doseresponse', 'eola1-ko1', False)),
             ('ic50_uM_EOLA1_KO2', ('doseresponse', 'eola1-ko2', False)),
             ('ic50_uM_KO1_reexpressed', ('doseresponse', 'ko1+eola1', False)),
             ('ic50_uM_KO1_dipyridamole', ('cytidinerescue', 'eola1-ko1', False)),
             ('ic50_uM_KO1_dipyridamole_cytidine', ('cytidinerescue', 'eola1-ko1', True))]
    for key, k in pairs:
        r = idx.get(k)
        ic = to_float(r.get('ic50_uM')) if r else None
        if not close(to_float(sj.get(key)), ic, 0.10):
            bad.append('%s = %s but CSV IC50 %s' % (key, sj.get(key), ic))
    return (not bad), ('summary.json consistent with CSVs' if not bad else '; '.join(bad[:5]))

CHECKS = [('channel_annotation', check_annotation),
          ('quantification_accuracy', check_quantification),
          ('statistics', check_statistics),
          ('dose_response_fits', check_ic50),
          ('sensitivity_phenotype', check_phenotype),
          ('cytidine_rescue', check_rescue),
          ('output_format', check_format),
          ('summary_consistency', check_consistency)]

def main():
    os.makedirs(LOGDIR, exist_ok=True)
    results = []
    for name, fn in CHECKS:
        try:
            ok, msg = fn()
        except Exception as exc:
            ok, msg = False, 'exception: %s' % exc
            traceback.print_exc()
        results.append({'constraint': name, 'passed': bool(ok), 'detail': msg})
    passed = sum(1 for r in results if r['passed'])
    score = round(passed / float(len(results)), 4)
    print('==== EOLA1 acC salvage grader ====')
    for r in results:
        print('[%s] %-26s %s' % ('PASS' if r['passed'] else 'FAIL', r['constraint'], r['detail']))
    print('score = %d/%d = %.4f' % (passed, len(results), score))
    with open(os.path.join(LOGDIR, 'reward.json'), 'w') as fh:
        json.dump({'score': score, 'reward': score}, fh)
    with open(os.path.join(LOGDIR, 'report.json'), 'w') as fh:
        json.dump({'score': score, 'constraints': results}, fh, indent=2)
    return 0

if __name__ == '__main__':
    main()
