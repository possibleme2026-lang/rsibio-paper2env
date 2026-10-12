#!/usr/bin/env python3
'''Deterministic generator for the MPN plasma-proteome benchmark dataset.

Running this file writes proteome_npx.csv, sample_metadata.csv and
clinical_train.csv into the requested output directory (default: cwd).
Only random.random() from the Mersenne Twister is used together with a
hand-rolled Box-Muller transform, so identical seeds give byte-identical
files on any CPython 3 interpreter. The file is executed once while the
image is built and is not part of the final container.
'''
import csv
import math
import os
import random
import sys

FEATURE_SEED = 771013
N_SAMPLES = 96
N_TEST = 32

MPN_UP = [('SORT1', 1.05), ('GP1BA', 0.95), ('MMP1', 0.85), ('PSPN', 0.80),
          ('BAG6', 0.75), ('THBS1', 0.70), ('PF4', 0.90), ('PPBP', 0.85),
          ('SELP', 0.72), ('PECAM1', 0.60), ('ITGA2B', 0.78), ('GP6', 0.65)]
MPN_DOWN = [('CD40', -0.70), ('NCAM1', -0.75), ('CDCP1', -0.62)]
PV_SPEC = [('TFRC', 1.45), ('SEMA7A', 1.00), ('EPOR', 1.20)]
JAK2_PROT = ['IL6', 'GH1']
NULL_PROT = ['ALB', 'APOA1', 'APOB', 'HP', 'CP', 'TF', 'FGA', 'FGB', 'FGG',
             'SERPINA1', 'A2M', 'C3', 'CFH', 'CRP', 'LBP', 'TIMP1', 'VCAM1',
             'ICAM1', 'SELE', 'VWF', 'F8', 'PLAT', 'SERPINC1', 'ANGPT2']

MPN_UP_MAP = dict(MPN_UP)
MPN_DOWN_MAP = dict(MPN_DOWN)
PV_MAP = dict(PV_SPEC)
PROTEINS = ([p for p, _ in MPN_UP] + [p for p, _ in MPN_DOWN] +
            [p for p, _ in PV_SPEC] + JAK2_PROT + NULL_PROT)

GOLD_DE_UP = [p for p, _ in MPN_UP] + [p for p, _ in PV_SPEC]
GOLD_DE_DOWN = [p for p, _ in MPN_DOWN]
GOLD_DE_SET = {}
for _p in GOLD_DE_UP:
    GOLD_DE_SET[_p] = 1
for _p in GOLD_DE_DOWN:
    GOLD_DE_SET[_p] = -1
GOLD_INFORMATIVE = set(GOLD_DE_UP) | set(GOLD_DE_DOWN)
GOLD_NULL = list(JAK2_PROT) + list(NULL_PROT)

PLATELET_PROT = {'PF4', 'PPBP', 'THBS1', 'GP6', 'ITGA2B'}
HB_PROT = {'TFRC', 'EPOR'}
AGE_PROT = {'IL6', 'CRP', 'LBP', 'TIMP1'}


def gauss(rng):
    u1 = rng.random()
    if u1 < 1e-12:
        u1 = 1e-12
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def generate_dataset(seed=FEATURE_SEED):
    rng = random.Random(seed)
    params = {}
    for p in PROTEINS:
        params[p] = {'base': 5.0 + 3.0 * rng.random(),
                     'ind_sd': 0.35 + 0.20 * rng.random(),
                     'meas_sd': 0.18 + 0.14 * rng.random(),
                     'miss': 0.012 + 0.020 * rng.random()}
    params['LBP']['miss'] = 0.11
    params['CDCP1']['miss'] = 0.09
    params['SEMA7A']['miss'] = 0.07
    params['TFRC']['miss'] = 0.03
    plates = ['PL1', 'PL2', 'PL3']
    plate_off = {}
    for pl in plates:
        for p in PROTEINS:
            plate_off[(pl, p)] = 0.25 * gauss(rng)
    samples = []
    for i in range(N_SAMPLES):
        age = min(88.0, max(28.0, 62.0 + 12.0 * gauss(rng)))
        sex = 'M' if rng.random() < 0.52 else 'F'
        group = 1 if rng.random() < 0.62 else 0
        if group == 1:
            u = rng.random()
            subtype = 'ET' if u < 0.38 else ('PV' if u < 0.78 else 'PMF')
        else:
            subtype = 'non-MPN'
        jak2 = 'positive' if rng.random() < (0.72 if group == 1 else 0.20) else 'negative'
        if group == 0:
            plt_m, plt_s = 375.0, 130.0
        elif subtype == 'ET':
            plt_m, plt_s = 590.0, 170.0
        elif subtype == 'PV':
            plt_m, plt_s = 480.0, 150.0
        else:
            plt_m, plt_s = 330.0, 150.0
        platelet = min(1200.0, max(80.0, plt_m + plt_s * gauss(rng)))
        if group == 0:
            hb_m, hb_s = 13.6, 1.5
        elif subtype == 'PV':
            hb_m, hb_s = 17.0, 1.1
        elif subtype == 'ET':
            hb_m, hb_s = 13.8, 1.2
        else:
            hb_m, hb_s = 10.6, 1.6
        hb = min(20.5, max(7.0, hb_m + hb_s * gauss(rng)))
        if group == 0:
            wbc_m, wbc_s = 9.5, 2.5
        elif subtype == 'PV':
            wbc_m, wbc_s = 13.0, 3.5
        elif subtype == 'ET':
            wbc_m, wbc_s = 9.8, 2.6
        else:
            wbc_m, wbc_s = 8.5, 3.0
        wbc = min(35.0, max(3.0, wbc_m + wbc_s * gauss(rng)))
        if group == 0:
            ldh_m, ldh_s = 195.0, 45.0
        elif subtype == 'PMF':
            ldh_m, ldh_s = 340.0, 90.0
        else:
            ldh_m, ldh_s = 230.0, 55.0
        ldh = min(900.0, max(100.0, ldh_m + ldh_s * gauss(rng)))
        plate = plates[rng.randrange(3)]
        samples.append({'sample_id': 'P%03d' % (i + 1), 'idx': i, 'age': age,
                        'sex': sex, 'group': group, 'subtype': subtype,
                        'jak2': jak2, 'platelet': platelet, 'hb': hb,
                        'wbc': wbc, 'ldh': ldh, 'plate': plate})
    values = {}
    for s in samples:
        partial = 0.5 if (s['group'] == 0 and rng.random() < 0.25) else 0.0
        row = {}
        for p in PROTEINS:
            prm = params[p]
            eff = 0.0
            if p in MPN_UP_MAP:
                eff += MPN_UP_MAP[p] * (1.0 if s['group'] == 1 else partial)
            elif p in MPN_DOWN_MAP:
                eff += MPN_DOWN_MAP[p] * (1.0 if s['group'] == 1 else partial)
            elif p in PV_MAP:
                if s['subtype'] == 'PV':
                    eff += PV_MAP[p]
                elif s['group'] == 1:
                    eff += 0.15 * PV_MAP[p]
            elif p in JAK2_PROT:
                if s['jak2'] == 'positive':
                    eff += 0.45
                if s['group'] == 1 and s['jak2'] == 'negative':
                    eff -= 0.45
            if p in PLATELET_PROT:
                eff += 0.0022 * (s['platelet'] - 350.0)
            if p in HB_PROT:
                eff += 0.060 * (s['hb'] - 13.5)
            if p in AGE_PROT:
                eff += 0.010 * (s['age'] - 62.0)
            v = prm['base'] + prm['ind_sd'] * gauss(rng) + eff
            v += plate_off[(s['plate'], p)]
            v += prm['meas_sd'] * gauss(rng)
            if rng.random() < prm['miss']:
                row[p] = None
            else:
                row[p] = round(v, 3)
        values[s['sample_id']] = row
    idx_mpn = [s['idx'] for s in samples if s['group'] == 1]
    idx_ctl = [s['idx'] for s in samples if s['group'] == 0]
    rng.shuffle(idx_mpn)
    rng.shuffle(idx_ctl)
    n_test_mpn = int(round(N_TEST * len(idx_mpn) / float(N_SAMPLES)))
    n_test_ctl = N_TEST - n_test_mpn
    test_idx = set(idx_mpn[:n_test_mpn]) | set(idx_ctl[:n_test_ctl])
    for s in samples:
        s['cohort'] = 'test' if s['idx'] in test_idx else 'train'
    return {'samples': samples, 'values': values, 'proteins': PROTEINS,
            'plate_offsets': plate_off}


def write_inputs(outdir):
    ds = generate_dataset()
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = '.'
    with open(os.path.join(outdir, 'proteome_npx.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['sample_id'] + ds['proteins'])
        for s in ds['samples']:
            row = ds['values'][s['sample_id']]
            w.writerow([s['sample_id']] + ['' if row[p] is None else '%.3f' % row[p]
                                           for p in ds['proteins']])
    with open(os.path.join(outdir, 'sample_metadata.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['sample_id', 'cohort', 'plate', 'age', 'sex', 'platelet_count',
                    'hemoglobin', 'wbc_count', 'ldh', 'jak2_v617f'])
        for s in ds['samples']:
            w.writerow([s['sample_id'], s['cohort'], s['plate'], '%.1f' % s['age'],
                        s['sex'], '%.0f' % s['platelet'], '%.1f' % s['hb'],
                        '%.1f' % s['wbc'], '%.0f' % s['ldh'], s['jak2']])
    with open(os.path.join(outdir, 'clinical_train.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['sample_id', 'diagnosis', 'mpn_subtype'])
        for s in ds['samples']:
            if s['cohort'] == 'train':
                w.writerow([s['sample_id'], 'MPN' if s['group'] == 1 else 'non-MPN',
                            s['subtype']])
    return ds


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else '.'
    write_inputs(target)
    print('inputs written to', target)
