# Deterministic generator for the T7 promoter redesign benchmark.
# This file is the protected reference copy used by the grader.
import csv, json, os, random

SEED = 20260801
CONSENSUS = 'TAATACGACTCACTATAGGGAGA'
CASSETTE_LEN = 3000
CANONICAL_TSS0 = 400
CDS_START = 500
CDS_LEN = 1023
CDS_END = CDS_START + CDS_LEN
UP_FLANK = 'GCGTACGCATGCGTACGCATGCA'
DN_FLANK = 'GCATGCGTACGCATGCGTACGCAT'
# Planted sites live strictly OUTSIDE the CDS [CDS_START, CDS_END)
# so the stop-repair pass never has to fight a planted window
# (the original [700, 860, ...] list had 700/860/1000/1150/1400
# inside the CDS, which made 'stop repair' unsatisfiable).
# Planted sites live strictly OUTSIDE the CDS window and away from the
# canonical TSS so the stop-repair pass never has to fight a planted
# window (with the original list, 5 of 8 plants landed inside the CDS
# and made the repair loop unsatisfiable -- RuntimeError 'stop repair
# failed' at build time).
PLANT_POSITIONS = [100, 160, 220, 280, 340, 1600, 1660, 1720]
STOPS = {'TAA', 'TAG', 'TGA'}

_B = 'TCAG'
_AA = ('FFLLSSSSYY**CC*W'
       'LLLLPPPPHHQQRRRR'
       'IIIMTTTTNNKKSSRR'
       'VVVVAAAADDEEGGGG')
CODON = {}
for _i, _b1 in enumerate(_B):
    for _j, _b2 in enumerate(_B):
        for _k, _b3 in enumerate(_B):
            CODON[_b1 + _b2 + _b3] = _AA[_i * 16 + _j * 4 + _k]


def build_spec():
    positions = list(range(-17, 7))
    def imp(r):
        if -17 <= r <= -12:
            return 1.00
        if -11 <= r <= -7:
            return 0.85
        if -6 <= r <= -1:
            return 0.65
        return 0.75
    trans = {'A': 'G', 'G': 'A', 'C': 'T', 'T': 'C'}
    pwm = {}
    for i, r in enumerate(positions):
        c = CONSENSUS[i % len(CONSENSUS)]
        w = imp(r)
        row = {}
        for b in 'ACGT':
            if b == c:
                row[b] = round(0.32 * w, 4)
            elif trans[c] == b:
                row[b] = round(-0.45 * w, 4)
            else:
                row[b] = round(-0.62 * w, 4)
        pwm[str(r)] = row
    dinu = {
        'AA': -0.02, 'AC': 0.03, 'AG': 0.05, 'AT': -0.04,
        'CA': -0.03, 'CC': 0.06, 'CG': 0.12, 'CT': 0.04,
        'GA': 0.04, 'GC': 0.08, 'GG': 0.06, 'GT': 0.01,
        'TA': -0.05, 'TC': 0.02, 'TG': 0.03, 'TT': -0.06,
    }
    return {
        'name': 'T7 Promoter Calculator (surrogate sequence-to-function model)',
        'units': 'log10 relative transcription rate',
        'basal_offset': -3.0,
        'core_region': [-17, 6],
        'core_consensus': CONSENSUS,
        'core_pwm': pwm,
        'upstream_flank': [-40, -18],
        'downstream_flank': [7, 30],
        'dinucleotide_weights': dinu,
        'detection_threshold': 0.0,
        'strong_threshold': 2.5,
        'nms_window': 12,
    }


def revcomp(s):
    return s.translate(str.maketrans('ACGT', 'TGCA'))[::-1]


def score_site(seq, tss, spec):
    n = len(seq)
    if tss - 17 < 0 or tss + 6 >= n:
        return None
    s = spec['basal_offset']
    pwm = spec['core_pwm']
    for r in range(-17, 7):
        s += pwm[str(r)][seq[tss + r]]
    dw = spec['dinucleotide_weights']
    for lo, hi in (spec['upstream_flank'], spec['downstream_flank']):
        for r in range(lo, hi):
            i = tss + r
            j = i + 1
            if i < 0 or j >= n:
                continue
            s += dw[seq[i] + seq[j]]
    return round(s, 6)


def scan_raw(seq, spec, thr):
    n = len(seq)
    out = []
    for tss in range(17, n - 6):
        sc = score_site(seq, tss, spec)
        if sc is not None and sc >= thr:
            out.append({'fpos0': tss, 'strand': '+', 'score': sc,
                        'sequence': seq[tss - 17:tss + 7]})
    rc = revcomp(seq)
    for tss in range(17, n - 6):
        sc = score_site(rc, tss, spec)
        if sc is not None and sc >= thr:
            fp = n - 1 - tss
            out.append({'fpos0': fp, 'strand': '-', 'score': sc,
                        'sequence': revcomp(seq[fp - 17:fp + 7])})
    return out


def scan(seq, spec, thr=None, win=None):
    if thr is None:
        thr = spec['detection_threshold']
    if win is None:
        win = spec['nms_window']
    raw = scan_raw(seq, spec, thr)
    kept = []
    for r in raw:
        ok = True
        for q in raw:
            if q is r or q['strand'] != r['strand']:
                continue
            if abs(q['fpos0'] - r['fpos0']) > win:
                continue
            if q['score'] > r['score'] or (q['score'] == r['score'] and q['fpos0'] < r['fpos0']):
                ok = False
                break
        if ok:
            kept.append({'position': r['fpos0'] + 1, 'strand': r['strand'],
                         'score': r['score'], 'sequence': r['sequence'],
                         'fpos0': r['fpos0']})
    kept.sort(key=lambda d: (d['fpos0'], d['strand']))
    return kept


def translate(seq):
    return ''.join(CODON.get(seq[i:i + 3], 'X') for i in range(0, len(seq) - 2, 3))


def build_cassette(seed=SEED):
    rng = random.Random(seed)
    spec = build_spec()
    seq = [rng.choice('ACGT') for _ in range(CASSETTE_LEN)]
    t = CANONICAL_TSS0
    seq[t - 17:t + 7] = list(CONSENSUS)
    seq[t - 40:t - 17] = list(UP_FLANK)
    seq[t + 7:t + 31] = list(DN_FLANK)
    for k in range(0, CDS_LEN, 3):
        while True:
            cod = ''.join(rng.choice('ACGT') for _ in range(3))
            if cod not in STOPS:
                break
        seq[CDS_START + k:CDS_START + k + 3] = list(cod)
    seq[CDS_START:CDS_START + 3] = list('ATG')
    seq[CDS_END - 3:CDS_END] = list('TAA')
    s = ''.join(seq)
    planted = []
    for p in PLANT_POSITIONS:
        hit = None
        for _ in range(8000):
            nmut = rng.choice([4, 5, 5, 6, 6, 7])
            core = list(CONSENSUS)
            for idx in rng.sample(range(len(CONSENSUS)), nmut):
                core[idx] = rng.choice([b for b in 'ACGT' if b != CONSENSUS[idx]])
            trial = list(s)
            trial[p - 17:p + 7] = core
            cand = ''.join(trial)
            sc = score_site(cand, p, spec)
            if sc is not None and 0.30 <= sc <= 1.90:
                s = cand
                hit = sc
                break
        if hit is None:
            raise RuntimeError('planting failed at %d' % p)
        planted.append({'pos0': p, 'score': hit})
    for _ in range(400):
        bad = None
        for k in range(0, CDS_LEN - 3, 3):
            if s[CDS_START + k:CDS_START + k + 3] in STOPS:
                bad = k
                break
        if bad is None:
            break
        fixed = False
        for off in (2, 1, 0):
            for b in 'ACGT':
                pos = CDS_START + bad + off
                if s[pos] == b:
                    continue
                cand = s[:pos] + b + s[pos + 1:]
                if cand[CDS_START + bad:CDS_START + bad + 3] in STOPS:
                    continue
                ok = True
                for pl in planted:
                    sc = score_site(cand, pl['pos0'], spec)
                    if sc is None or sc < 0.2 or sc > 2.3:
                        ok = False
                        break
                if ok:
                    s = cand
                    fixed = True
                    break
            if fixed:
                break
        if not fixed:
            raise RuntimeError('stop repair failed')
    for pl in planted:
        pl['score'] = score_site(s, pl['pos0'], spec)
    return {'sequence': s, 'canonical_tss0': CANONICAL_TSS0,
            'cds_start': CDS_START, 'cds_end': CDS_END, 'planted': planted}


def build_heldout(n=1500, seed=SEED + 11):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        seq = [rng.choice('ACGT') for _ in range(90)]
        core = list(CONSENSUS)
        nm = rng.randint(0, 9)
        for idx in rng.sample(range(len(CONSENSUS)), nm):
            core[idx] = rng.choice([b for b in 'ACGT' if b != CONSENSUS[idx]])
        seq[33:57] = core
        rows.append(('ho_%04d' % (i + 1), ''.join(seq)))
    return rows


def build_benchmark(n=500, seed=SEED + 22):
    rng = random.Random(seed)
    spec = build_spec()
    rows = []
    for i in range(n):
        seq = [rng.choice('ACGT') for _ in range(90)]
        core = list(CONSENSUS)
        nm = rng.randint(0, 11)
        for idx in rng.sample(range(len(CONSENSUS)), nm):
            core[idx] = rng.choice([b for b in 'ACGT' if b != CONSENSUS[idx]])
        seq[33:57] = core
        t = ''.join(seq)
        measured = score_site(t, 50, spec) + rng.gauss(0.0, 0.8)
        rows.append(('bm_%04d' % (i + 1), t, round(measured, 4)))
    return rows


def build_all(outdir):
    spec = build_spec()
    cass = build_cassette()
    with open(os.path.join(outdir, 'model_spec.json'), 'w') as fh:
        json.dump(spec, fh, indent=2)
    with open(os.path.join(outdir, 'cassette.fa'), 'w') as fh:
        fh.write('>t7_cassette_wildtype\n')
        for i in range(0, len(cass['sequence']), 60):
            fh.write(cass['sequence'][i:i + 60] + '\n')
    ann = {'length': len(cass['sequence']),
           'canonical_tss_1based': cass['canonical_tss0'] + 1,
           'cds_start_0based': cass['cds_start'],
           'cds_end_0based': cass['cds_end']}
    with open(os.path.join(outdir, 'cassette_annotation.json'), 'w') as fh:
        json.dump(ann, fh, indent=2)
    with open(os.path.join(outdir, 'heldout_promoters.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['seq_id', 'seq'])
        for sid, seq in build_heldout():
            w.writerow([sid, seq])
    with open(os.path.join(outdir, 'benchmark_promoters.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['seq_id', 'seq', 'measured_log10_rate'])
        for sid, seq, rate in build_benchmark():
            w.writerow([sid, seq, rate])


if __name__ == '__main__':
    build_all(os.path.dirname(os.path.abspath(__file__)))
