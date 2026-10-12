#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python3 - <<'PYEOF'
import csv, json, math, os

DATA = '/app/data'
OUT = '/app/output'

SPEC = json.load(open(os.path.join(DATA, 'model_spec.json')))
ANN = json.load(open(os.path.join(DATA, 'cassette_annotation.json')))
WT = ''.join(l.strip() for l in open(os.path.join(DATA, 'cassette.fa')) if not l.startswith('>')).upper()
N = len(WT)
CDS_START = ANN['cds_start_0based']
CDS_END = ANN['cds_end_0based']
CANON0 = ANN['canonical_tss_1based'] - 1
DET = SPEC['detection_threshold']
STRONG = SPEC['strong_threshold']
WIN = SPEC['nms_window']
PWM = SPEC['core_pwm']
DINU = SPEC['dinucleotide_weights']
BASAL = SPEC['basal_offset']
FLANKS = [tuple(SPEC['upstream_flank']), tuple(SPEC['downstream_flank'])]

_B = 'TCAG'
_AA = ('FFLLSSSSYY**CC*W' 'LLLLPPPPHHQQRRRR' 'IIIMTTTTNNKKSSRR' 'VVVVAAAADDEEGGGG')
CODON = {}
for _i, _b1 in enumerate(_B):
    for _j, _b2 in enumerate(_B):
        for _k, _b3 in enumerate(_B):
            CODON[_b1 + _b2 + _b3] = _AA[_i * 16 + _j * 4 + _k]


def revcomp(s):
    return s.translate(str.maketrans('ACGT', 'TGCA'))[::-1]


def score_site(seq, tss):
    if tss - 17 < 0 or tss + 6 >= len(seq):
        return None
    s = BASAL
    for r in range(-17, 7):
        s += PWM[str(r)][seq[tss + r]]
    for lo, hi in FLANKS:
        for r in range(lo, hi):
            i = tss + r
            j = i + 1
            if i < 0 or j >= len(seq):
                continue
            s += DINU[seq[i] + seq[j]]
    return round(s, 6)


def all_scores(seq):
    sc = {}
    n = len(seq)
    for tss in range(17, n - 6):
        v = score_site(seq, tss)
        if v is not None:
            sc[('+', tss)] = v
    rc = revcomp(seq)
    for tss in range(17, n - 6):
        v = score_site(rc, tss)
        if v is not None:
            sc[('-', n - 1 - tss)] = v
    return sc


def rescore(sc, seq, pos):
    n = len(seq)
    rc = revcomp(seq)
    for d in range(-45, 46):
        fp = pos + d
        if 17 <= fp <= n - 7:
            v = score_site(seq, fp)
            if v is not None:
                sc[('+', fp)] = v
        tss = n - 1 - fp
        if 17 <= tss <= n - 7:
            v = score_site(rc, tss)
            if v is not None:
                sc[('-', fp)] = v
    return sc


def sites_from(sc):
    raw = [{'fpos0': fp, 'strand': st, 'score': v}
           for (st, fp), v in sc.items() if v >= DET]
    kept = []
    for r in raw:
        ok = True
        for q in raw:
            if q is r or q['strand'] != r['strand']:
                continue
            if abs(q['fpos0'] - r['fpos0']) > WIN:
                continue
            if q['score'] > r['score'] or (q['score'] == r['score'] and q['fpos0'] < r['fpos0']):
                ok = False
                break
        if ok:
            kept.append(r)
    kept.sort(key=lambda d: d['fpos0'])
    return kept


def cryptic_total(sites):
    return sum(s['score'] for s in sites if s['score'] < STRONG)


# ---- 1. predictions on the held-out set ----
with open(os.path.join(OUT, 'predictions.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['seq_id', 'predicted_log10_rate'])
    for row in csv.DictReader(open(os.path.join(DATA, 'heldout_promoters.csv'))):
        w.writerow([row['seq_id'], '%.4f' % score_site(row['seq'], 50)])

# ---- 2. wild-type site map ----
sc = all_scores(WT)
wt_sites = sites_from(sc)
wt_total = cryptic_total(wt_sites)
with open(os.path.join(OUT, 'sites_wildtype.json'), 'w') as fh:
    out = []
    for s in wt_sites:
        core = WT[s['fpos0'] - 17:s['fpos0'] + 7]
        if s['strand'] == '-':
            core = revcomp(core)
        out.append({'position': s['fpos0'] + 1, 'strand': s['strand'],
                    'score': round(s['score'], 4),
                    'class': 'strong' if s['score'] >= STRONG else 'cryptic',
                    'sequence': core})
    json.dump({'sites': out}, fh, indent=2)

# ---- 3. constrained synonymous redesign ----
seq = list(WT)
protected = set(range(max(0, CANON0 - 40), min(N, CANON0 + 31)))


def aa_of(cod):
    return CODON.get(cod, 'X')


def silent(pos, newb):
    if not (CDS_START <= pos < CDS_END):
        return True
    k = CDS_START + 3 * ((pos - CDS_START) // 3)
    old = ''.join(seq[k:k + 3])
    new = old[:(pos - k)] + newb + old[(pos - k) + 1:]
    return aa_of(old) == aa_of(new) and aa_of(old) != '*'


def n_changed():
    return sum(1 for a, b in zip(WT, seq) if a != b)


target = 0.35 * wt_total
for it in range(80):
    cur = sites_from(sc)
    tot = cryptic_total(cur)
    if tot <= target or n_changed() >= 55:
        break
    crypt = sorted([s for s in cur if s['score'] < STRONG], key=lambda s: -s['score'])
    best = None
    for site in crypt[:4]:
        lo = max(0, site['fpos0'] - 22)
        hi = min(N, site['fpos0'] + 13)
        for p in range(lo, hi):
            if p in protected:
                continue
            for b in 'ACGT':
                if seq[p] == b or not silent(p, b):
                    continue
                old = seq[p]
                seq[p] = b
                trial = ''.join(seq)
                new_sc = dict(sc)
                rescore(new_sc, trial, p)
                nt = cryptic_total(sites_from(new_sc))
                if best is None or nt < best[0]:
                    best = (nt, p, b, new_sc)
                seq[p] = old
    if best is None or best[0] >= tot - 1e-9:
        break
    seq[best[1]] = best[2]
    sc = best[3]

rd = ''.join(seq)
rd_sc = all_scores(rd)
rd_sites = sites_from(rd_sc)
rd_total = cryptic_total(rd_sites)
rd_cds = rd[CDS_START:CDS_END]
wt_cds = WT[CDS_START:CDS_END]
n_changes = sum(1 for a, b in zip(WT, rd) if a != b)
cds_changes = sum(1 for a, b in zip(wt_cds, rd_cds) if a != b)
gc_wt = (wt_cds.count('G') + wt_cds.count('C')) / float(len(wt_cds))
gc_rd = (rd_cds.count('G') + rd_cds.count('C')) / float(len(rd_cds))
frac = (wt_total - rd_total) / wt_total if wt_total > 0 else 0.0
wt_canon = score_site(WT, CANON0)
rd_canon = score_site(rd, CANON0)

with open(os.path.join(OUT, 'redesigned_cassette.fa'), 'w') as fh:
    fh.write('>t7_cassette_redesigned\n')
    for i in range(0, len(rd), 60):
        fh.write(rd[i:i + 60] + '\n')

report = {
    'canonical_site_position': CANON0 + 1,
    'wildtype': {'n_sites': len(wt_sites),
                 'n_cryptic': len([s for s in wt_sites if s['score'] < STRONG]),
                 'total_cryptic_score': round(wt_total, 4),
                 'canonical_score': round(wt_canon, 4)},
    'redesigned': {'n_sites': len(rd_sites),
                   'n_cryptic': len([s for s in rd_sites if s['score'] < STRONG]),
                   'total_cryptic_score': round(rd_total, 4),
                   'canonical_score': round(rd_canon, 4)},
    'nucleotide_changes': n_changes,
    'cds_nucleotide_changes': cds_changes,
    'gc_content_cds_wildtype': round(gc_wt, 4),
    'gc_content_cds_redesigned': round(gc_rd, 4),
    'cryptic_score_reduction_fraction': round(frac, 4),
}
with open(os.path.join(OUT, 'design_report.json'), 'w') as fh:
    json.dump(report, fh, indent=2)
print('done: changes=%d cds_changes=%d reduction=%.3f' % (n_changes, cds_changes, frac))
PYEOF
