#!/usr/bin/env python3
# Grader for the T7 promoter redesign benchmark.
import csv, json, math, os, sys, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_gen import (build_spec, build_cassette, build_heldout, scan,
                      score_site, translate)

OUT = '/app/output'
REWARD_DIR = '/logs/verifier'

SPEC = build_spec()
CASS = build_cassette()
WT = CASS['sequence']
WT_SITES = scan(WT, SPEC)
DET = SPEC['detection_threshold']
STRONG = SPEC['strong_threshold']
CANON0 = CASS['canonical_tss0']
CDS_START = CASS['cds_start']
CDS_END = CASS['cds_end']
WT_CDS = WT[CDS_START:CDS_END]
WT_PROTEIN = translate(WT_CDS)
WT_CANON_SCORE = score_site(WT, CANON0, SPEC)


def cryptic_total(sites):
    return sum(s['score'] for s in sites if s['score'] < STRONG)


WT_TOTAL = cryptic_total(WT_SITES)
WT_NCRYPT = len([s for s in WT_SITES if s['score'] < STRONG])
WT_GC = (WT_CDS.count('G') + WT_CDS.count('C')) / float(len(WT_CDS))

CHECKS = []


def check(name, ok, detail=''):
    CHECKS.append({'name': name, 'pass': bool(ok), 'detail': detail})
    print(('PASS ' if ok else 'FAIL ') + name + (' :: ' + detail if detail else ''))


def read_text(path):
    with open(path) as fh:
        return fh.read()


# ---------------- predictions ----------------
agent_pred = {}
pred_err = ''
pp = os.path.join(OUT, 'predictions.csv')
if os.path.exists(pp):
    try:
        with open(pp, newline='') as fh:
            for row in csv.DictReader(fh):
                agent_pred[str(row['seq_id']).strip()] = float(row['predicted_log10_rate'])
    except Exception as exc:
        pred_err = str(exc)
else:
    pred_err = 'missing file'

heldout = build_heldout()
ids = [sid for sid, _ in heldout]
true_scores = {sid: score_site(seq, 50, SPEC) for sid, seq in heldout}
missing = [i for i in ids if i not in agent_pred]
badnum = [i for i in ids if i in agent_pred and not math.isfinite(agent_pred[i])]
check('predictions_format', (not missing) and (not badnum),
      'missing=%d bad=%d %s' % (len(missing), len(badnum), pred_err))

rmse = float('inf')
if not missing and not badnum:
    rmse = math.sqrt(sum((agent_pred[i] - true_scores[i]) ** 2 for i in ids) / float(len(ids)))
check('predictions_rmse_le_0.05', rmse <= 0.05,
      'rmse=%.4f' % rmse if math.isfinite(rmse) else 'rmse=NA')

# ---------------- wild-type sites ----------------
agent_sites = []
sites_err = ''
sp = os.path.join(OUT, 'sites_wildtype.json')
if os.path.exists(sp):
    try:
        data = json.load(open(sp))
        items = data['sites'] if isinstance(data, dict) else data
        for s in items:
            agent_sites.append((int(s['position']), str(s['strand']).strip(), float(s['score'])))
    except Exception as exc:
        sites_err = str(exc)
else:
    sites_err = 'missing file'

truth_sites = [(s['position'], s['strand'], s['score']) for s in WT_SITES]


def f1(agent, truth, tol_pos=2, tol_score=0.15):
    used = set()
    tp = 0
    for (p, st, sc) in agent:
        best = None
        bestd = None
        for i, (q, st2, sc2) in enumerate(truth):
            if i in used or st2 != st:
                continue
            if abs(q - p) <= tol_pos and abs(sc2 - sc) <= tol_score:
                d = abs(q - p)
                if best is None or d < bestd:
                    best, bestd = i, d
        if best is not None:
            used.add(best)
            tp += 1
    fp = len(agent) - tp
    fn = len(truth) - tp
    if tp == 0:
        return 0.0
    return 2.0 * tp / (2.0 * tp + fp + fn)


f1v = f1(agent_sites, truth_sites) if agent_sites else 0.0
check('wildtype_site_f1_ge_0.90', f1v >= 0.90,
      'f1=%.3f agent=%d truth=%d %s' % (f1v, len(agent_sites), len(truth_sites), sites_err))

# ---------------- re-designed cassette ----------------
rd = None
rd_err = ''
rp = os.path.join(OUT, 'redesigned_cassette.fa')
if os.path.exists(rp):
    try:
        lines = [l.strip() for l in read_text(rp).splitlines() if l.strip()]
        rd = ''.join(l for l in lines if not l.startswith('>')).upper()
    except Exception as exc:
        rd_err = str(exc)
else:
    rd_err = 'missing file'

check('redesign_length_3000', rd is not None and len(rd) == len(WT),
      'len=%s expected=%d %s' % (len(rd) if rd else 'NA', len(WT), rd_err))
check('redesign_alphabet_ACGT', rd is not None and len(rd) > 0 and set(rd) <= set('ACGT'),
      'alphabet=%s' % (''.join(sorted(set(rd))) if rd else 'NA'))

valid = rd is not None and len(rd) == len(WT)
if valid:
    rd_cds = rd[CDS_START:CDS_END]
    rd_sites = scan(rd, SPEC)
    rd_total = cryptic_total(rd_sites)
    rd_ncrypt = len([s for s in rd_sites if s['score'] < STRONG])
    frac = (WT_TOTAL - rd_total) / WT_TOTAL if WT_TOTAL > 0 else 0.0
    rd_gc = (rd_cds.count('G') + rd_cds.count('C')) / float(len(rd_cds))
    n_changes = sum(1 for a, b in zip(WT, rd) if a != b)
    cds_changes = sum(1 for a, b in zip(WT_CDS, rd_cds) if a != b)
    rd_canon = score_site(rd, CANON0, SPEC)
else:
    rd_cds = ''
    rd_sites = []
    rd_total = None
    rd_ncrypt = None
    frac = None
    rd_gc = None
    n_changes = None
    cds_changes = None
    rd_canon = None

check('cds_protein_identical', valid and translate(rd_cds) == WT_PROTEIN,
      'no valid redesign' if not valid else 'protein_len=%d' % len(WT_PROTEIN))
check('canonical_core_preserved',
      valid and WT[CANON0 - 17:CANON0 + 7] == rd[CANON0 - 17:CANON0 + 7],
      '' if valid else 'no valid redesign')
check('cryptic_score_reduction_ge_0.50', valid and frac is not None and frac >= 0.50,
      'wt=%.4f rd=%s reduction=%s' % (WT_TOTAL, rd_total, frac) if valid else 'no valid redesign')
new_strong = [s for s in rd_sites if s['score'] >= STRONG and abs(s['fpos0'] - CANON0) > 40] \
    if valid else []
check('no_new_strong_site', valid and len(new_strong) == 0,
      'n_new_strong=%d' % len(new_strong) if valid else 'no valid redesign')
check('cds_gc_within_0.02', valid and rd_gc is not None and abs(rd_gc - WT_GC) <= 0.02,
      'wt=%.4f rd=%.4f' % (WT_GC, rd_gc) if valid else 'no valid redesign')
check('change_budget', valid and n_changes <= 60 and cds_changes <= 82,
      'total=%d cds=%d' % (n_changes, cds_changes) if valid else 'no valid redesign')
check('canonical_score_preserved',
      valid and rd_canon is not None and rd_canon >= 0.97 * WT_CANON_SCORE,
      'wt=%.4f rd=%s' % (WT_CANON_SCORE, rd_canon) if valid else 'no valid redesign')

# ---------------- design report ----------------
rep = None
rep_err = ''
rpp = os.path.join(OUT, 'design_report.json')
if os.path.exists(rpp):
    try:
        rep = json.load(open(rpp))
    except Exception as exc:
        rep_err = str(exc)
else:
    rep_err = 'missing file'


def close(a, b, tol):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


ok = False
detail = rep_err
if isinstance(rep, dict) and valid:
    try:
        rw = rep['wildtype']
        rr = rep['redesigned']
        ok = (int(rep['canonical_site_position']) == CANON0 + 1
              and close(rw['total_cryptic_score'], WT_TOTAL, 0.05)
              and int(rw['n_cryptic']) == WT_NCRYPT
              and close(rw['canonical_score'], WT_CANON_SCORE, 0.05)
              and close(rr['total_cryptic_score'], rd_total, 0.05)
              and int(rr['n_cryptic']) == rd_ncrypt
              and close(rr['canonical_score'], rd_canon, 0.05)
              and int(rep['nucleotide_changes']) == n_changes
              and int(rep['cds_nucleotide_changes']) == cds_changes
              and close(rep['gc_content_cds_wildtype'], WT_GC, 0.005)
              and close(rep['gc_content_cds_redesigned'], rd_gc, 0.005)
              and close(rep['cryptic_score_reduction_fraction'], frac, 0.02))
        detail = 'ok' if ok else 'field mismatch'
    except Exception as exc:
        ok = False
        detail = str(exc)
check('design_report_consistent', ok, detail)

# ---------------- reward ----------------
npass = sum(1 for c in CHECKS if c['pass'])
score = npass / float(len(CHECKS)) if CHECKS else 0.0
os.makedirs(REWARD_DIR, exist_ok=True)
with open(os.path.join(REWARD_DIR, 'reward.json'), 'w') as fh:
    json.dump({'score': round(score, 4), 'reward': round(score, 4)}, fh)
print('TOTAL %d/%d score=%.4f' % (npass, len(CHECKS), score))
