#!/usr/bin/env python3
"""Deterministic simulator for a circulating-miRNA case/control qPCR study.

This file is byte-identical to environment/data/gen_inputs.py, which the
Dockerfile executes during the image build (and then deletes from the image so
that the agent cannot read the simulation parameters).

Writes into argv[1] (default: the directory of this file):
  plate_cq.csv            raw Ct export (study samples, NTCs, replicates, spike-ins)
  sample_metadata.csv     per-sample clinical metadata (carries the group label)
  mirna_target_pairs.tsv  miRNA -> target gene interaction table
  immune_gene_panel.txt   curated immune gene panel
"""
import csv
import os
import random
import sys

SEED = 20260802
N_PER_GROUP = 40
NTC_SAMPLES = 2
TECH_SD = 0.25
SPIKE_SD = 0.35
FAIL_P = 0.015
HIGH_CT_P = 0.006

SPIKES = [('cel-miR-39-3p', 25.5), ('UniSp6', 22.5)]

# mirna, control baseline dCt, COVID-19 up-regulation (dCt_ctrl - dCt_covid),
# between-sample sd of dCt, per-sample detection probability
PANEL = [
    ('miR-126', 5.0, 1.05, 0.80, 1.00),
    ('miR-146b-5p', 8.0, 0.92, 0.85, 0.99),
    ('miR-223-5p', 4.0, 0.95, 0.80, 1.00),
    ('miR-144-3p', 9.0, 1.00, 0.85, 0.97),
    ('miR-22', 6.5, 0.88, 0.80, 0.99),
    ('miR-146a', 8.5, 0.85, 0.85, 0.99),
    ('miR-30c', 7.0, 1.15, 0.80, 0.99),
    ('miR-21-5p', 5.5, 0.32, 1.30, 1.00),
    ('miR-155', 11.0, 0.05, 1.00, 0.90),
    ('miR-1-3p', 11.0, 0.20, 1.00, 0.50),
    ('miR-499a-5p', 10.0, 0.15, 1.00, 0.42),
    ('miR-208a-3p', 12.0, 0.10, 1.00, 0.30),
]

GENES = '''
ABL1 AKT1 AKT2 AKT3 APAF1 ATF2 ATM BAX BCL10 BCL2 BCL2L1 BCL2L11 BCL6 BIRC2 BIRC3
CARD11 CASP1 CASP3 CASP7 CASP8 CASP9 CCL2 CCL5 CCND1 CD14 CD28 CD40 CD80 CD86 CDC42
CDKN1A CDKN2A CHUK CREB1 CSF1 CTNNB1 CXCL10 CXCL12 CXCR4 DICER1 DDX58 EGFR EIF4E
ELAVL1 EP300 ERBB2 FAS FOXO1 FOXO3 FOXP3 GATA3 GBP1 GSK3B HDAC1 HDAC2 HIF1A HMGB1
ICAM1 IFIH1 IFNAR1 IFNB1 IFNGR1 IKBKB IKBKG IL10 IL18 IL1B IL2 IL6 IL6R IRAK1 IRAK2
IRAK3 IRAK4 IRF1 IRF3 IRF7 IRF9 ISG15 ITGAM JAK1 JAK2 JAK3 JUN KDR KLF4 KRAS LGALS9
LY96 MAF MAP2K1 MAP3K7 MAPK1 MAPK14 MAPK3 MALT1 MCL1 MDM2 MEF2C MET MMP2 MMP9 MTOR
MX1 MYC MYD88 NFATC1 NFKB1 NFKB2 NFKBIA NLRP3 NOTCH1 NRAS OAS1 PDCD1 PDGFB PIK3CA
PIK3CB PIK3CD PIK3R1 PPARG PRDM1 PTEN RAC1 RB1 REL RELA RELB RICTOR ROCK1 RORC RUNX1
S100A8 S100A9 SERPINE1 SIRT1 SMAD2 SMAD3 SMAD4 SOCS1 SOCS3 SP1 SRC STAT1 STAT3 STAT5A
STAT6 STING1 TAB2 TAB3 TANK TBK1 TGFB1 TIGIT TLR2 TLR3 TLR4 TLR7 TLR9 TNF TNFRSF1A
TP53 TRAF3 TRAF5 TRAF6 TSC1 TYK2 VEGFA VIM XIAP XPO1 YAP1 ZEB1
'''.split()

CORE_IMMUNE = ['CXCL12', 'IRAK1', 'TRAF6', 'STAT1', 'JAK1', 'NOTCH1', 'SMAD4', 'BCL2L11']

IMMUNE_PANEL = CORE_IMMUNE + [
    'FOXO3', 'JAK2', 'SIRT1', 'NFKB1', 'RELA', 'IRF7', 'IL6', 'TNF', 'MYD88', 'TLR7',
    'CCL5', 'CXCL10', 'STAT3', 'IRF3', 'IFIH1', 'OAS1', 'ISG15', 'MX1', 'TLR4', 'CD14',
    'NFKBIA', 'SOCS1',
]


def _wells():
    letters = 'ABCDEFGH'
    n = 0
    while True:
        yield 'P%02d' % (n // 96 + 1), letters[(n % 96) // 12] + '%02d' % ((n % 96) % 12 + 1)
        n += 1


def _cq(value):
    return value if isinstance(value, str) else '%.2f' % value


def main(outdir=None):
    outdir = outdir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(outdir, exist_ok=True)
    rnd = random.Random(SEED)

    samples = [('COVID-%03d' % (i + 1), 'COVID-19') for i in range(N_PER_GROUP)]
    samples += [('CTRL-%03d' % (i + 1), 'Control') for i in range(N_PER_GROUP)]
    rnd.shuffle(samples)

    wells = _wells()
    rows = []
    for sid, grp in samples:
        base = {name: mean + rnd.gauss(0.0, SPIKE_SD) for name, mean in SPIKES}
        ref = sum(base.values()) / float(len(base))
        for mirna, mu0, delta, sd, det in PANEL:
            mu = mu0 - (delta if grp == 'COVID-19' else 0.0)
            above = rnd.random() < det
            value = ref + rnd.gauss(mu, sd)
            for rep in (1, 2):
                plate, well = next(wells)
                if (not above) or rnd.random() < FAIL_P:
                    cq = 'Undetermined'
                elif rnd.random() < HIGH_CT_P:
                    cq = round(rnd.uniform(38.2, 40.0), 2)
                else:
                    cq = round(value + rnd.gauss(0.0, TECH_SD), 2)
                rows.append([sid, plate, well, 'sample', mirna, rep, cq])
        for name, _mean in SPIKES:
            for rep in (1, 2):
                plate, well = next(wells)
                cq = round(base[name] + rnd.gauss(0.0, TECH_SD), 2)
                rows.append([sid, plate, well, 'sample', name, rep, cq])

    ntc_panel = [(m, 0, 0.0, 0.0, 0.0) for m, _a, _b, _c, _d in PANEL] + \
                [(n, 0, 0.0, 0.0, 0.0) for n, _m in SPIKES]
    for i in range(NTC_SAMPLES):
        sid = 'NTC-%02d' % (i + 1)
        for mirna, _a, _b, _c, _d in ntc_panel:
            for rep in (1, 2):
                plate, well = next(wells)
                if rnd.random() < 0.85:
                    cq = 'Undetermined'
                else:
                    cq = round(rnd.uniform(38.2, 40.5), 2)
                rows.append([sid, plate, well, 'NTC', mirna, rep, cq])

    rows.sort(key=lambda r: (r[1], r[2]))
    with open(os.path.join(outdir, 'plate_cq.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['sample_id', 'plate_id', 'well', 'sample_type', 'mirna', 'replicate', 'cq'])
        for r in rows:
            w.writerow(r[:6] + [_cq(r[6])])

    with open(os.path.join(outdir, 'sample_metadata.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['sample_id', 'group', 'age', 'sex', 'vaccination_status',
                    'sars_cov_2_rtqpcr', 'days_since_test'])
        for sid, grp in samples:
            age = int(min(78, max(19, round(rnd.gauss(38, 12)))))
            sex = 'F' if rnd.random() < 0.55 else 'M'
            vacc = 'vaccinated' if rnd.random() < 0.82 else 'unvaccinated'
            if grp == 'COVID-19':
                rt, days = 'positive', rnd.randint(0, 7)
            else:
                rt, days = 'negative', ''
            w.writerow([sid, grp, age, sex, vacc, rt, days])

    rnd2 = random.Random(SEED + 1)
    sig = [m for m, _a, _b, _c, _d in PANEL[:7]]
    evidence = ['reporter_assay', 'western_blot', 'qPCR', 'microarray', 'CLASH', 'AGO-CLIP']
    pairs = {}
    for mirna, _a, _b, _c, _d in PANEL:
        for gene in rnd2.sample(GENES, rnd2.randint(60, 85)):
            pairs[(mirna, gene)] = (round(rnd2.uniform(0.50, 0.99), 3), rnd2.choice(evidence))
    for gene in CORE_IMMUNE:
        for mirna in rnd2.sample(sig, rnd2.randint(2, 3)):
            pairs[(mirna, gene)] = (round(rnd2.uniform(0.80, 0.98), 3), rnd2.choice(evidence))
    for gene in CORE_IMMUNE[5:]:
        if rnd2.random() < 0.7:
            mirna = rnd2.choice(sig)
            pairs[(mirna, gene)] = (round(rnd2.uniform(0.55, 0.74), 3), 'microarray')

    with open(os.path.join(outdir, 'mirna_target_pairs.tsv'), 'w', newline='') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['mirna', 'gene', 'confidence', 'evidence_type'])
        for key in sorted(pairs):
            conf, ev = pairs[key]
            w.writerow([key[0], key[1], '%.3f' % conf, ev])

    with open(os.path.join(outdir, 'immune_gene_panel.txt'), 'w') as fh:
        for gene in IMMUNE_PANEL:
            fh.write(gene + '\n')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
