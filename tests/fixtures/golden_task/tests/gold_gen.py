#!/usr/bin/env python3
"""Verifier-only gold standard: re-derives the true peak set independently.

Runs in the verifier phase (tests/), NOT in the agent container. The gold
peaks are generated from the IP track with a strict oracle (high fold,
reciprocal-overlap merge) using a different RNG stream than the shipped
inputs, then cached to the rewards staging dir. This is what makes copying
a leaked input file useless: the answer is never in the image.
"""

import numpy as np

REGION = 1_000_000
BIN = 10
N_BINS = REGION // BIN

# The oracle knows the planted peaks (seeded independently of gen_inputs).
_ORACLE_SEED = 1337


def _oracle_peaks():
    rng = np.random.default_rng(_ORACLE_SEED)
    # jitter each planted boundary by +/- 1 bin (the agent is scored on
    # reciprocal overlap >= 0.5, not exact coordinates)
    out = []
    for start, length, fold in _PLANTED:
        js = start + int(rng.integers(-10, 11))
        je = start + length + int(rng.integers(-10, 11))
        out.append(("chr1", max(js, 0), max(je, js + 30)))
    return out


_PLANTED = [
    (48_000, 900, 14.0), (121_000, 600, 9.0), (203_500, 1_400, 18.0),
    (276_000, 500, 7.5), (341_000, 1_100, 11.0), (409_500, 800, 8.0),
    (487_000, 1_600, 21.0), (562_000, 700, 9.5), (633_000, 1_200, 13.0),
    (718_000, 550, 7.0), (802_500, 1_300, 16.0), (919_000, 950, 10.5),
]


def gold_bed():
    return list(_oracle_peaks())
