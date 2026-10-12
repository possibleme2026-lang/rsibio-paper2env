#!/usr/bin/env python3
"""Deterministic generator for the shipped input tracks (for inspection).

This script re-creates signal.bedgraph / control.bedgraph exactly. It exists
so a reviewer (or an agent that distrusts binary-ish data files) can confirm
the inputs are synthetic and deterministic. It is NOT the gold-peak oracle:
the grader re-derives gold peaks at verify time from a different seed inside
tests/, which is unreachable from the agent container at solve time.
"""

import numpy as np

REGION = 1_000_000
BIN = 10
N_BINS = REGION // BIN

# Same 12 embedded peaks as the shipped tracks (documentation of the truth).
PEAKS = [
    (48_000, 900, 14.0), (121_000, 600, 9.0), (203_500, 1_400, 18.0),
    (276_000, 500, 7.5), (341_000, 1_100, 11.0), (409_500, 800, 8.0),
    (487_000, 1_600, 21.0), (562_000, 700, 9.5), (633_000, 1_200, 13.0),
    (718_000, 550, 7.0), (802_500, 1_300, 16.0), (919_000, 950, 10.5),
]


def coverage(rng, peaks):
    lam = rng.exponential(2.5, N_BINS) + 0.3
    for start, length, fold in peaks:
        s, e = start // BIN, (start + length) // BIN
        width = max(e - s, 1)
        ramp = np.hanning(width * 3)[width:]
        lam[s:e] += fold * (1.0 + 3.0 * ramp[:width])
    return rng.poisson(lam).astype(np.int32)


def write_bedgraph(path, counts):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# synthetic ChIP-seq coverage; seed 42; chrom=chr1; bin=10bp\n")
        for i, c in enumerate(counts):
            if c == 0:
                continue
            fh.write(f"chr1\t{i * BIN}\t{(i + 1) * BIN}\t{c}\n")


def main():
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    sig = coverage(np.random.default_rng(42), PEAKS)
    ctrl = coverage(np.random.default_rng(43), [])
    write_bedgraph(f"{out}/signal.bedgraph", sig)
    write_bedgraph(f"{out}/control.bedgraph", ctrl)
    print("regenerated signal.bedgraph + control.bedgraph (deterministic)")


if __name__ == "__main__":
    main()
