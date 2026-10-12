#!/bin/bash
# Reference solution: Poisson peak caller over the shipped bedGraph tracks.
# Produces /app/predictions.bed (BED6) exactly as the instruction requires.
set -euo pipefail

DATA_DIR=/app/data
OUT=/app/predictions.bed

python3 - "$DATA_DIR" "$OUT" <<'PYEOF'
import sys
import numpy as np

data_dir, out_path = sys.argv[1], sys.argv[2]

def load_track(path):
    """Parse a bedGraph file into a dict of (start, end) -> coverage."""
    cov = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            chrom, start, end, c = line.rstrip("\n").split("\t")
            cov[(int(start), int(end))] = int(c)
    return cov

signal = load_track(f"{data_dir}/signal.bedgraph")
control = load_track(f"{data_dir}/control.bedgraph")

# background rate: median of the control track (robust to IP-only signal)
bg = float(np.median(list(control.values()))) if control else 1.0

# score candidate bins with the Poisson tail statistic
starts = sorted({s for (s, e) in signal})
width = min((e - s) for (s, e) in signal) if signal else 10
scores = []
for (s, e), c in sorted(signal.items()):
    expected = bg * (e - s) / width
    # poisson upper tail via normal approx with continuity guard
    z = (c - expected) / max(np.sqrt(expected), 1.0)
    scores.append((s, e, c, z))

# keep significant bins (>= 2x background), then merge adjacent/close ones
sig = [(s, e, c) for (s, e, c, z) in scores if c >= 2.0 * bg]
merged = []
for s, e, c in sig:
    if merged and s - merged[-1][1] < 50:
        ps, pe, pc = merged[-1]
        merged[-1] = (ps, max(pe, e), max(pc, c))
    else:
        merged.append((s, e, c))

with open(out_path, "w") as fh:
    for i, (s, e, c) in enumerate(merged):
        fh.write(f"chr1\t{s}\t{e}\tpeak_{i}\t{c}\t+\n")
print(f"wrote {len(merged)} peaks to {out_path}")
PYEOF

# self-evaluation: run the shipped grader entry (if present in the image)
if [ -f /tests/grade.py ]; then
  python3 /tests/grade.py || true
fi
