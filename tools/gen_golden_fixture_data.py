"""One-shot generator for the golden fixture's deterministic input data.

Writes two bedGraph tracks over a synthetic 1 Mb region (chrom chr1, 0-based
half-open coordinates, 10 bp bins):
  signal.bedgraph  - IP coverage: exponential background + 12 embedded
                     enrichment peaks (seed 42)
  control.bedgraph - input/control coverage: same background law, no peaks
                     (seed 43)

The gold peak list is NOT written here; tests/gold_gen.py re-derives it from
an independent seed (1337) at verifier time so the agent can never read it.
"""

from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "golden_task" / "environment" / "data"

REGION = 1_000_000
BIN = 10
N_BINS = REGION // BIN

# The 12 embedded peaks: (start_bp, length_bp, fold_enrichment)
PEAKS = [
    (48_000, 900, 14.0), (121_000, 600, 9.0), (203_500, 1_400, 18.0),
    (276_000, 500, 7.5), (341_000, 1_100, 11.0), (409_500, 800, 8.0),
    (487_000, 1_600, 21.0), (562_000, 700, 9.5), (633_000, 1_200, 13.0),
    (718_000, 550, 7.0), (802_500, 1_300, 16.0), (919_000, 950, 10.5),
]

def coverage(rng: np.random.Generator, peaks) -> np.ndarray:
    lam = rng.exponential(2.5, N_BINS) + 0.3          # background coverage
    for start, length, fold in peaks:
        s, e = start // BIN, (start + length) // BIN
        width = max(e - s, 1)
        ramp = np.hanning(width * 3)[width:]           # smooth shoulder profile
        lam[s:e] += fold * (1.0 + 3.0 * ramp[:width]) # plateau + shoulders
    return rng.poisson(lam).astype(np.int32)

def write_bedgraph(path: Path, counts: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("# synthetic ChIP-seq coverage; seed 42; chrom=chr1; bin=10bp\n")
        for i, c in enumerate(counts):
            if c == 0:
                continue
            fh.write(f"chr1\t{i * BIN}\t{(i + 1) * BIN}\t{c}\n")

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sig = coverage(np.random.default_rng(42), PEAKS)
    ctrl = coverage(np.random.default_rng(43), [])
    write_bedgraph(OUT / "signal.bedgraph", sig)
    write_bedgraph(OUT / "control.bedgraph", ctrl)
    for name in ("signal.bedgraph", "control.bedgraph"):
        p = OUT / name
        print(f"{name}: {p.stat().st_size} bytes, {sum(1 for _ in p.open())} lines")

if __name__ == "__main__":
    main()
