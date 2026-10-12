Synthetic ChIP-seq input tracks for the golden fixture task.

Contents
--------
signal.bedgraph   1.6 MB, ~79k lines. Per-10bp-bin coverage of the
                  immunoprecipitation (IP) track over a synthetic 1 Mb
                  region of chr1. Contains 12 embedded enrichment peaks
                  (deterministic, numpy seed 42).
control.bedgraph  1.6 MB, ~79k lines. Matching input/control coverage
                  track: same background law (exponential, seed 43), no
                  embedded peaks.
gen_inputs.py     Deterministic generator that re-creates both tracks
                  byte-identically (for inspection / provenance).

Format
------
Plain bedGraph: chrom, chromStart (0-based), chromEnd (exclusive),
integer coverage. Comment lines start with '#'. Zero-coverage bins are
omitted.

Provenance
----------
Fully synthetic; no real sequencing data is shipped. The gold peak
list is NOT in this directory - it is re-derived from an independent
seed inside tests/ at verifier time, so an agent solving the task
cannot read the answer from the image.
