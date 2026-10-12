# Omega-seq: UMI-error-corrected molecule counting and transcript-end mapping

You are the computational analyst for an Omega-seq single-cell RNA-seq project.
Omega-seq restricts unique-molecular-identifier (UMI) incorporation to reverse
transcription, so distinct UMI sequences correspond to distinct RNA molecules
and the library carries almost no phantom-UMI background. You are handed one
UMI-tagged, cell-barcoded alignment file and asked to produce a
per-cell / per-gene molecule count table, a 3'-end profile, and a small QC
summary, following the pipeline conventions below **exactly**.

## Inputs

Everything lives under `/app/data` (do not modify it):

* `aligned.sam` (~20 000 records, SAM v1.6, single-end 90 nt reads). Each record
  carries `CB:Z:<16 nt cell barcode>` and `UB:Z:<8 nt UMI>` tags. MAPQ is 60 for
  good alignments and 0-10 for alignments that must be discarded. A few percent
  of reads fall in intergenic space. Alignments target `chr2L`, `chr3R`,
  `chrERCC`.
* `annotation.gtf` (9-column, 1-based inclusive). Contains `gene` and `exon`
  features for six Drosophila genes (`dpp`, `wg`, `numb`, `pros`, `dpn`, `ase`)
  and twenty ERCC spike-ins (`ERCC-00001` ... `ERCC-00020`). Gene models have
  one to three exons and are separated by >30 kb of intergenic space.
* `library_meta.tsv` - columns `cell_barcode`, `library`, `input_fg`. Libraries
  with `input_fg == 0` are negative controls.
* `ercc_expected.tsv` - columns `ercc_id`, `expected_molecules_IN1000`, the
  number of ERCC molecules that should exist in library `IN1000`.

## Task

Implement the Omega-seq counting pipeline and write three files under
`/app/output`.

1. **Read filtering and gene assignment.** Keep SAM records that are mapped
   (FLAG 0x4 clear) and have `MAPQ >= 20`. Assign each kept read to the gene on
   the same chromosome whose exonic intervals overlap the read's aligned
   reference span (`POS` .. `POS + reference-consumed CIGAR length - 1`) by the
   greatest number of bases; ties go to the gene with the smaller start
   coordinate. Reads with zero overlap are dropped.
2. **3' end.** A read's 3' end is the 3'-most aligned reference base with
   respect to the transcription direction of the gene it was assigned to:
   `POS + ref_len - 1` when FLAG 0x10 is clear, and `POS` when FLAG 0x10 is set.
   Soft-clipped (`S`) and hard-clipped (`H`) bases are not aligned and do not
   contribute to `ref_len`.
3. **UMI error correction.** Within each `(cell, gene)` group, two distinct UMI
   sequences belong to the same UMI family when they differ by exactly one
   substitution, or when they are connected through a chain of such pairs - i.e.
   the connected components of the Hamming-distance-1 graph over the distinct
   UMI sequences of that group.
4. **Molecule counting.** A molecule is one `(cell, gene, UMI family)` triple.
   Report, for every `(cell, gene)` pair with at least one assigned read, the
   molecule count and the assigned-read count.
5. **Transcript-end profile.** Bin every assigned read's 3' end into 5 nt windows
   with `bin = (end // 5) * 5`. For each `(cell, gene)` and each bin, count how
   many *distinct UMI families* have at least one read in that bin. Report the
   modal bin (ties broken by the smaller coordinate) and its family count.
6. **QC metrics.** Report the total molecule count, the total number of assigned
   reads, the deduplication rate `1 - molecules/reads`, the total molecule count
   of the negative-control libraries, and the phantom-UMI index
   `sum over (cell, gene, bin) of max(0, n_families_in_bin - 1) / total_molecules`.

## Constraints

1. `counts.tsv` must have exactly the tab-separated header
   `cell\tgene\tmolecules\treads`, integer values with `molecules >= 1` and
   `reads >= 1`, and no duplicate `(cell, gene)` rows.
2. `counts.tsv` must contain exactly one row for every `(cell, gene)` pair that
   has at least one assigned read; rows for pairs with no assigned reads are not
   allowed.
3. UMI families must be built by Hamming-distance-1 collapsing. Counting exact
   UMI strings (no error correction) is not acceptable.
4. `end_profile.tsv` must have exactly the tab-separated header
   `cell\tgene\tstrand\tmodal_end\tmodal_families` and must report the modal
   5 nt end bin for every `(cell, gene)` pair present in `counts.tsv`.
5. The negative-control libraries (`input_fg == 0`) must contain at most 5
   molecules in total.
6. The Spearman correlation between the molecules you report for the twenty
   ERCC genes in library `IN1000` and `expected_molecules_IN1000` must be at
   least 0.90.
7. `metrics.json` must be a JSON object containing the numeric keys
   `total_molecules`, `total_reads_assigned`, `dedup_rate`, `phantom_index`
   and `nc_molecules`.

## Output

* `/app/output/counts.tsv`
  ```
  cell	gene	molecules	reads
  AAACCTGAGAAACCAT-2	dpp	17	64
  AAACCTGAGAAACCAT-2	numb	14	70
  ```
* `/app/output/end_profile.tsv`
  ```
  cell	gene	strand	modal_end	modal_families
  AAACCTGAGAAACCAT-2	dpp	+	127995	9
  ```
* `/app/output/metrics.json`
  ```json
  {"total_molecules": 3458, "total_reads_assigned": 19123,
   "dedup_rate": 0.8192, "phantom_index": 0.09, "nc_molecules": 1}
  ```

## Scoring

`tests/grade.py` independently recomputes the reference answer from `/app/data`
and awards graded partial credit: `counts.tsv` schema (10%), exact total molecule
count (15%), per-`(cell,gene)` molecule agreement (25%), negative-control
background (10%), transcript-end profile (15%), ERCC spike-in correlation (10%)
and `metrics.json` QC values (15%).
