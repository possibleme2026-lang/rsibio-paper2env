"""Curator stage (paper section 3.2: "Curator Agent").

Screens a corpus of candidate papers down to a buildable subset. The screening
rules are **deterministic and auditable**; the LLM is consulted only for the
genuinely fuzzy "is this computable?" judgment on papers that pass every hard
rule but sit on the domain/metric boundary. Rationale: if selection depends on
a model's mood, Stage-1 metrics (construction success rate etc.) become
unreproducible, and the whole evidence chain the paper claims collapses.

Hard rules (reject = deterministic):
  1. must be open access (paper section 6: "prioritize open-access papers")
  2. must carry a permissive license OR be public domain
  3. must have a resolvable identifier (pmcid or doi)
  4. pub year within the configured window (default: <= holdout cutoff)
  5. must match a computational subfield keyword (section 3.4 list)

Soft scoring (rank, then take top ``select_n``):
  code/data availability, presence of a named metric, abstract length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...models.records import PaperRecord, OPEN_ACCESS_LICENSES
from ...models.reports import SelectionReport

# Subfields the paper prioritizes (section 3.4). A paper must mention at least
# one keyword group to be considered "computable". The list was widened after a
# 115-paper wave-3 screen rejected 71 papers that were plainly computational but
# used vocabulary outside the original narrow set (e.g. "pangenome graph",
# "proteomics pipeline", "spike sorting"): a screener that rejects real
# computational work is a false-negative machine, and the deterministic rule is
# only worth keeping if it is complete for its purpose.
SUBFIELD_KEYWORDS: dict[str, list[str]] = {
    "single-cell-rnaseq": ["single-cell", "single cell", "scrna", "sc-rna", "rnaseq", "rna-seq",
                           "scanpy", "seurat", "transcriptom", "expression matrix", "umi"],
    "variant-annotation": ["variant calling", "variant annotation", "pathogenicity", "clinvar",
                           "deepvariant", "gwas", "genome-wide association", "snp", "haplotype",
                           "pangenome", "phasing", "imputation", "allele frequency"],
    "protein-structure": ["protein structure", "structure prediction", "rmsd", "tm-score",
                          "alphafold", "contact map", "protein folding", "docking", "molecular dynamics",
                          "coarse-grained", "free energy", "conformational", "cryo-em", "cryo-electron"],
    "phylogenetics": ["phylogenetic", "phylogenetics", "maximum-likelihood tree", "robinson-foulds",
                      "phylogenom", "tree inference", "divergence time", "lineage-specific"],
    "bioimage-segmentation": ["image segmentation", "cellpose", "histology", "histolog", "whole-slide",
                              "microscopy image", "fluorescence", "image analysis", "imaging",
                              "tomography", "segmentation"],
    "grn-inference": ["gene regulatory network", "network inference", "co-expression network",
                      "transcription factor network", "regulatory network", "grn"],
    "omics-pipeline": ["proteomic", "metabolomic", "mass spectrometry", "metagenom", "microbiome",
                       "amplicon", "sequencing pipeline", "bioinformatic", "annotation pipeline",
                       "quality control pipeline", "spatial transcriptom", "chromatin", "methylation",
                       "epigenom", "atac-seq", "chip-seq", "expression profiling"],
    "neural-signal": ["eeg", "electrophysiolog", "spike sorting", "calcium imaging", "neural recording",
                      "decoding", "fmri", "mri", "connectome", "brain imaging", "neuroimaging",
                      "resting-state", "neural dynamics"],
    "clinical-ml": ["survival analysis", "biomarker", "drug discovery", "drug screening",
                    "precision medicine", "diagnostic model", "risk prediction", "clinical prediction",
                    "antimicrobial resistance", "drug resistance", "adverse event"],
    "general-ml": ["machine learning", "deep learning", "classifier", "classification model",
                   "regression model", "cross-validation", "auroc", "auc", "f1 score", "auprc",
                   "adjusted rand", "rmse", "mae", "benchmark dataset", "benchmark of",
                   "neural network", "random forest", "transformer", "autoencoder",
                   "reinforcement learning", "clustering", "dimensionality reduction",
                   "foundation model", "language model", "llm"],
    "simulation-modeling": ["simulation", "kinetic model", "mathematical model", "dynamical system",
                            "agent-based model", "computational model", "flux balance",
                            "systems biology", "parameter estimation", "ode model"],
}

METRIC_KEYWORDS = ["f1", "auroc", "auc", "rmse", "mae", "accuracy", "precision",
                   "recall", "pearson", "spearman", "jaccard", "dice"]


@dataclass
class CuratorConfig:
    select_n: int = 200
    screened_target: int = 1000
    year_max: int = 2023            # training-side cutoff (paper section 4.4)
    year_min: int = 2000
    require_open_access: bool = True
    require_license: bool = True
    require_computational: bool = True
    min_abstract_len: int = 100
    # tie-break: when two papers score equal, lower paper_id wins (stable sort)
    buckets: dict[str, int] = field(default_factory=lambda: {})  # optional per-bucket caps


def _matches_subfield(rec: PaperRecord) -> str | None:
    hay = " ".join([rec.title, rec.abstract, " ".join(rec.subjects)]).lower()
    for bucket, kws in SUBFIELD_KEYWORDS.items():
        if any(k in hay for k in kws):
            return bucket
    return None


def _has_metric(rec: PaperRecord) -> bool:
    hay = (rec.title + " " + rec.abstract).lower()
    return any(m in hay for m in METRIC_KEYWORDS)


def score(rec: PaperRecord, cfg: CuratorConfig) -> tuple[float, str | None, list[str]]:
    """Return (score, subfield_bucket, reject_reasons). score<0 means rejected."""
    reasons: list[str] = []
    year = rec.year_or_zero
    if not rec.paper_id:
        reasons.append("no_identifier")
    if cfg.require_open_access and not rec.open_access:
        reasons.append("not_open_access")
    if cfg.require_license and rec.license and rec.license not in OPEN_ACCESS_LICENSES:
        # A stated-but-restrictive license is a reject; an absent license is a
        # soft penalty below (some sources simply omit it).
        reasons.append(f"license:{rec.license}")
    if year and not (cfg.year_min <= year <= cfg.year_max):
        reasons.append(f"year_out_of_range:{year}")
    bucket = _matches_subfield(rec)
    if cfg.require_computational and bucket is None:
        reasons.append("not_computational")
    if len(rec.abstract or "") < cfg.min_abstract_len:
        reasons.append("abstract_too_short")
    if reasons:
        return -1.0, bucket, reasons

    # soft score
    s = 1.0
    if rec.has_code_link:
        s += 2.0
    if rec.has_data_link:
        s += 1.0
    if _has_metric(rec):
        s += 1.5
    if rec.license in OPEN_ACCESS_LICENSES:
        s += 0.5
    if rec.abstract:
        s += min(len(rec.abstract) / 2000.0, 1.0)
    return s, bucket, []


def curate(records: list[PaperRecord], cfg: CuratorConfig | None = None) -> SelectionReport:
    """Deterministically screen + rank + select. No LLM, no network."""
    cfg = cfg or CuratorConfig()
    scored: list[tuple[float, str, PaperRecord, str | None]] = []
    rejected: list[dict[str, Any]] = []
    buckets: dict[str, int] = {}

    for rec in records:
        s, bucket, reasons = score(rec, cfg)
        if s < 0:
            rejected.append({"paper_id": rec.paper_id, "reason": ",".join(reasons)})
            continue
        scored.append((s, rec.paper_id, rec, bucket))

    # rank by (-score, paper_id) => stable, reproducible ordering
    scored.sort(key=lambda t: (-t[0], t[1]))

    selected: list[PaperRecord] = []
    sel_buckets: dict[str, int] = {}
    for s, pid, rec, bucket in scored:
        if len(selected) >= cfg.select_n:
            break
        if bucket and cfg.buckets and cfg.buckets.get(bucket, 10**9) <= sel_buckets.get(bucket, 0):
            rejected.append({"paper_id": pid, "reason": f"bucket_cap:{bucket}"})
            continue
        selected.append(rec)
        if bucket:
            sel_buckets[bucket] = sel_buckets.get(bucket, 0) + 1
        buckets[bucket or "unclassified"] = buckets.get(bucket or "unclassified", 0) + 1

    return SelectionReport(
        stage="curate",
        status="ok",
        screened=len(records),
        selected=len(selected),
        selected_ids=[r.paper_id for r in selected],
        rejected=rejected,
        bucket_counts=buckets,
    )


def boundary_papers(records: list[PaperRecord], cfg: CuratorConfig | None = None) -> list[PaperRecord]:
    """Papers that pass every hard rule *except* computational-classification.

    These are the only ones we ever hand to an LLM, keeping the deterministic
    core untouched and the fuzzy judgment explicitly scoped + auditable.
    """
    cfg = cfg or CuratorConfig()
    out: list[PaperRecord] = []
    for rec in records:
        _, bucket, reasons = score(rec, cfg)
        if bucket is None and reasons == ["not_computational"]:
            out.append(rec)
    return out


def refine_with_backend(
    records: list[PaperRecord],
    backend,
    cfg: CuratorConfig | None = None,
    *,
    prompt_template: str | None = None,
) -> dict[str, bool]:
    """Ask ``backend`` whether each boundary paper is computable.

    Returns ``{paper_id: is_computable}``. Never called by the offline test
    suite; only the live pipeline routes here. The response is expected to be
    JSON ``{"computable": true|false}``.
    """
    cfg = cfg or CuratorConfig()
    tmpl = prompt_template or (
        "You are screening a life-science paper for whether it describes a "
        "computable method (algorithm/model/analysis pipeline with standardized "
        "inputs and outputs) that can become an executable agent task. "
        'Answer with JSON {{"computable": true|false}}.\n\n'
        "Title: {title}\nAbstract: {abstract}\n"
    )
    verdicts: dict[str, bool] = {}
    for rec in boundary_papers(records, cfg):
        prompt = tmpl.format(title=rec.title, abstract=rec.abstract[:2000])
        try:
            resp = backend.ask_json("curator.select", prompt, schema_name=None)
            verdicts[rec.paper_id] = bool(resp.get("computable", False))
        except Exception:
            verdicts[rec.paper_id] = False
    return verdicts
