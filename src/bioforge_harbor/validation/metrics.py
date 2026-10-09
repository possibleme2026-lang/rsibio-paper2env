"""Pure metric functions for the four-stage protocol (paper sections 4.1-4.4).

Every function here is total and side-effect-free: it takes structured records
and returns a number in a documented range, or ``None`` when the input is empty
(so a caller can never mistake "no data" for "score 0"). This is deliberate —
the paper's Stage-1 table defines rates over *sets of environments*, and a
silent 0.0 on an empty set would be read as a catastrophic result.

Metrics mirror the paper's tables one-to-one so a reader can check code against
the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Iterable


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _rate(flags: Iterable[bool]) -> float | None:
    flags = list(flags)
    return (sum(1 for f in flags if f) / len(flags)) if flags else None


def _safe_mean(xs: list[float]) -> float | None:
    return mean(xs) if xs else None


def _ci95_normal(xs: list[float]) -> tuple[float, float] | None:
    """Wald 95% CI on the mean (n>=2). Adequate for reporting; not a claim of
    statistical rigor beyond descriptive."""
    if len(xs) < 2:
        return None
    m = mean(xs)
    se = pstdev(xs) / (len(xs) ** 0.5)
    return (m - 1.96 * se, m + 1.96 * se)


# --------------------------------------------------------------------------
# Stage 1: environment quality verification (paper 4.1)
# --------------------------------------------------------------------------
def build_success_rate(env_records: list[dict[str, Any]]) -> float | None:
    """Fraction of environments that build and run without error.

    A record is ``{"built": bool, "ran": bool, ...}``.
    """
    return _rate([bool(e.get("built")) and bool(e.get("ran")) for e in env_records])


def baseline_reproduction_consistency(env_records: list[dict[str, Any]]) -> float | None:
    """Mean normalized agreement between reproduced baseline and paper-reported.

    Each record may carry ``baseline`` and ``paper`` floats plus optional
    ``tolerance`` (default 0.15). Returns mean agreement in [0,1].
    """
    vals: list[float] = []
    for e in env_records:
        b, p = e.get("baseline"), e.get("paper")
        if b is None or p is None:
            continue
        tol = float(e.get("tolerance", 0.15)) or 1e-9
        vals.append(max(0.0, 1.0 - abs(b - p) / tol))
    return _safe_mean(vals)


def expert_approval_rate(review_records: list[dict[str, Any]]) -> float | None:
    """Fraction of *reviewed* environments approved by a human expert.

    A record is ``{"reviewed": bool, "approved": bool}``. Only reviewed ones
    count in the denominator.
    """
    reviewed = [r for r in review_records if r.get("reviewed")]
    return _rate([bool(r.get("approved")) for r in reviewed])


def shortcut_success_rate(probe_records: list[dict[str, Any]]) -> float | None:
    """Fraction of environments where a trivial strategy FOOLED the evaluator.

    **LOWER IS BETTER (↓)** — this is a *defect* rate of the grader, not a
    score of the environment. A record is ``{"shortcut_passed": bool}`` where
    True means the shortcut slipped through (bad). Keeping the name honest is
    what stops a paper table from reporting a defect rate as a win.
    """
    return _rate([bool(p.get("shortcut_passed")) for p in probe_records])


def shortcut_detection_rate(probe_records: list[dict[str, Any]]) -> float | None:
    """Fraction of environments where the shortcut was caught (↑, = 1 - success).

    Reported alongside :func:`shortcut_success_rate` so either convention can
    be read off without re-deriving it; both are emitted by ``stage1_summary``.
    """
    s = shortcut_success_rate(probe_records)
    return None if s is None else 1.0 - s


def stage1_summary(env_records, review_records, probe_records) -> dict[str, Any]:
    return {
        "stage": 1,
        "n_envs": len(env_records),
        "build_success_rate": build_success_rate(env_records),
        "baseline_reproduction_consistency": baseline_reproduction_consistency(env_records),
        "expert_approval_rate": expert_approval_rate(review_records),
        # both directions of the shortcut signal, so nobody has to guess which
        # way the arrow points when transcribing this into a table
        "shortcut_success_rate": shortcut_success_rate(probe_records),      # ↓ defect rate
        "shortcut_detection_rate": shortcut_detection_rate(probe_records),  # ↑ caught rate
    }


# --------------------------------------------------------------------------
# Stage 2: weight-free self-improvement (paper 4.2)
# --------------------------------------------------------------------------
@dataclass
class Attempt:
    model: str
    env: str
    round: int          # 1..K
    reward: float
    exceeded_baseline: bool = False


def first_attempt_improvement_rate(attempts: list[Attempt],
                                   baseline: dict[str, float]) -> float | None:
    """Fraction of environments where *at least one* model improved by round 2.

    Mirrors RSI-Forge's 'improved after the first attempt in X% of environments'.
    An environment counts as improved if any model's round-2 reward exceeds its
    round-1 reward (improvement signal exists), independent of weight updates.
    """
    envs = sorted({a.env for a in attempts})
    if not envs:
        return None
    by: dict[tuple[str, str, int], Attempt] = {(a.model, a.env, a.round): a for a in attempts}
    models = sorted({a.model for a in attempts})
    improved = 0
    for env in envs:
        hit = False
        for m in models:
            r1 = by.get((m, env, 1))
            r2 = by.get((m, env, 2))
            if r1 and r2 and r2.reward > r1.reward:
                hit = True
                break
        improved += int(hit)
    return improved / len(envs)


def exceeded_baseline_rate(attempts: list[Attempt],
                           baseline: dict[str, float]) -> float | None:
    """Fraction of (env) where some model's best reward exceeds the reproduced
    baseline (paper: 'in 68 of 120 environments, model performance exceeded the
    reproduced baseline')."""
    envs = sorted({a.env for a in attempts})
    if not envs:
        return None
    exceeded = 0
    for env in envs:
        best = max((a.reward for a in attempts if a.env == env), default=None)
        base = baseline.get(env)
        if best is not None and base is not None and best > base:
            exceeded += 1
    return exceeded / len(envs)


def continuous_improvement_rate(attempts: list[Attempt]) -> float | None:
    """Fraction of (model, env) trajectories that improve monotonically across
    all K rounds (a stricter signal than 'improved once')."""
    groups: dict[tuple[str, str], list[Attempt]] = {}
    for a in attempts:
        groups.setdefault((a.model, a.env), []).append(a)
    rates: list[float] = []
    for key, grp in groups.items():
        grp = sorted(grp, key=lambda a: a.round)
        rewards = [a.reward for a in grp]
        if len(rewards) < 2:
            continue
        monotone = all(rewards[i + 1] > rewards[i] for i in range(len(rewards) - 1))
        rates.append(1.0 if monotone else 0.0)
    return _safe_mean(rates)


def stage2_summary(attempts: list[Attempt], baseline: dict[str, float]) -> dict[str, Any]:
    return {
        "stage": 2,
        "n_models": len({a.model for a in attempts}),
        "n_envs": len({a.env for a in attempts}),
        "n_rounds": max((a.round for a in attempts), default=0),
        "first_attempt_improvement_rate": first_attempt_improvement_rate(attempts, baseline),
        "exceeded_baseline_rate": exceeded_baseline_rate(attempts, baseline),
        "continuous_improvement_rate": continuous_improvement_rate(attempts),
    }


# --------------------------------------------------------------------------
# Stage 3: harness optimization (paper 4.3)
# --------------------------------------------------------------------------
def variance_decomposition(cell_means: dict[tuple[str, str], float]) -> dict[str, Any]:
    """Decompose observed spread into 'model differences' vs 'harness differences'.

    ``cell_means`` maps ``(model, harness) -> mean reward``. Returns between-model
    and between-harness variance components (simple two-way, one obs/cell). The
    paper's testable claim from HarnessOpt-Bench is whether model variance
    exceeds harness variance; this function makes that comparison numeric.
    """
    if not cell_means:
        return {"model_variance": None, "harness_variance": None, "dominant": None}
    models = sorted({m for m, _ in cell_means})
    harnesses = sorted({h for _, h in cell_means})
    overall = mean(cell_means.values())

    model_marginal = {m: mean([cell_means[(m, h)] for h in harnesses if (m, h) in cell_means])
                      for m in models}
    harness_marginal = {h: mean([cell_means[(m, h)] for m in models if (m, h) in cell_means])
                        for h in harnesses}

    model_var = _var([model_marginal[m] for m in model_marginal], overall)
    harness_var = _var([harness_marginal[h] for h in harness_marginal], overall)

    dominant = None
    if model_var is not None and harness_var is not None:
        dominant = "model" if model_var >= harness_var else "harness"
    return {"model_variance": model_var, "harness_variance": harness_var, "dominant": dominant}


def _var(xs: list[float], mu: float) -> float | None:
    if len(xs) < 2:
        return None
    return mean([(x - mu) ** 2 for x in xs])


def harness_delta(baseline: dict[str, float], variant: dict[str, float]) -> dict[str, Any]:
    """Per-key and aggregate delta between two harness variants (same keys)."""
    keys = sorted(set(baseline) | set(variant))
    per = {k: (variant.get(k, 0.0) - baseline.get(k, 0.0)) for k in keys}
    deltas = list(per.values())
    return {
        "per_key": per,
        "mean_delta": _safe_mean(deltas),
        "n_improved": sum(1 for d in deltas if d > 0),
        "n_regressed": sum(1 for d in deltas if d < 0),
    }


# --------------------------------------------------------------------------
# Stage 4: post-training with held-out benchmarks (paper 4.4)
# --------------------------------------------------------------------------
def time_split(records: list[dict[str, Any]], *, train_cutoff: int,
               holdout_start: int, holdout_end: int) -> dict[str, list[str]]:
    """Split paper ids by publication year (deterministic, no leakage).

    - train: year <= train_cutoff  (paper: 'published before 2024')
    - holdout: holdout_start <= year <= holdout_end ('2025-2026', locked pre-train)
    - excluded: everything in the gap or out of both windows (recorded, not
      silently dropped)
    """
    train, holdout, excluded = [], [], []
    for r in records:
        pid = r.get("paper_id", "")
        year = int(r.get("pub_year") or r.get("year") or 0)
        if year and year <= train_cutoff:
            train.append(pid)
        elif year and holdout_start <= year <= holdout_end:
            holdout.append(pid)
        else:
            excluded.append(pid)
    return {"train": train, "holdout": holdout, "excluded": excluded}


def ngram_contamination(text_a: str, text_b: str, *, n: int = 8) -> float:
    """Fraction of n-grams of text_a that also appear in text_b.

    A crude but reproducible contamination probe: high overlap means the
    'held-out' text shares long spans with training text. Returns 0..1.
    """
    ga = _ngrams(text_a, n)
    gb = _ngrams(text_b, n)
    if not ga:
        return 0.0
    return len(ga & gb) / len(ga)


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = (text or "").lower().split()
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def control_contrast(arm_scores: dict[str, float]) -> dict[str, Any]:
    """Compare the paper's five control arms (4.4 table) against each other.

    ``arm_scores`` maps arm name -> mean held-out score. Reports each arm's
    lift over 'no_environment' (the baseline capability arm) so the causal
    claims ('auto vs random', 'auto+review vs auto') are numeric.
    """
    base = arm_scores.get("no_environment")
    out = {"arms": dict(arm_scores), "lift_over_no_env": {}}
    if base is None:
        return out
    for arm, score in arm_scores.items():
        out["lift_over_no_env"][arm] = round(score - base, 6)
    return out
