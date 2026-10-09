"""Four-stage validation protocol (paper section 4).

Each stage is a *runner + metric computer*, not a training loop. The framework
defines the inputs (environment sets, attempt logs, harness variants, time
splits), computes the paper's exact metrics from structured result files, and
emits reports. Actual heavy execution (Docker builds, model rollouts, RL
training) is left as pluggable executors — this layer is what makes the
protocol reproducible and testable offline.
"""

from __future__ import annotations

from . import metrics  # noqa: F401
