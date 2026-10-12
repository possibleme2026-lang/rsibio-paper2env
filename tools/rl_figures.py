"""Paper-grade figures for the tinker RL experiment.

Reads  out/rl_runs/experiment_log.json   (training telemetry)
       out/rl_runs/experiment_eval.json  (held-out eval + bootstrap)
Writes paper/figures/rl_fig{1,2,3}_*.{pdf,png}  (vector PDF for LaTeX)

Reward decomposition is recovered exactly from the recorded fields:
  schema component  = 0.30 if schema_ok else reward   (0 or 0.10 partial)
  gates component   = 0.45 if gates_ok else 0
  quality component = 0.25 * fidelity
(the reward_fn only reaches gates/quality after schema passes, so the sum
reconstructs the total for every sample).

Usage: python tools/rl_figures.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "out" / "rl_runs"
FIG_DIR = REPO / "paper" / "figures"

BASE_C = "#4C72B0"   # colorblind-safe blue
RL_C = "#DD8452"     # colorblind-safe orange
GOOD_C = "#55A868"
METRICS = ("reward", "schema_ok", "gates_ok", "fidelity")
LABELS = {
    "reward": "Total reward",
    "schema_ok": "Schema-valid rate",
    "gates_ok": "Gates pass rate",
    "fidelity": "Provenance fidelity",
}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}")
    plt.close(fig)
    print(f"[fig] {stem}.pdf/.png", flush=True)


def boot_ci(values, n=10000, seed=7):
    rng = random.Random(seed)
    vals = list(values)
    if len(vals) <= 1:
        m = sum(vals) / len(vals) if vals else 0.0
        return m, m, m
    means = []
    for _ in range(n):
        idx = [rng.randrange(len(vals)) for _ in vals]
        means.append(sum(vals[i] for i in idx) / len(idx))
    means.sort()
    return (sum(vals) / len(vals),
            means[int(0.025 * n)], means[int(0.975 * n)])


# ---------------------------------------------------------------- fig 1
def fig_training(log):
    steps = [s["step"] for s in log["steps"]]
    r = [s["mean_reward"] for s in log["steps"]]
    sr = [s["schema_rate"] for s in log["steps"]]
    gr = [s["gates_rate"] for s in log["steps"]]

    def roll(xs, w=5):
        out = []
        for i in range(len(xs)):
            lo = max(0, i - w + 1)
            out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
        return out

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax1.plot(steps, r, "o", ms=2.5, alpha=0.30, color=BASE_C,
             label="per-step mean")
    ax1.plot(steps, roll(r), "-", lw=1.8, color=RL_C,
             label="rolling mean (w=5)")
    grad = [s for s in log["steps"] if s["groups_kept"] > 0]
    if grad:
        ax1.scatter([s["step"] for s in grad],
                    [s["mean_reward"] for s in grad],
                    marker="v", s=14, color=GOOD_C, zorder=5,
                    label=f"gradient applied ({len(grad)} steps)")
    ax1.set_xlabel("GRPO step")
    ax1.set_ylabel("Reward (training samples)")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(frameon=False, loc="lower right")
    ax1.set_title("(a) Training reward")

    ax2.plot(steps, sr, "-o", ms=2.5, lw=1.5, color=BASE_C,
             label="schema-valid")
    ax2.plot(steps, gr, "-s", ms=2.5, lw=1.5, color=RL_C,
             label="emit + gates pass")
    ax2.set_xlabel("GRPO step")
    ax2.set_ylabel("Fraction of samples")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(frameon=False, loc="lower right")
    ax2.set_title("(b) Verifiable-reward components")
    fig.tight_layout()
    save(fig, "rl_fig1_training")


# ---------------------------------------------------------------- fig 2
def fig_heldout(ev):
    pp = ev["per_paper_means"]
    base, rl = pp["base"], pp["rl"]
    papers = sorted(base)
    boot = ev["paired_bootstrap_rl_minus_base"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) grouped bars with per-arm 95% CI (bootstrap over papers)
    x = np.arange(len(METRICS))
    w = 0.36
    for i, (arm, col) in enumerate((("base", BASE_C), ("rl", RL_C))):
        mus, los, his = [], [], []
        for m in METRICS:
            mu, lo, hi = boot_ci([base[p][m] for p in papers] if arm == "base"
                                 else [rl[p][m] for p in papers])
            mus.append(mu); los.append(mu - lo); his.append(hi - mu)
        ax1.bar(x + (i - 0.5) * w, mus, w, color=col, alpha=0.9,
                label="base (gpt-oss-20b)" if arm == "base" else "RL post-trained",
                yerr=[los, his], capsize=2.5,
                error_kw={"elinewidth": 0.9})
    for j, m in enumerate(METRICS):
        if boot[m]["significant"]:
            ytop = max(
                max(base[p][m] for p in papers), max(rl[p][m] for p in papers))
            ax1.text(j, min(ytop + 0.09, 1.12), "*", ha="center", fontsize=10)
    ax1.set_xticks(x, [LABELS[m].replace(" ", "\n", 1) for m in METRICS])
    ax1.set_ylabel("Held-out papers (mean over 6 papers)")
    ax1.set_ylim(0, 1.18)
    ax1.legend(frameon=False, loc="upper left")
    ax1.set_title("(a) Base vs. RL post-trained")
    ax1.annotate("*: 95% CI of paired delta excludes 0", xy=(0.5, -0.32),
                 xycoords="axes fraction", ha="center", fontsize=7,
                 color="0.35")

    # (b) per-paper slope chart for total reward
    for p in papers:
        y0, y1 = base[p]["reward"], rl[p]["reward"]
        ax2.plot([0, 1], [y0, y1], "-o", ms=3.5, lw=1.1,
                 color=GOOD_C if y1 >= y0 else "#C44E52", alpha=0.85)
    ax2.set_xticks([0, 1], ["base", "RL"])
    ax2.set_ylabel("Total reward")
    ax2.set_xlim(-0.25, 1.25)
    ax2.set_title("(b) Reward per held-out paper")
    mu0 = sum(base[p]["reward"] for p in papers) / len(papers)
    mu1 = sum(rl[p]["reward"] for p in papers) / len(papers)
    ax2.scatter([0, 1], [mu0, mu1], marker="D", s=26, color="k", zorder=6)
    ax2.annotate(f"{mu0:.2f}", (0, mu0), xytext=(-12, 4),
                 textcoords="offset points", fontsize=7)
    ax2.annotate(f"{mu1:.2f}", (1, mu1), xytext=(5, 4),
                 textcoords="offset points", fontsize=7)
    fig.tight_layout()
    save(fig, "rl_fig2_heldout")


# ---------------------------------------------------------------- fig 3
def fig_decomposition(ev):
    ps = ev["per_sample"]
    comps = ("schema", "gates", "quality")
    comp_lab = {"schema": "Schema validity (0.30)",
                "gates": "Emit + four-gate verify (0.45)",
                "quality": "Provenance fidelity (0.25)"}
    means = {}
    for arm in ("base", "rl"):
        rows = ps[arm]
        if not rows:
            means[arm] = {c: 0.0 for c in comps}
            continue
        mc = {}
        for c in comps:
            if c == "schema":
                vals = [0.30 if s["schema_ok"] else s["reward"] for s in rows]
            elif c == "gates":
                vals = [0.45 if s["gates_ok"] else 0.0 for s in rows]
            else:
                vals = [0.25 * s["fidelity"] for s in rows]
            mc[c] = sum(vals) / len(vals)
        means[arm] = mc
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    x = np.arange(2)
    bottom = np.zeros(2)
    colors = {"schema": BASE_C, "gates": RL_C, "quality": GOOD_C}
    for c in comps:
        vals = np.array([means["base"][c], means["rl"][c]])
        ax.bar(x, vals, 0.5, bottom=bottom, color=colors[c],
               label=comp_lab[c], alpha=0.92)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.02:
                ax.text(xi, b + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white")
        bottom += vals
    ax.set_xticks(x, ["base\n(gpt-oss-20b)", "RL post-trained"])
    ax.set_ylabel("Mean reward contribution")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("Reward decomposition (held-out)")
    fig.tight_layout()
    save(fig, "rl_fig3_decomposition")


# ---------------------------------------------------------------- main
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log = json.loads((LOG_DIR / "experiment_log.json").read_text(encoding="utf-8"))
    ev = json.loads((LOG_DIR / "experiment_eval.json").read_text(encoding="utf-8"))
    fig_training(log)
    fig_heldout(ev)
    fig_decomposition(ev)

    # text summary next to the figures
    boot = ev["paired_bootstrap_rl_minus_base"]
    lines = ["# RL experiment results (auto-generated by tools/rl_figures.py)", ""]
    lines.append(f"base model: {ev['base_model']}  |  RL ckpt: {ev['rl_checkpoint']}")
    lines.append(f"held-out papers: {len(ev['split']['held_out'])}  |  "
                 f"samples/paper: {ev['config']['eval_samples_per_paper']}  |  "
                 f"eval temperature: {ev['config']['eval_temperature']}")
    lines.append("")
    lines.append("| metric | base | RL | delta | 95% CI | significant |")
    lines.append("|---|---|---|---|---|---|")
    for m in METRICS:
        s = ev["summary"][m]
        b = boot[m]
        sig = "yes" if b["significant"] else "no"
        lines.append(f"| {LABELS[m]} | {s['base']:.3f} | {s['rl']:.3f} | "
                     f"+{b['mean_delta']:.3f} | [{b['ci95'][0]:.3f}, "
                     f"{b['ci95'][1]:.3f}] | {sig} |")
    (REPO / "out" / "rl_runs" / "RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("[done] out/rl_runs/RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
