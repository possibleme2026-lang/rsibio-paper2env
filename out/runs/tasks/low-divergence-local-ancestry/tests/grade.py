#!/usr/bin/env python3
"""Grader for the low-divergence local-ancestry task.

The reference decoder is recomputed from a protected copy of the simulator
(tests/gold_gen.py, fixed seed 20260730), so the gold standard is never
readable by the agent and tampering with /app/data cannot move it.
Partial credit is awarded per constraint item.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gold_gen

SEED = 20260730
LEVELS = ["fst_0002", "fst_0020", "fst_0150"]
SAMPLES = ["hap1", "hap2", "hap3", "hap4"]
GRID = list(range(1, 121))
PSEUDO = 0.5
TOL_POST = 1e-4
TOL_ACC = 2e-3
LL_SLACK = 0.5
TOL_LL = 1.0
FEASIBLE_ACC = 0.95
OUT = "/app/output"
REWARD_PATH = "/logs/verifier/reward.json"
REPORT_PATH = "/logs/verifier/report.json"
WEIGHTS = {"format": 0.08, "posterior": 0.15, "viterbi": 0.12, "posterior_path": 0.08,
           "lambda_hat": 0.15, "switch_counts": 0.10, "accuracy": 0.10,
           "hmm_vs_mle": 0.04, "aggregates": 0.10, "diagnostics": 0.08}


def emission(x, pA, pB):
    e = np.empty((x.shape[0], 2), dtype=float)
    e[:, 0] = np.where(x == 1, pA, 1.0 - pA)
    e[:, 1] = np.where(x == 1, pB, 1.0 - pB)
    return e


def forward_backward(e, lam, dist):
    m = e.shape[0]
    q = 1.0 - np.exp(-lam * dist)
    alpha = np.empty((m, 2))
    scale = np.empty(m)
    a0 = 0.5 * e[0, 0]
    a1 = 0.5 * e[0, 1]
    s = a0 + a1
    scale[0] = s
    alpha[0, 0] = a0 / s
    alpha[0, 1] = a1 / s
    for l in range(1, m):
        ql = q[l - 1]
        p0 = alpha[l - 1, 0]
        p1 = alpha[l - 1, 1]
        a0 = (p0 * (1.0 - ql) + p1 * ql) * e[l, 0]
        a1 = (p0 * ql + p1 * (1.0 - ql)) * e[l, 1]
        s = a0 + a1
        scale[l] = s
        alpha[l, 0] = a0 / s
        alpha[l, 1] = a1 / s
    beta = np.empty((m, 2))
    beta[m - 1, 0] = 1.0
    beta[m - 1, 1] = 1.0
    for l in range(m - 2, -1, -1):
        ql = q[l]
        b0 = (1.0 - ql) * e[l + 1, 0] * beta[l + 1, 0] + ql * e[l + 1, 1] * beta[l + 1, 1]
        b1 = ql * e[l + 1, 0] * beta[l + 1, 0] + (1.0 - ql) * e[l + 1, 1] * beta[l + 1, 1]
        s = b0 + b1
        beta[l, 0] = b0 / s
        beta[l, 1] = b1 / s
    post = alpha * beta
    post = post / post.sum(axis=1, keepdims=True)
    return post, float(np.log(scale).sum())


def loglik_grid(e, dist, grid):
    lam = np.asarray(grid, dtype=float)
    q = 1.0 - np.exp(-np.outer(lam, dist))
    alpha = np.empty((lam.shape[0], 2))
    alpha[:, 0] = 0.5 * e[0, 0]
    alpha[:, 1] = 0.5 * e[0, 1]
    s = alpha.sum(axis=1)
    alpha[:, 0] /= s
    alpha[:, 1] /= s
    ll = np.log(s)
    for l in range(1, e.shape[0]):
        ql = q[:, l - 1]
        a0 = (alpha[:, 0] * (1.0 - ql) + alpha[:, 1] * ql) * e[l, 0]
        a1 = (alpha[:, 0] * ql + alpha[:, 1] * (1.0 - ql)) * e[l, 1]
        s = a0 + a1
        alpha[:, 0] = a0 / s
        alpha[:, 1] = a1 / s
        ll = ll + np.log(s)
    return ll


def viterbi(e, lam, dist):
    m = e.shape[0]
    q = 1.0 - np.exp(-lam * dist)
    d0 = math.log(0.5) + math.log(e[0, 0])
    d1 = math.log(0.5) + math.log(e[0, 1])
    psi = np.zeros((m, 2), dtype=np.int64)
    for l in range(1, m):
        ql = q[l - 1]
        lp = math.log(1.0 - ql)
        lq = math.log(ql)
        if d0 + lp >= d1 + lq:
            b0 = d0 + lp
            psi[l, 0] = 0
        else:
            b0 = d1 + lq
            psi[l, 0] = 1
        if d0 + lq >= d1 + lp:
            b1 = d0 + lq
            psi[l, 1] = 0
        else:
            b1 = d1 + lp
            psi[l, 1] = 1
        d0 = b0 + math.log(e[l, 0])
        d1 = b1 + math.log(e[l, 1])
    path = np.empty(m, dtype=np.int8)
    s = 0 if d0 >= d1 else 1
    path[m - 1] = s
    for l in range(m - 2, -1, -1):
        s = psi[l + 1, s]
        path[l] = s
    return path


def n_switches(path):
    return int(np.sum(path[1:] != path[:-1]))


def load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def path_file(path, n):
    try:
        with open(path) as fh:
            s = fh.read().strip()
    except Exception:
        return None
    if len(s) != n or any(c not in "AB" for c in s):
        return None
    return np.asarray([0 if c == "A" else 1 for c in s], dtype=np.int8)


def posterior_file(path, snp_ids):
    try:
        with open(path) as fh:
            head = fh.readline().strip().split()
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    except Exception:
        return None
    n = len(snp_ids)
    if head != ["snp_id", "p_A", "p_B"] or len(lines) != n:
        return None
    arr = np.empty((n, 2), dtype=float)
    for i, ln in enumerate(lines):
        parts = ln.split()
        if len(parts) != 3 or parts[0] != snp_ids[i]:
            return None
        try:
            a = float(parts[1])
            b = float(parts[2])
        except ValueError:
            return None
        if not (math.isfinite(a) and math.isfinite(b)):
            return None
        arr[i, 0] = a
        arr[i, 1] = b
    if arr.min() < -1e-9 or arr.max() > 1.0 + 1e-9:
        return None
    if np.max(np.abs(arr.sum(axis=1) - 1.0)) > 1e-6:
        return None
    return np.clip(arr, 0.0, 1.0)


def main():
    sim = gold_gen.build(SEED)
    dist = np.diff(sim["cM"]) / 100.0
    snp_ids = sim["snp_id"]
    report = load_json(os.path.join(OUT, "report.json")) or {}
    entries = {}
    for e in (report.get("per_chromosome") or []):
        if isinstance(e, dict):
            entries[(str(e.get("fst_level")), str(e.get("sample")))] = e
    level_entries = {}
    for e in (report.get("per_fst_level") or []):
        if isinstance(e, dict):
            level_entries[str(e.get("fst_level"))] = e

    items = {}

    def add(name, ok):
        items.setdefault(name, []).append(bool(ok))

    ref = {}
    for lvl in LEVELS:
        L = sim["levels"][lvl]
        pA = (L["panel_a"].sum(axis=0) + PSEUDO) / (L["panel_a"].shape[0] + 1.0)
        pB = (L["panel_b"].sum(axis=0) + PSEUDO) / (L["panel_b"].shape[0] + 1.0)
        for si, sample in enumerate(SAMPLES):
            x = L["admixed"][si]
            e = emission(x, pA, pB)
            lls = loglik_grid(e, dist, GRID)
            lam_opt = int(GRID[int(np.argmax(lls))])
            ent = entries.get((lvl, sample), {})
            raw = ent.get("lambda_hat")
            lam_valid = (isinstance(raw, (int, float)) and not isinstance(raw, bool)
                         and float(raw).is_integer() and 1 <= int(raw) <= GRID[-1])
            lam_use = int(raw) if lam_valid else lam_opt
            post, ll_use = forward_backward(e, lam_use, dist)
            vit = viterbi(e, lam_use, dist)
            ppos = np.where(post[:, 0] >= post[:, 1], 0, 1).astype(np.int8)
            mle = np.where(e[:, 0] >= e[:, 1], 0, 1).astype(np.int8)
            truth = L["truth"][si]
            ref[(lvl, sample)] = {
                "lls": lls, "lam_valid": lam_valid, "lam_use": lam_use, "ll_use": ll_use,
                "post": post, "vit": vit, "ppos": ppos, "truth": truth, "ent": ent,
                "n_sw_post": n_switches(ppos), "n_sw_vit": n_switches(vit),
                "acc_post": float(np.mean(ppos == truth)),
                "acc_vit": float(np.mean(vit == truth)),
                "acc_mle": float(np.mean(mle == truth)),
            }

    for lvl in LEVELS:
        for sample in SAMPLES:
            r = ref[(lvl, sample)]
            ent = r["ent"]
            arr = posterior_file(os.path.join(OUT, "posteriors", lvl, sample + ".tsv"), snp_ids)
            vpath = path_file(os.path.join(OUT, "paths", lvl, sample + ".viterbi.txt"), len(snp_ids))
            qpath = path_file(os.path.join(OUT, "paths", lvl, sample + ".posterior.txt"), len(snp_ids))
            add("format", arr is not None and vpath is not None and qpath is not None)
            add("posterior", arr is not None and float(np.max(np.abs(arr - r["post"]))) <= TOL_POST)
            add("viterbi", vpath is not None and bool(np.array_equal(vpath, r["vit"])))
            add("posterior_path", qpath is not None and bool(np.array_equal(qpath, r["ppos"])))
            lam_ok = False
            if r["lam_valid"]:
                lam = int(ent.get("lambda_hat"))
                slack = float(np.max(r["lls"]) - r["lls"][lam - 1])
                rep_ll = ent.get("log_likelihood_at_lambda_hat")
                ll_ok = isinstance(rep_ll, (int, float)) and abs(float(rep_ll) - r["ll_use"]) <= TOL_LL
                lam_ok = (slack <= LL_SLACK) and ll_ok
            add("lambda_hat", lam_ok)
            add("switch_counts", ent.get("n_switches_posterior") == r["n_sw_post"]
                and ent.get("n_switches_viterbi") == r["n_sw_vit"])
            accs = (ent.get("accuracy_posterior"), ent.get("accuracy_viterbi"),
                    ent.get("accuracy_per_site_mle"))
            refs = (r["acc_post"], r["acc_vit"], r["acc_mle"])
            ok = True
            for a, b in zip(accs, refs):
                if not isinstance(a, (int, float)) or abs(float(a) - b) > TOL_ACC:
                    ok = False
            add("accuracy", ok)
            gate = ent.get("hmm_beats_per_site_mle")
            add("hmm_vs_mle", isinstance(gate, bool) and gate == (r["acc_vit"] > r["acc_mle"]))
            it = ent.get("implied_admixture_time_generations")
            add("diagnostics", r["lam_valid"] and isinstance(it, (int, float))
                and abs(float(it) - 2.0 * int(ent.get("lambda_hat"))) <= 1e-6)

    n_feasible = 0
    for lvl in LEVELS:
        rs = [ref[(lvl, s)] for s in SAMPLES]
        mean_lam = float(np.mean([r["lam_use"] for r in rs]))
        msp = float(np.mean([r["n_sw_post"] for r in rs]))
        msv = float(np.mean([r["n_sw_vit"] for r in rs]))
        infl = msp / max(msv, 1.0)
        macp = float(np.mean([r["acc_post"] for r in rs]))
        macv = float(np.mean([r["acc_vit"] for r in rs]))
        macm = float(np.mean([r["acc_mle"] for r in rs]))
        lr = level_entries.get(lvl, {})

        def close(a, b, rel=0.01, abs_=0.05):
            try:
                a = float(a)
            except (TypeError, ValueError):
                return False
            return abs(a - b) <= max(abs_, rel * abs(b))

        ok = (close(lr.get("mean_lambda_hat"), mean_lam, abs_=0.05)
              and close(lr.get("mean_switches_posterior"), msp, abs_=0.5)
              and close(lr.get("mean_switches_viterbi"), msv, abs_=0.5)
              and close(lr.get("tract_inflation_factor"), infl)
              and close(lr.get("mean_accuracy_posterior"), macp, abs_=TOL_ACC)
              and close(lr.get("mean_accuracy_viterbi"), macv, abs_=TOL_ACC)
              and close(lr.get("mean_accuracy_per_site_mle"), macm, abs_=TOL_ACC))
        add("aggregates", ok)
        feas = bool(macv >= FEASIBLE_ACC)
        n_feasible += int(feas)
        add("diagnostics", lr.get("feasible") is feas)
    summary = report.get("summary") or {}
    add("diagnostics", summary.get("n_feasible_fst_levels") == n_feasible)

    breakdown = {}
    total = 0.0
    for name, w in WEIGHTS.items():
        oks = items.get(name, [])
        frac = float(np.mean(oks)) if oks else 0.0
        breakdown[name] = {"weight": w, "n_items": len(oks),
                           "passed": int(sum(oks)), "fraction": round(frac, 4)}
        total += w * frac
    total = float(max(0.0, min(1.0, total)))

    os.makedirs(os.path.dirname(REWARD_PATH), exist_ok=True)
    with open(REWARD_PATH, "w") as fh:
        json.dump({"score": round(total, 6), "reward": round(total, 6)}, fh)
    with open(REPORT_PATH, "w") as fh:
        json.dump({"score": round(total, 6), "constraints": breakdown,
                   "feasible_levels": n_feasible}, fh, indent=2)
    print(json.dumps({"score": round(total, 6), "constraints": breakdown}, indent=2))


if __name__ == "__main__":
    main()
