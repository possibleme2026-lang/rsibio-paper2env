#!/usr/bin/env python3
"""Grader: compares /app/output against the protected reference implementation."""
from __future__ import annotations

import json
import os
import re
import sys

GOLD_PATH = "/tmp/gold.json"
RES_PATH = "/app/output/results.json"
TREE_DIR = "/app/output/trees"
REWARD_PATH = "/logs/verifier/reward.json"
REPORT_PATH = "/logs/verifier/report.json"

CHECKS = []


def add(name, ok, detail=""):
    CHECKS.append({"name": name, "passed": bool(ok), "detail": str(detail)})


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def close(a, b, tol):
    a, b = num(a), num(b)
    return a is not None and b is not None and abs(a - b) <= tol


def finish():
    n = len(CHECKS)
    passed = sum(1 for c in CHECKS if c["passed"])
    score = round(passed / n, 4) if n else 0.0
    os.makedirs("/logs/verifier", exist_ok=True)
    with open(REWARD_PATH, "w") as fh:
        json.dump({"score": score, "reward": score}, fh)
    with open(REPORT_PATH, "w") as fh:
        json.dump({"score": score, "n_checks": n, "n_passed": passed, "checks": CHECKS},
                  fh, indent=2)
    print(json.dumps({"score": score, "n_passed": passed, "n_checks": n}, indent=2))
    for c in CHECKS:
        print(("PASS " if c["passed"] else "FAIL ") + c["name"] + " :: " + c["detail"])


# ------------------------- newick handling -------------------------

def parse_newick(text):
    s = re.sub(r"\s+", "", text).rstrip(";")
    pos = [0]

    def parse_len():
        if pos[0] < len(s) and s[pos[0]] == ":":
            pos[0] += 1
            buf = ""
            while pos[0] < len(s) and s[pos[0]] not in ",()":
                buf += s[pos[0]]
                pos[0] += 1
            return float(buf)
        return 0.0

    def node():
        if pos[0] >= len(s):
            raise ValueError("truncated newick")
        if s[pos[0]] == "(":
            pos[0] += 1
            kids = [node()]
            while pos[0] < len(s) and s[pos[0]] == ",":
                pos[0] += 1
                kids.append(node())
            if pos[0] >= len(s) or s[pos[0]] != ")":
                raise ValueError("malformed newick")
            pos[0] += 1
            label = ""
            while pos[0] < len(s) and s[pos[0]] not in ",():;":
                label += s[pos[0]]
                pos[0] += 1
            return {"kids": kids, "label": label, "length": parse_len()}
        label = ""
        while pos[0] < len(s) and s[pos[0]] not in ",():;":
            label += s[pos[0]]
            pos[0] += 1
        return {"kids": [], "label": label, "length": parse_len()}

    return node()


def tree_maps(text):
    root = parse_newick(text)
    out = {}

    def rec(n):
        if not n["kids"]:
            return frozenset([n["label"]]), 0.0
        leaves = set()
        dists = []
        for k in n["kids"]:
            sub, d = rec(k)
            leaves |= set(sub)
            dists.append(d + k["length"])
        h = sum(dists) / len(dists)
        out[frozenset(leaves)] = h
        return frozenset(leaves), h

    rec(root)
    return out


# ------------------------------ main ------------------------------

PLACEHOLDERS = ["clock_mae", "clock_pairs", "median_delta", "significance_call",
                "wilcoxon_p", "event_probe_counts", "trunk_private_counts",
                "private_fraction", "cophenetic_correlation", "tree_topology",
                "tree_branch_lengths", "x_enrichment"]


def main():
    try:
        with open(GOLD_PATH) as fh:
            gold = json.load(fh)
    except Exception as exc:
        print("FATAL: gold not available:", exc)
        for name in PLACEHOLDERS:
            add(name, False, "gold unavailable")
        finish()
        return

    res = None
    try:
        with open(RES_PATH) as fh:
            res = json.load(fh)
    except Exception as exc:
        for name in PLACEHOLDERS:
            add(name, False, "cannot load %s (%s)" % (RES_PATH, exc))
        finish()
        return

    gclock = gold["clock"]
    rclock = res.get("clock", {}) if isinstance(res, dict) else {}
    add("clock_mae", close(rclock.get("normal_mae_years"), gclock["normal_mae_years"], 0.5),
        "agent=%s gold=%.4f" % (rclock.get("normal_mae_years"), gclock["normal_mae_years"]))

    gct = gclock["per_cancer_type"]
    rct = rclock.get("per_cancer_type", {}) or {}

    pairs_ok, pairs_det = True, []
    for ct in gct:
        got = (rct.get(ct) or {}).get("n_pairs")
        if num(got) != gct[ct]["n_pairs"]:
            pairs_ok = False
            pairs_det.append("%s:%s!=%s" % (ct, got, gct[ct]["n_pairs"]))
    add("clock_pairs", pairs_ok, "; ".join(pairs_det) or "n_pairs correct for all 8 cancer types")

    med_ok, med_det = True, []
    for ct in gct:
        got = (rct.get(ct) or {}).get("median_delta_years")
        if not close(got, gct[ct]["median_delta_years"], 0.5):
            med_ok = False
            med_det.append("%s:%s vs %.3f" % (ct, got, gct[ct]["median_delta_years"]))
    add("median_delta", med_ok, "; ".join(med_det) or "median tumour-normal deltas within 0.5 y")

    call_ok, call_det = True, []
    for ct in gct:
        got = (rct.get(ct) or {}).get("call")
        if got != gct[ct]["call"]:
            call_ok = False
            call_det.append("%s:%s!=%s" % (ct, got, gct[ct]["call"]))
    add("significance_call", call_ok, "; ".join(call_det) or "all acceleration calls correct")

    p_ok, p_det = True, []
    for ct in gct:
        a = num((rct.get(ct) or {}).get("wilcoxon_p"))
        g = gct[ct]["wilcoxon_p"]
        if a is None:
            p_ok = False
            p_det.append("%s:missing" % ct)
        elif not (abs(a - g) <= 0.05 or (a < 1e-6 and g < 1e-6)):
            p_ok = False
            p_det.append("%s:%.3g vs %.3g" % (ct, a, g))
    add("wilcoxon_p", p_ok, "; ".join(p_det) or "Wilcoxon p-values agree")

    gph = gold["phylo"]["per_cancer_type"]
    rph = (res.get("phylo", {}) or {}).get("per_cancer_type", {}) or {}

    ev_ok, ev_det = True, []
    for ct in gph:
        g = gph[ct]["n_event_probes"]
        if not close((rph.get(ct) or {}).get("n_event_probes"), g, max(3.0, 0.02 * g)):
            ev_ok = False
            ev_det.append("%s:%s vs %s" % (ct, (rph.get(ct) or {}).get("n_event_probes"), g))
    add("event_probe_counts", ev_ok, "; ".join(ev_det) or "event-probe counts agree")

    tp_ok, tp_det = True, []
    for ct in gph:
        r = rph.get(ct) or {}
        gt, gp = gph[ct]["n_trunk_probes"], gph[ct]["n_private_probes"]
        if num(r.get("n_trunk_probes")) != gt or not close(r.get("n_private_probes"), gp,
                                                            max(3.0, 0.02 * gp)):
            tp_ok = False
            tp_det.append("%s trunk %s/%s private %s/%s" % (ct, r.get("n_trunk_probes"), gt,
                                                            r.get("n_private_probes"), gp))
    add("trunk_private_counts", tp_ok, "; ".join(tp_det) or "trunk/private counts agree")

    pf_ok, pf_det = True, []
    for ct in gph:
        if not close((rph.get(ct) or {}).get("private_fraction"), gph[ct]["private_fraction"], 0.01):
            pf_ok = False
            pf_det.append(ct)
    add("private_fraction", pf_ok, ",".join(pf_det) or "private fractions agree")

    cp_ok, cp_det = True, []
    for ct in gph:
        if not close((rph.get(ct) or {}).get("cophenetic_correlation"),
                     gph[ct]["cophenetic_correlation"], 0.01):
            cp_ok = False
            cp_det.append("%s:%s vs %.4f" % (ct, (rph.get(ct) or {}).get("cophenetic_correlation"),
                                              gph[ct]["cophenetic_correlation"]))
    add("cophenetic_correlation", cp_ok, "; ".join(cp_det) or "cophenetic correlations agree")

    topo_ok, bl_ok, t_det, b_det = True, True, [], []
    for ct in gold["trees"]:
        path = os.path.join(TREE_DIR, ct + ".nwk")
        try:
            with open(path) as fh:
                maps = tree_maps(fh.read())
        except Exception as exc:
            topo_ok = bl_ok = False
            t_det.append("%s:%s" % (ct, exc))
            continue
        gtree = {frozenset(k): v for k, v in gold["trees"][ct]}
        if set(maps.keys()) != set(gtree.keys()):
            topo_ok = False
            t_det.append("%s: clade sets differ" % ct)
        else:
            for k, v in maps.items():
                if abs(v - gtree[k]) > 1e-4:
                    bl_ok = False
                    b_det.append("%s: edge lengths differ" % ct)
                    break
    add("tree_topology", topo_ok, "; ".join(t_det) or "all 8 Newick topologies match")
    add("tree_branch_lengths", bl_ok, "; ".join(b_det) or "all 8 Newick branch lengths match")

    gx = gold["phylo"]["x_enrichment"]
    rx = (res.get("phylo", {}) or {}).get("x_enrichment", {}) or {}
    px = num(rx.get("hypergeom_p"))
    x_ok = (close(rx.get("event_chrx_fraction"), gx["event_chrx_fraction"], 0.01)
            and close(rx.get("background_chrx_fraction"), gx["background_chrx_fraction"], 0.01)
            and px is not None and px < 0.01)
    add("x_enrichment", x_ok, "event_chrx=%s bg_chrx=%s p=%s (gold p=%.3g)"
        % (rx.get("event_chrx_fraction"), rx.get("background_chrx_fraction"), px, gx["hypergeom_p"]))

    finish()


if __name__ == "__main__":
    main()
