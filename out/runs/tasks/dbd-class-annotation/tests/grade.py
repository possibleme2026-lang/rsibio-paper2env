#!/usr/bin/env python3
# Deep grader for the DBD-class annotation task.
import json
import math
import os
import sys

sys.path.insert(0, "/tests")
import gold_gen as G

OUT_DIR = "/app/output"
DATA_DIR = "/app/data"
REWARD_DIR = "/logs/verifier"
TAB = chr(9)
HEADER = ["seq_id", "predicted_class", "score", "p_value", "strand", "start", "end", "ambiguous"]
ANN_HEADER = ["class_name", "n_sites", "total_bp", "bp_fraction", "mean_score"]
WEIGHTS = {"format": 0.10, "fields": 0.05, "score": 0.10, "pvalue": 0.10, "f1": 0.30,
           "none_recall": 0.07, "coverage": 0.08, "annotation": 0.12, "ambiguous": 0.08}
CLASS_SET = set(G.CLASSES)


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def ramp(x, lo, hi):
    return clamp((x - lo) / float(hi - lo))


def to_float(tok):
    try:
        return float(tok)
    except Exception:
        return None


def read_fasta(path):
    seqs = []
    sid = None
    buf = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sid is not None:
                    seqs.append((sid, "".join(buf)))
                sid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if sid is not None:
        seqs.append((sid, "".join(buf)))
    return seqs


def verify_row(row, seq, quant):
    """Return (score_ok, pvalue_ok) for one submitted row."""
    cls = row[1]
    strand = row[4]
    try:
        start = int(row[5])
        end = int(row[6])
        rep_score = float(row[2])
        rep_p = float(row[3])
    except Exception:
        return False, False
    if start < 0 or end > len(seq) or start >= end:
        return False, False
    word = seq[start:end] if strand == "+" else G.revcomp(seq[start:end])
    if cls == "None":
        cands = [c for c in G.CLASSES if len(quant[c]["q"]) == (end - start)]
    else:
        cands = [cls] if cls in quant else []
    score_ok = False
    p_ok = False
    for c in cands:
        ctx = quant[c]
        si = G.score_int(ctx, word)
        if abs(si / 100.0 - rep_score) <= 0.011:
            score_ok = True
            pr = G.pvalue_from_int(ctx, si)
            if rep_p > 0.0 and pr > 0.0 and abs(math.log10(pr) - math.log10(rep_p)) <= 0.05:
                p_ok = True
    return score_ok, p_ok


def main():
    scores = {}
    detail = {}
    gold = G.build()
    models = G.round_models(gold["models"])
    bg = G.round_bg(gold["bg"])
    quant = G.build_quantizers(models, bg)
    test = gold["test"]
    seq_ids = [sid for sid, _ in test]
    seq_map = dict(test)
    truth = dict(zip(seq_ids, gold["test_labels"]))
    ref_rows, n_tests, p_cut = G.reference_pipeline(test, quant, G.ALPHA, G.SEQ_LEN)
    ref = dict((r["seq_id"], r) for r in ref_rows)
    ref_ann = G.annotation_table(ref_rows, list(G.CLASSES), len(test), G.SEQ_LEN)
    detail["n_tests"] = n_tests
    detail["p_cut"] = p_cut

    # reference self-check (diagnostics only)
    ref_pred = dict((r["seq_id"], r["predicted_class"]) for r in ref_rows)
    lab = list(G.CLASSES) + ["None"]
    tp = dict((l, 0) for l in lab)
    fp = dict((l, 0) for l in lab)
    fn = dict((l, 0) for l in lab)
    for sid in seq_ids:
        t = truth[sid]
        p = ref_pred[sid]
        if t == p:
            tp[t] += 1
        else:
            fn[t] += 1
            fp[p] += 1
    f1s = []
    for l in lab:
        pr = tp[l] / float(tp[l] + fp[l]) if (tp[l] + fp[l]) else 0.0
        rc = tp[l] / float(tp[l] + fn[l]) if (tp[l] + fn[l]) else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0)
    detail["reference_macro_f1"] = sum(f1s) / len(f1s)

    integrity = False
    try:
        on_disk = dict(read_fasta(os.path.join(DATA_DIR, "test_sequences.fa")))
        integrity = (on_disk == seq_map)
    except Exception:
        integrity = False
    detail["input_integrity_ok"] = bool(integrity)

    header_ok = False
    ids_ok = False
    fields_ok = False
    rows = []
    try:
        with open(os.path.join(OUT_DIR, "predictions.tsv")) as fh:
            lines = [ln.rstrip(chr(10)) for ln in fh if ln.strip()]
        if lines:
            header_ok = (lines[0].split(TAB) == HEADER)
            rows = [ln.split(TAB) for ln in lines[1:]]
    except Exception:
        rows = []
    ids_ok = (len(rows) == len(seq_ids)) and all(len(r) > 0 and r[0] == sid
                                                  for r, sid in zip(rows, seq_ids))
    fields_ok = bool(rows) and all(
        len(r) == 8 and to_float(r[2]) is not None and to_float(r[3]) is not None
        and r[4] in ("+", "-") and r[5].lstrip("-").isdigit() and r[6].lstrip("-").isdigit()
        and r[7] in ("0", "1") for r in rows)
    scores["format"] = 0.4 * float(header_ok) + 0.4 * float(ids_ok) + 0.2 * float(fields_ok)

    pred = {}
    if ids_ok and all(len(r) == 8 for r in rows):
        pred = dict((r[0], r) for r in rows)
    detail["n_rows"] = len(rows)

    good_fields = 0
    for r in rows:
        if len(r) != 8:
            continue
        cls = r[1]
        if cls not in CLASS_SET and cls != "None":
            continue
        if r[4] not in ("+", "-") or r[7] not in ("0", "1"):
            continue
        try:
            s = int(r[5])
            e = int(r[6])
        except Exception:
            continue
        if not (0 <= s < e <= G.SEQ_LEN):
            continue
        if cls != "None" and (e - s) != len(models[cls]):
            continue
        good_fields += 1
    scores["fields"] = good_fields / float(len(seq_ids))

    sample_ids = seq_ids[::17]
    n_score_ok = 0
    n_p_ok = 0
    n_sampled = 0
    for sid in sample_ids:
        r = pred.get(sid)
        if r is None:
            continue
        n_sampled += 1
        s_ok, p_ok = verify_row(r, seq_map[sid], quant)
        n_score_ok += int(s_ok)
        n_p_ok += int(p_ok)
    detail["sampled_rows"] = n_sampled
    detail["score_rows_ok"] = n_score_ok
    detail["pvalue_rows_ok"] = n_p_ok
    scores["score"] = ramp(n_score_ok / float(max(1, n_sampled)), 0.5, 0.95)
    scores["pvalue"] = ramp(n_p_ok / float(max(1, n_sampled)), 0.5, 0.95)

    tp2 = dict((l, 0) for l in lab)
    fp2 = dict((l, 0) for l in lab)
    fn2 = dict((l, 0) for l in lab)
    for sid in seq_ids:
        t = truth[sid]
        r = pred.get(sid)
        p = r[1] if r is not None and len(r) == 8 else "<missing>"
        if t == p:
            tp2[t] += 1
        else:
            fn2[t] += 1
            if p in fp2:
                fp2[p] += 1
    f1s2 = []
    for l in lab:
        pr = tp2[l] / float(tp2[l] + fp2[l]) if (tp2[l] + fp2[l]) else 0.0
        rc = tp2[l] / float(tp2[l] + fn2[l]) if (tp2[l] + fn2[l]) else 0.0
        f1s2.append(2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0)
    macro = sum(f1s2) / len(f1s2)
    detail["agent_macro_f1"] = macro
    scores["f1"] = ramp(macro, 0.20, 0.55)

    none_true = [sid for sid in seq_ids if truth[sid] == "None"]
    none_hit = 0
    for sid in none_true:
        r = pred.get(sid)
        if r is not None and len(r) == 8 and r[1] == "None":
            none_hit += 1
    none_recall = none_hit / float(max(1, len(none_true)))
    detail["none_recall"] = none_recall
    scores["none_recall"] = ramp(none_recall, 0.30, 0.70)

    n_good_classes = 0
    per_class_recall = {}
    for c in G.CLASSES:
        tot = sum(1 for sid in seq_ids if truth[sid] == c)
        hit = 0
        for sid in seq_ids:
            if truth[sid] != c:
                continue
            r = pred.get(sid)
            if r is not None and len(r) == 8 and r[1] == c:
                hit += 1
        rec = hit / float(tot) if tot else 0.0
        per_class_recall[c] = rec
        if rec >= 0.5:
            n_good_classes += 1
    detail["per_class_recall"] = per_class_recall
    detail["classes_with_recall_ge_0.5"] = n_good_classes
    scores["coverage"] = clamp(n_good_classes / 20.0)

    ann_rows = {}
    ann_header_ok = False
    try:
        with open(os.path.join(OUT_DIR, "annotation.tsv")) as fh:
            lines = [ln.rstrip(chr(10)) for ln in fh if ln.strip()]
        if lines:
            ann_header_ok = (lines[0].split(TAB) == ANN_HEADER)
            for ln in lines[1:]:
                f = ln.split(TAB)
                if len(f) == 5:
                    ann_rows[f[0]] = f
    except Exception:
        ann_rows = {}
    good_ann = 0
    for c in G.CLASSES:
        r = ann_rows.get(c)
        if r is None:
            continue
        try:
            n = int(r[1])
            bp = int(r[2])
            bf = float(r[3])
            ms = float(r[4])
        except Exception:
            continue
        refa = ref_ann[c]
        if n != refa["n_sites"] or bp != refa["total_bp"]:
            continue
        if abs(bf - refa["bp_fraction"]) > 1e-6:
            continue
        if abs(ms - refa["mean_score"]) > 0.05:
            continue
        good_ann += 1
    detail["annotation_header_ok"] = ann_header_ok
    detail["annotation_classes_ok"] = good_ann
    scores["annotation"] = good_ann / float(len(G.CLASSES))

    amb_ids = seq_ids[::13][:40]
    amb_tot = 0
    amb_ok = 0
    for sid in amb_ids:
        r = pred.get(sid)
        if r is None or len(r) != 8:
            continue
        amb_tot += 1
        if r[7] == str(ref[sid]["ambiguous"]):
            amb_ok += 1
    agree = amb_ok / float(max(1, amb_tot))
    detail["ambiguous_agreement"] = agree
    scores["ambiguous"] = ramp(agree, 0.60, 0.90)

    total = 0.0
    for k in WEIGHTS:
        total += WEIGHTS[k] * scores.get(k, 0.0)
    total = round(clamp(total), 4)

    report = {"checks": scores, "weights": WEIGHTS, "detail": detail, "score": total}
    try:
        os.makedirs(REWARD_DIR, exist_ok=True)
        with open(os.path.join(REWARD_DIR, "reward.json"), "w") as fh:
            json.dump({"score": float(total), "reward": float(total)}, fh)
        with open(os.path.join(REWARD_DIR, "report.json"), "w") as fh:
            json.dump(report, fh, indent=2)
    except Exception:
        pass
    print("=== DBD-class annotation grading ===")
    for k in ["format", "fields", "score", "pvalue", "f1", "none_recall", "coverage",
              "annotation", "ambiguous"]:
        print("%-13s weight=%.2f credit=%.3f" % (k, WEIGHTS[k], scores.get(k, 0.0)))
    print("reference_macro_f1=%.3f agent_macro_f1=%.3f" % (detail["reference_macro_f1"], macro))
    print("none_recall=%.3f classes_ge_0.5=%d ann_ok=%d amb_agree=%.3f integrity=%s"
          % (none_recall, n_good_classes, good_ann, agree, integrity))
    print("TOTAL SCORE = %.4f" % total)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        os.makedirs(REWARD_DIR, exist_ok=True)
        with open(os.path.join(REWARD_DIR, "reward.json"), "w") as fh:
            json.dump({"score": 0.0, "reward": 0.0}, fh)
        print("grader failure: %s" % exc)
        raise
