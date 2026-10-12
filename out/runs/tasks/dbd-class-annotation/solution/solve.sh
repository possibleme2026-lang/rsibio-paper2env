#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python3 - <<'PY'
import math, bisect

DATA = "/app/data"
OUT = "/app/output"
BASES = ("A", "C", "G", "T")
IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}
TAB = chr(9)
NL = chr(10)
ALPHA = 0.05


def read_fasta(path):
    out, sid, buf = [], None, []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if sid is not None:
                out.append((sid, "".join(buf)))
            sid, buf = line[1:].split()[0], []
        else:
            buf.append(line)
    if sid is not None:
        out.append((sid, "".join(buf)))
    return out


def read_models(path):
    models, order = {}, []
    for line in open(path).read().splitlines()[1:]:
        f = line.split(TAB)
        if len(f) != 6:
            continue
        if f[0] not in models:
            models[f[0]] = []
            order.append(f[0])
        models[f[0]].append(dict((b, float(f[2 + i])) for i, b in enumerate(BASES)))
    return models, order


def read_bg(path):
    bg = {}
    for line in open(path).read().splitlines()[1:]:
        f = line.split(TAB)
        if len(f) == 2:
            bg[f[0]] = float(f[1])
    return bg


def quantize(x):
    return int(math.floor(x * 100.0 + 0.5))


def build(ctx_models, bg):
    out = {}
    for name, rows in ctx_models.items():
        w = len(rows)
        q = [dict((b, quantize(math.log2(row[b] / bg[b]))) for b in BASES) for row in rows]
        counts = {0: 1.0}
        for j in range(w):
            nxt = {}
            for s, c in counts.items():
                for b in BASES:
                    ns = s + q[j][b]
                    nxt[ns] = nxt.get(ns, 0.0) + c
            counts = nxt
        ss = sorted(counts)
        cum = [0.0] * len(ss)
        acc = 0.0
        for i in range(len(ss) - 1, -1, -1):
            acc += counts[ss[i]]
            cum[i] = acc
        out[name] = {"w": w, "q": q, "qa": [[q[j][b] for b in BASES] for j in range(w)],
                     "ss": ss, "cum": cum, "total": float(4 ** w)}
    return out


def revcomp(s):
    return "".join(COMP[c] for c in reversed(s))


def best_site(seq, ctx):
    w, qa = ctx["w"], ctx["qa"]
    best = None
    for strand, s in (("+", seq), ("-", revcomp(seq))):
        idx = [IDX[c] for c in s]
        for i in range(len(s) - w + 1):
            tot = 0
            for j in range(w):
                tot += qa[j][idx[i + j]]
            if best is None or tot > best[0]:
                best = (tot, i, strand)
    return best


def pval(ctx, s_int):
    i = bisect.bisect_left(ctx["ss"], s_int)
    if i >= len(ctx["ss"]):
        return 0.0
    return ctx["cum"][i] / ctx["total"]


def main():
    models, order = read_models(DATA + "/dbd_class_models.tsv")
    bg = read_bg(DATA + "/background.tsv")
    quant = build(models, bg)
    seqs = read_fasta(DATA + "/test_sequences.fa")
    seq_len = len(seqs[0][1])
    n_tests = 2 * sum(seq_len - quant[c]["w"] + 1 for c in quant)
    p_cut = ALPHA / n_tests
    rows = []
    ann = dict((c, {"n": 0, "bp": 0, "scores": []}) for c in order)
    for sid, seq in seqs:
        per = {}
        for name, ctx in quant.items():
            s_int, pos, strand = best_site(seq, ctx)
            per[name] = (s_int, pos, strand, pval(ctx, s_int))
        ranked = sorted(per.items(), key=lambda kv: (kv[1][3], -kv[1][0], kv[0]))
        passing = [kv for kv in ranked if kv[1][3] <= p_cut]
        if passing:
            win = passing[0][0]
            info = per[win]
            amb = 1 if (len(passing) > 1 and passing[1][1][3] <= 5.0 * passing[0][1][3]) else 0
            cls = win
        else:
            info = ranked[0][1]
            cls, amb = "None", 0
        s_int, pos, strand, p = info
        if cls != "None":
            start, end = pos, pos + quant[cls]["w"]
        else:
            start, end = pos, pos + len(models[ranked[0][0]])
        rows.append([sid, cls, "%.2f" % (s_int / 100.0), "%.6e" % p, strand, str(start), str(end), str(amb)])
        if cls != "None":
            ann[cls]["n"] += 1
            ann[cls]["bp"] += end - start
            ann[cls]["scores"].append(s_int / 100.0)
    with open(OUT + "/predictions.tsv", "w") as fh:
        fh.write(TAB.join(["seq_id", "predicted_class", "score", "p_value", "strand", "start", "end", "ambiguous"]) + NL)
        for r in rows:
            fh.write(TAB.join(r) + NL)
    total_bp = float(len(seqs) * seq_len)
    with open(OUT + "/annotation.tsv", "w") as fh:
        fh.write(TAB.join(["class_name", "n_sites", "total_bp", "bp_fraction", "mean_score"]) + NL)
        for c in order:
            a = ann[c]
            ms = (sum(a["scores"]) / len(a["scores"])) if a["scores"] else 0.0
            fh.write(TAB.join([c, str(a["n"]), str(a["bp"]), "%.8f" % (a["bp"] / total_bp), "%.2f" % ms]) + NL)


main()
PY
