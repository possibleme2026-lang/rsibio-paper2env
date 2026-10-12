#!/usr/bin/env python3
# Protected generator + reference implementation for the DBD-class annotation task.
# Mirrors environment/data/gen_inputs.py exactly; lives only in the verifier phase.
import bisect
import math
import random

BASES = ("A", "C", "G", "T")
CLASSES = ("bZIP", "bHLH", "Homeodomain", "POU", "Paired", "Forkhead", "T-box", "ETS",
           "Rel", "MADS", "HMG", "C2H2-ZF", "Nuclear-Receptor", "GATA-ZF", "DM-ZF",
           "Tryptophan-Cluster", "Cold-Shock", "Runt", "SAND", "Winged-Helix",
           "HTH-LysR", "STAT", "IRF")
BG = {"A": 0.295, "C": 0.205, "G": 0.205, "T": 0.295}
SEED_MODELS = 20241220
SEED_TEST = 771103
SEED_CAL = 550219
N_TEST = 520
N_CAL = 200
SEQ_LEN = 120
ALPHA = 0.05
TAB = chr(9)
NL = chr(10)
IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


def draw_base(rng, probs):
    r = rng.random()
    acc = 0.0
    for b in BASES:
        acc += probs[b]
        if r < acc:
            return b
    return BASES[-1]


def make_models(seed):
    rng = random.Random(seed)
    models = {}
    for name in CLASSES:
        width = 15 + rng.randrange(5)
        rows = []
        for _ in range(width):
            lam = 0.89 + 0.07 * rng.random()
            cons = BASES[rng.randrange(4)]
            probs = {}
            for b in BASES:
                probs[b] = (1.0 - lam) * 0.25 + (lam if b == cons else 0.0)
            probs = dict((b, probs[b] * (0.97 + 0.06 * rng.random())) for b in BASES)
            tot = sum(probs.values())
            rows.append(dict((b, probs[b] / tot) for b in BASES))
        models[name] = rows
    return models


def gc_background(gc):
    p = dict(BG)
    p["C"] = BG["C"] * gc / (BG["C"] + BG["G"])
    p["G"] = BG["G"] * gc / (BG["C"] + BG["G"])
    at = 1.0 - gc
    p["A"] = BG["A"] * at / (BG["A"] + BG["T"])
    p["T"] = BG["T"] * at / (BG["A"] + BG["T"])
    return p


def paint(seq, models, name, offset, rho, rng):
    for k, row in enumerate(models[name]):
        b = draw_base(rng, row)
        if rng.random() < rho:
            i = BASES.index(b)
            b = BASES[(i + 1 + rng.randrange(3)) % 4]
        seq[offset + k] = b


def generate_sequences(seed, n_seq, models, length):
    rng = random.Random(seed)
    n_motif = int(round(n_seq * 0.75))
    has = [True] * n_motif + [False] * (n_seq - n_motif)
    rng.shuffle(has)
    prim = []
    while len(prim) < n_motif:
        pool = list(CLASSES)
        rng.shuffle(pool)
        prim.extend(pool)
    prim = prim[:n_motif]
    rng.shuffle(prim)
    out = []
    labels = []
    k = 0
    for i, flag in enumerate(has):
        gc = 0.41 + rng.uniform(-0.03, 0.03)
        probs = gc_background(gc)
        seq = [draw_base(rng, probs) for _ in range(length)]
        if rng.random() < 0.20:
            unit = "".join(BASES[rng.randrange(4)] for _ in range(1 + rng.randrange(3)))
            patch = (unit * (6 + rng.randrange(10)))[:length]
            st = rng.randrange(0, length - len(patch) + 1)
            for j, ch in enumerate(patch):
                seq[st + j] = ch
        if flag:
            name = prim[k]
            k += 1
            w = len(models[name])
            st = rng.randrange(0, length - w + 1)
            paint(seq, models, name, st, rng.uniform(0.0, 0.03), rng)
            for _ in range((0, 0, 1, 1, 2)[rng.randrange(5)]):
                dname = CLASSES[rng.randrange(len(CLASSES))]
                if dname == name:
                    continue
                dw = len(models[dname])
                dst = None
                for _try in range(25):
                    cand = rng.randrange(0, length - dw + 1)
                    if cand + dw <= st or st + w <= cand:
                        dst = cand
                        break
                if dst is not None:
                    paint(seq, models, dname, dst, rng.uniform(0.05, 0.20), rng)
            labels.append(name)
        else:
            labels.append("None")
        out.append(("seq_%06d" % (i + 1), "".join(seq)))
    return out, labels


def build():
    models = make_models(SEED_MODELS)
    test, test_labels = generate_sequences(SEED_TEST, N_TEST, models, SEQ_LEN)
    cal, cal_labels = generate_sequences(SEED_CAL, N_CAL, models, SEQ_LEN)
    return {"models": models, "bg": dict(BG), "test": test, "test_labels": test_labels,
            "cal": cal, "cal_labels": cal_labels}


def round_models(models):
    out = {}
    for name, rows in models.items():
        out[name] = [dict((b, float("%.6f" % row[b])) for b in BASES) for row in rows]
    return out


def round_bg(bg):
    return dict((b, float("%.6f" % bg[b])) for b in BASES)


def revcomp(s):
    return "".join(COMP[ch] for ch in reversed(s))


def quantize(x):
    return int(math.floor(x * 100.0 + 0.5))


def build_quantizers(models, bg):
    out = {}
    for name, rows in models.items():
        w = len(rows)
        q = []
        for row in rows:
            q.append(dict((b, quantize(math.log2(row[b] / bg[b]))) for b in BASES))
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
        out[name] = {"w": w, "q": q, "ss": ss, "cum": cum, "total": float(4 ** w)}
    return out


def score_int(ctx, word):
    s = 0
    q = ctx["q"]
    for j, ch in enumerate(word):
        s += q[j][ch]
    return s


def pvalue_from_int(ctx, s):
    i = bisect.bisect_left(ctx["ss"], s)
    if i >= len(ctx["ss"]):
        return 0.0
    return ctx["cum"][i] / ctx["total"]


def pvalue_word(ctx, word):
    return pvalue_from_int(ctx, score_int(ctx, word))


def best_site(seq, ctx):
    w = ctx["w"]
    qa = [[ctx["q"][j][b] for b in BASES] for j in range(w)]
    best = None
    for strand, s in (("+", seq), ("-", revcomp(seq))):
        sidx = [IDX[ch] for ch in s]
        for i in range(len(s) - w + 1):
            tot = 0
            for j in range(w):
                tot += qa[j][sidx[i + j]]
            if best is None or tot > best[0]:
                best = (tot, i, strand)
    return best


def reference_pipeline(seqs, quant, alpha=ALPHA, seq_len=SEQ_LEN):
    n_tests = 2 * sum(seq_len - quant[c]["w"] + 1 for c in quant)
    p_cut = alpha / n_tests
    out = []
    for sid, seq in seqs:
        per = {}
        for name, ctx in quant.items():
            s_int, pos, strand = best_site(seq, ctx)
            per[name] = {"score": s_int / 100.0, "start": pos, "end": pos + ctx["w"],
                         "strand": strand, "p": pvalue_from_int(ctx, s_int)}
        ranked = sorted(per.items(), key=lambda kv: (kv[1]["p"], -kv[1]["score"], kv[0]))
        passing = [(n, v) for n, v in ranked if v["p"] <= p_cut]
        if passing:
            winner = passing[0][0]
            amb = 1 if (len(passing) > 1 and passing[1][1]["p"] <= 5.0 * passing[0][1]["p"]) else 0
            info = per[winner]
        else:
            winner = "None"
            amb = 0
            info = ranked[0][1]
        out.append({"seq_id": sid, "predicted_class": winner, "score": info["score"],
                    "p_value": info["p"], "strand": info["strand"], "start": info["start"],
                    "end": info["end"], "ambiguous": amb})
    return out, n_tests, p_cut


def annotation_table(rows, classes, n_seq, seq_len):
    acc = dict((c, {"n": 0, "bp": 0, "scores": []}) for c in classes)
    for r in rows:
        c = r["predicted_class"]
        if c == "None" or c not in acc:
            continue
        acc[c]["n"] += 1
        acc[c]["bp"] += r["end"] - r["start"]
        acc[c]["scores"].append(r["score"])
    out = {}
    for c in classes:
        a = acc[c]
        out[c] = {"n_sites": a["n"], "total_bp": a["bp"],
                  "bp_fraction": a["bp"] / float(n_seq * seq_len),
                  "mean_score": (sum(a["scores"]) / len(a["scores"])) if a["scores"] else 0.0}
    return out
