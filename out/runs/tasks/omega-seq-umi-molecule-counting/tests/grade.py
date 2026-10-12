#!/usr/bin/env python3
"""Grader for bioforge/omega-seq-umi-molecule-counting.

Recomputes the reference answer directly from the (agent-visible) inputs with an
independent implementation of the documented pipeline, then awards graded
partial credit across seven independently checkable constraints.
"""
import json
import math
import os
import re
from collections import Counter, defaultdict

DATA = "/app/data"
OUT = "/app/output"
LOG = "/logs/verifier"

WEIGHTS = {
    "schema_counts": 0.10,
    "total_molecules": 0.15,
    "row_agreement": 0.25,
    "nc_background": 0.10,
    "end_profile": 0.15,
    "ercc_correlation": 0.10,
    "qc_metrics": 0.15,
}


def parse_gtf(path):
    genes = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] not in ("gene", "exon"):
                continue
            m = re.search(r'gene_id "([^"]+)"', f[8])
            if not m:
                continue
            gid = m.group(1)
            g = genes.setdefault(gid, {"chrom": f[0], "strand": f[6],
                                       "start": int(f[3]), "end": int(f[4]),
                                       "exons": []})
            if f[2] == "exon":
                g["exons"].append((int(f[3]), int(f[4])))
    return genes


def ref_len(cigar):
    n = 0
    for c, op in re.findall(r"(\d+)([MIDNSHP=X])", cigar):
        if op in "MDN=X":
            n += int(c)
    return n


def load_libraries():
    libs = {}
    with open(os.path.join(DATA, "library_meta.tsv")) as fh:
        for line in fh:
            if line.startswith("cell_barcode"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            libs[p[0]] = {"library": p[1], "input_fg": float(p[2])}
    return libs


def load_ercc_expected():
    exp = {}
    with open(os.path.join(DATA, "ercc_expected.tsv")) as fh:
        for line in fh:
            if line.startswith("ercc_id"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                exp[p[0]] = float(p[1])
    return exp


def reference(genes):
    bychrom = defaultdict(list)
    for gid, g in genes.items():
        for s, e in g["exons"]:
            bychrom[g["chrom"]].append((s, e, gid))

    reads = []
    with open(os.path.join(DATA, "aligned.sam")) as fh:
        for line in fh:
            if line.startswith("@"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 11:
                continue
            try:
                flag = int(f[1]); mapq = int(f[4]); pos = int(f[3])
            except ValueError:
                continue
            if flag & 4 or mapq < 20:
                continue
            cb = ub = None
            for t in f[11:]:
                if t.startswith("CB:Z:"):
                    cb = t[5:]
                elif t.startswith("UB:Z:"):
                    ub = t[5:]
            if cb is None or ub is None:
                continue
            rl = ref_len(f[5])
            best, bov = None, 0
            for s, e, gid in bychrom.get(f[2], []):
                ov = min(e, pos + rl - 1) - max(s, pos) + 1
                if ov > bov:
                    bov, best = ov, gid
            if best is None or bov < 1:
                continue
            end3 = pos + rl - 1 if not (flag & 16) else pos
            reads.append((cb, best, ub.upper(), end3))

    groups = defaultdict(set)
    for cb, gid, umi, _end in reads:
        groups[(cb, gid)].add(umi)

    mol, fammap = {}, {}
    for key, us in groups.items():
        us = sorted(us)
        n = len(us)
        par = list(range(n))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                a, b = us[i], us[j]
                if len(a) != len(b):
                    continue
                d = 0
                for x, y in zip(a, b):
                    if x != y:
                        d += 1
                        if d > 1:
                            break
                if d <= 1:
                    ra, rb = find(i), find(j)
                    if ra != rb:
                        par[ra] = rb
        mp = {}
        for i, u in enumerate(us):
            mp[u] = find(i)
        mol[key] = len(set(mp.values()))
        fammap[key] = mp

    rcnt = Counter((cb, gid) for cb, gid, _u, _e in reads)

    binfam = defaultdict(set)
    for cb, gid, umi, end in reads:
        b = (end // 5) * 5
        binfam[(cb, gid, b)].add(fammap[(cb, gid)][umi])

    bycg = defaultdict(list)
    for (cb, gid, b), fams in binfam.items():
        bycg[(cb, gid)].append((b, len(fams)))
    modal = {}
    for k, lst in bycg.items():
        lst.sort(key=lambda t: (-t[1], t[0]))
        modal[k] = lst[0]

    total_mol = sum(mol.values())
    total_reads = len(reads)
    phantom = sum(max(0, len(s) - 1) for s in binfam.values())
    return {
        "mol": mol,
        "reads": rcnt,
        "modal": modal,
        "total_molecules": total_mol,
        "total_reads": total_reads,
        "phantom_index": (phantom / total_mol) if total_mol else 0.0,
    }


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    n = len(a)
    if n < 3:
        return 0.0
    ra, rb = _rank(a), _rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((rb[i] - mb) ** 2 for i in range(n)))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def load_counts(path):
    header = None
    rows = {}
    with open(path) as fh:
        first = True
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            p = line.split("\t")
            if first:
                first = False
                if p and p[0] == "cell":
                    header = p
                    continue
            if len(p) != 4:
                raise ValueError("counts.tsv row is not 4 columns")
            rows[(p[0], p[1])] = (int(p[2]), int(p[3]))
    return header, rows


def main():
    res = {k: 0.0 for k in WEIGHTS}
    detail = {"errors": []}

    try:
        genes = parse_gtf(os.path.join(DATA, "annotation.gtf"))
        gold = reference(genes)
        libs = load_libraries()
        nc_cells = {c for c, v in libs.items() if v["input_fg"] == 0}
        ercc_expected = load_ercc_expected()
    except Exception as exc:  # noqa: BLE001
        detail["errors"].append("reference failure: %r" % (exc,))
        gold = None

    header = rows = None
    try:
        header, rows = load_counts(os.path.join(OUT, "counts.tsv"))
    except Exception as exc:  # noqa: BLE001
        detail["errors"].append("counts.tsv unreadable: %r" % (exc,))

    if gold is not None and rows is not None:
        # --- C1: schema -------------------------------------------------
        ok = header == ["cell", "gene", "molecules", "reads"]
        if ok:
            for _k, (m, r) in rows.items():
                if m < 1 or r < 1:
                    ok = False
                    break
        res["schema_counts"] = 1.0 if ok else 0.0

        # --- C2: exact total molecule count -----------------------------
        agent_total = sum(m for m, _r in rows.values())
        res["total_molecules"] = 1.0 if agent_total == gold["total_molecules"] else 0.0
        detail["agent_total_molecules"] = agent_total
        detail["reference_total_molecules"] = gold["total_molecules"]

        # --- C3: per-(cell,gene) molecule agreement ---------------------
        gold_keys = set(gold["mol"])
        matched = sum(1 for k in gold_keys
                      if k in rows and rows[k][0] == gold["mol"][k])
        denom = max(len(gold_keys), len(rows), 1)
        frac = matched / denom
        res["row_agreement"] = max(0.0, min(1.0, (frac - 0.50) / 0.47))
        detail["row_match_fraction"] = round(frac, 4)

        # --- C4: negative-control background ----------------------------
        nc_agent = sum(m for (c, _g), (m, _r) in rows.items() if c in nc_cells)
        detail["nc_molecules_agent"] = nc_agent
        res["nc_background"] = 1.0 if nc_agent <= 5 else 0.0

    # --- C5: transcript end profile -------------------------------------
    try:
        ep = {}
        with open(os.path.join(OUT, "end_profile.tsv")) as fh:
            hdr = None
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                p = line.split("\t")
                if hdr is None and p[0] == "cell":
                    hdr = p
                    continue
                if len(p) != 5:
                    continue
                ep[(p[0], p[1])] = (p[2], int(p[3]), int(p[4]))
        if hdr == ["cell", "gene", "strand", "modal_end", "modal_families"] and gold:
            n_ok = 0
            for k, (b, _nf) in gold["modal"].items():
                if k in ep and abs(ep[k][1] - b) <= 5:
                    n_ok += 1
            frac = n_ok / max(1, len(gold["modal"]))
            res["end_profile"] = max(0.0, min(1.0, (frac - 0.50) / 0.45))
            detail["end_modal_fraction"] = round(frac, 4)
    except Exception as exc:  # noqa: BLE001
        detail["errors"].append("end_profile.tsv: %r" % (exc,))

    # --- C6: ERCC spike-in correlation ----------------------------------
    try:
        if rows is not None and libs:
            cells = [c for c, v in libs.items() if v["library"] == "IN1000"]
            if cells and ercc_expected:
                cell = cells[0]
                xs = [ercc_expected[n] for n in sorted(ercc_expected)]
                ys = [rows.get((cell, n), (0, 0))[0] for n in sorted(ercc_expected)]
                rho = spearman(xs, ys)
                detail["ercc_spearman"] = round(rho, 4)
                res["ercc_correlation"] = (1.0 if rho >= 0.90
                                           else max(0.0, (rho - 0.50) / 0.40))
    except Exception as exc:  # noqa: BLE001
        detail["errors"].append("ercc: %r" % (exc,))

    # --- C7: QC metrics --------------------------------------------------
    try:
        with open(os.path.join(OUT, "metrics.json")) as fh:
            mj = json.load(fh)
        if gold:
            checks = []
            checks.append(abs(mj.get("total_molecules", -1) - gold["total_molecules"]) < 0.5)
            checks.append(abs(mj.get("total_reads_assigned", -1) - gold["total_reads"]) < 0.5)
            dedup_gold = (1.0 - gold["total_molecules"] / gold["total_reads"]) if gold["total_reads"] else 0.0
            checks.append(abs(mj.get("dedup_rate", -1) - dedup_gold) <= 0.005)
            checks.append(abs(mj.get("phantom_index", -1) - gold["phantom_index"]) <= 0.02)
            nc_gold = sum(v for k, v in gold["mol"].items() if k[0] in nc_cells)
            checks.append(abs(mj.get("nc_molecules", -1) - nc_gold) < 0.5)
            res["qc_metrics"] = sum(1 for c in checks if c) / len(checks)
            detail["qc_checks_passed"] = sum(1 for c in checks if c)
            detail["reference_phantom_index"] = round(gold["phantom_index"], 4)
    except Exception as exc:  # noqa: BLE001
        detail["errors"].append("metrics.json: %r" % (exc,))

    score = sum(WEIGHTS[k] * res[k] for k in WEIGHTS)
    score = round(max(0.0, min(1.0, score)), 4)
    detail["constraints"] = {k: round(v, 4) for k, v in res.items()}
    detail["score"] = score

    os.makedirs(LOG, exist_ok=True)
    with open(os.path.join(LOG, "reward.json"), "w") as fh:
        json.dump({"score": score, "reward": score}, fh)
    print(json.dumps(detail, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        os.makedirs(LOG, exist_ok=True)
        with open(os.path.join(LOG, "reward.json"), "w") as fh:
            json.dump({"score": 0.0, "reward": 0.0}, fh)
        print(json.dumps({"fatal": repr(exc)}))
        raise SystemExit(0)
