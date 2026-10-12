#!/bin/bash
set -euo pipefail
mkdir -p /app/output
cat > /app/solve.py <<'PY'
import json, os, re
from collections import defaultdict, Counter

DATA = "/app/data"


def parse_gtf(path):
    genes = {}
    for line in open(path):
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


def main():
    genes = parse_gtf(os.path.join(DATA, "annotation.gtf"))
    bychrom = defaultdict(list)
    for gid, g in genes.items():
        for s, e in g["exons"]:
            bychrom[g["chrom"]].append((s, e, gid))

    reads = []
    for line in open(os.path.join(DATA, "aligned.sam")):
        if line.startswith("@"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 11:
            continue
        flag = int(f[1]); mapq = int(f[4]); pos = int(f[3])
        if (flag & 4) or mapq < 20:
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
    for cb, gid, umi, _e in reads:
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
        mp = {u: find(i) for i, u in enumerate(us)}
        mol[key] = len(set(mp.values()))
        fammap[key] = mp

    rcnt = Counter((cb, gid) for cb, gid, _u, _e in reads)

    binfam = defaultdict(set)
    for cb, gid, umi, end in reads:
        binfam[(cb, gid, (end // 5) * 5)].add(fammap[(cb, gid)][umi])

    bycg = defaultdict(list)
    for (cb, gid, b), fams in binfam.items():
        bycg[(cb, gid)].append((b, len(fams)))
    modal = {}
    for k, lst in bycg.items():
        lst.sort(key=lambda t: (-t[1], t[0]))
        modal[k] = lst[0]

    libs = {}
    for line in open(os.path.join(DATA, "library_meta.tsv")):
        if line.startswith("cell_barcode"):
            continue
        p = line.rstrip("\n").split("\t")
        libs[p[0]] = (p[1], float(p[2]))

    total_mol = sum(mol.values())
    total_reads = len(reads)
    phantom = sum(max(0, len(s) - 1) for s in binfam.values())
    nc_cells = {c for c, (_n, fg) in libs.items() if fg == 0.0}
    nc_mol = sum(v for k, v in mol.items() if k[0] in nc_cells)

    with open("/app/output/counts.tsv", "w") as fh:
        fh.write("cell\tgene\tmolecules\treads\n")
        for k in sorted(mol):
            fh.write("%s\t%s\t%d\t%d\n" % (k[0], k[1], mol[k], rcnt.get(k, 0)))

    with open("/app/output/end_profile.tsv", "w") as fh:
        fh.write("cell\tgene\tstrand\tmodal_end\tmodal_families\n")
        for k in sorted(modal):
            fh.write("%s\t%s\t%s\t%d\t%d\n" % (k[0], k[1], genes[k[1]]["strand"],
                                                 modal[k][0], modal[k][1]))

    metrics = {
        "total_molecules": total_mol,
        "total_reads_assigned": total_reads,
        "dedup_rate": (1.0 - total_mol / total_reads) if total_reads else 0.0,
        "phantom_index": (phantom / total_mol) if total_mol else 0.0,
        "nc_molecules": nc_mol,
    }
    with open("/app/output/metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)


main()
PY
python /app/solve.py
