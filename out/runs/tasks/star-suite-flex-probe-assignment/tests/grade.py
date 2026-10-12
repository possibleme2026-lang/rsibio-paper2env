#!/usr/bin/env python3
import json
import os
import sys
import tempfile
import subprocess
import shutil
from collections import defaultdict
import numpy as np

def parse_fastq(path):
    reads = []
    with open(path) as f:
        while True:
            header = f.readline()
            if not header:
                break
            seq = f.readline().strip()
            plus = f.readline()
            qual = f.readline().strip()
            reads.append(seq)
    return reads

def revcomp(s):
    comp = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}
    return ''.join(comp.get(c, 'N') for c in reversed(s))

def load_probes(path):
    probes = []
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                probes.append((parts[0], parts[1], parts[2]))
    return probes

def load_cell_barcodes(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def seq_to_arr(s):
    base_to_int = {'A':0, 'C':1, 'G':2, 'T':3, 'N':4}
    return np.array([base_to_int.get(c, 4) for c in s], dtype=np.int8)

def compute_reference(probes, cell_barcodes, reads):
    probe_ids = [p[0] for p in probes]
    sample_ids = [p[1] for p in probes]
    probe_seqs = [p[2] for p in probes]
    
    probe_arr = np.array([seq_to_arr(s) for s in probe_seqs])
    probe_rc_arr = np.array([seq_to_arr(revcomp(s)) for s in probe_seqs])
    bc_arr = np.array([seq_to_arr(bc) for bc in cell_barcodes])
    
    total_reads = len(reads)
    assigned_reads = 0
    ambiguous_probe_reads = 0
    unassigned_probe_reads = 0
    invalid_cell_barcode_reads = 0
    corrected_cell_barcodes = 0
    molecules = set()
    
    for read in reads:
        if len(read) < 68:
            continue
        bc = read[0:16]
        umi = read[16:28]
        probe_seq = read[28:68]
        
        bc_read_arr = seq_to_arr(bc)
        diffs = (bc_arr != bc_read_arr).sum(axis=1)
        min_dist = diffs.min()
        if min_dist == 0:
            idx = np.where(diffs == 0)[0]
            if len(idx) != 1:
                invalid_cell_barcode_reads += 1
                continue
            corrected_bc = cell_barcodes[idx[0]]
        elif min_dist == 1:
            idx = np.where(diffs == 1)[0]
            if len(idx) != 1:
                invalid_cell_barcode_reads += 1
                continue
            corrected_bc = cell_barcodes[idx[0]]
            corrected_cell_barcodes += 1
        else:
            invalid_cell_barcode_reads += 1
            continue
        
        probe_read_arr = seq_to_arr(probe_seq)
        dist_fwd = (probe_arr != probe_read_arr).sum(axis=1)
        dist_rev = (probe_rc_arr != probe_read_arr).sum(axis=1)
        min_fwd = dist_fwd.min()
        min_rev = dist_rev.min()
        global_min = min(min_fwd, min_rev)
        
        if global_min > 2:
            unassigned_probe_reads += 1
            continue
        
        candidates = set()
        if min_fwd == global_min:
            for i in np.where(dist_fwd == global_min)[0]:
                candidates.add(i)
        if min_rev == global_min:
            for i in np.where(dist_rev == global_min)[0]:
                candidates.add(i)
        
        if len(candidates) != 1:
            ambiguous_probe_reads += 1
            continue
        
        probe_idx = candidates.pop()
        assigned_reads += 1
        probe_id = probe_ids[probe_idx]
        sample_id = sample_ids[probe_idx]
        molecules.add((sample_id, probe_id, corrected_bc, umi))
    
    counts = defaultdict(int)
    for (sample_id, probe_id, bc, umi) in molecules:
        counts[(probe_id, sample_id)] += 1
    
    all_probes = sorted(set(p[0] for p in probes))
    all_samples = sorted(set(p[1] for p in probes))
    gold_counts = {}
    for pid in all_probes:
        for sid in all_samples:
            gold_counts[(pid, sid)] = counts.get((pid, sid), 0)
    
    gold_summary = {
        "total_reads": total_reads,
        "assigned_reads": assigned_reads,
        "ambiguous_probe_reads": ambiguous_probe_reads,
        "unassigned_probe_reads": unassigned_probe_reads,
        "invalid_cell_barcode_reads": invalid_cell_barcode_reads,
        "corrected_cell_barcodes": corrected_cell_barcodes,
        "unique_umis_total": len(molecules)
    }
    
    return gold_counts, gold_summary

def parse_agent_counts(path, probes):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        header = f.readline().strip()
        if header != 'probe_id\tsample_id\tumi_count':
            return None
        counts = {}
        rows = 0
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                return None
            pid, sid, cnt = parts
            try:
                cnt = int(cnt)
            except ValueError:
                return None
            counts[(pid, sid)] = cnt
            rows += 1
        if rows != 1800:
            return None
        return counts

def parse_agent_summary(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data
    except Exception:
        return None

def check_constraints(agent_counts, agent_summary, gold_counts, gold_summary, probes):
    constraints = []
    
    c1_pass = agent_counts is not None
    constraints.append(("C1_probe_counts_format", c1_pass, "probe_counts.tsv has correct header and 1800 rows"))
    
    required_keys = ["total_reads", "assigned_reads", "ambiguous_probe_reads",
                     "unassigned_probe_reads", "invalid_cell_barcode_reads",
                     "corrected_cell_barcodes", "unique_umis_total"]
    c2_pass = False
    if agent_summary is not None:
        c2_pass = all(k in agent_summary and isinstance(agent_summary[k], int) for k in required_keys)
    constraints.append(("C2_summary_format", c2_pass, "summary.json has all required integer keys"))
    
    c3_pass = False
    if agent_summary is not None and "total_reads" in agent_summary:
        c3_pass = agent_summary["total_reads"] == gold_summary["total_reads"]
    constraints.append(("C3_total_reads", c3_pass, f"total_reads == {gold_summary['total_reads']}"))
    
    c4_pass = False
    if agent_summary is not None:
        c4_pass = (agent_summary.get("invalid_cell_barcode_reads") == gold_summary["invalid_cell_barcode_reads"] and
                   agent_summary.get("corrected_cell_barcodes") == gold_summary["corrected_cell_barcodes"])
    constraints.append(("C4_cell_barcode_stats", c4_pass, "invalid_cell_barcode_reads and corrected_cell_barcodes match gold"))
    
    c5_pass = False
    if agent_summary is not None:
        c5_pass = (agent_summary.get("assigned_reads") == gold_summary["assigned_reads"] and
                   agent_summary.get("ambiguous_probe_reads") == gold_summary["ambiguous_probe_reads"] and
                   agent_summary.get("unassigned_probe_reads") == gold_summary["unassigned_probe_reads"])
    constraints.append(("C5_probe_assignment_stats", c5_pass, "assigned, ambiguous, unassigned read counts match gold"))
    
    c6_pass = False
    if agent_summary is not None:
        c6_pass = agent_summary.get("unique_umis_total") == gold_summary["unique_umis_total"]
    constraints.append(("C6_unique_umis_total", c6_pass, f"unique_umis_total == {gold_summary['unique_umis_total']}"))
    
    c7_pass = False
    if agent_counts is not None:
        c7_pass = True
        for key in gold_counts:
            if agent_counts.get(key) != gold_counts[key]:
                c7_pass = False
                break
    constraints.append(("C7_counts_match", c7_pass, "all 1800 probe counts match gold exactly"))
    
    return constraints

def main():
    tmpdir = tempfile.mkdtemp()
    try:
        gold_gen_src = '/tests/gold_gen.py'
        if not os.path.exists(gold_gen_src):
            gold_gen_src = '/app/data/gen_inputs.py'
        shutil.copy(gold_gen_src, os.path.join(tmpdir, 'gold_gen.py'))
        subprocess.run(['python', 'gold_gen.py'], cwd=tmpdir, check=True, capture_output=True)
        
        probes = load_probes(os.path.join(tmpdir, 'probes.tsv'))
        cell_barcodes = load_cell_barcodes(os.path.join(tmpdir, 'cell_barcodes.txt'))
        reads = parse_fastq(os.path.join(tmpdir, 'reads.fastq'))
        
        gold_counts, gold_summary = compute_reference(probes, cell_barcodes, reads)
    finally:
        shutil.rmtree(tmpdir)
    
    agent_counts = parse_agent_counts('/app/output/probe_counts.tsv', probes)
    agent_summary = parse_agent_summary('/app/output/summary.json')
    
    constraints = check_constraints(agent_counts, agent_summary, gold_counts, gold_summary, probes)
    
    passed = sum(1 for _, p, _ in constraints if p)
    total = len(constraints)
    score = passed / total if total > 0 else 0.0
    
    os.makedirs('/logs/verifier', exist_ok=True)
    reward = {
        "score": score,
        "reward": score,
        "constraints": [{"name": name, "passed": p, "description": desc} for name, p, desc in constraints]
    }
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump(reward, f, indent=2)
    
    print(json.dumps(reward, indent=2))

if __name__ == '__main__':
    main()
