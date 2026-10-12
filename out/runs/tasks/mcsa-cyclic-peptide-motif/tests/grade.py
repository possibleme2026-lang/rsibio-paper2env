#!/usr/bin/env python3
import json
import csv
import os
import sys
from gold_gen import generate_gold

L = 18
M = 8
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)

def load_input():
    ids = []
    seqs = []
    with open('/app/data/cyclic_peptides.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row['peptide_id'])
            seqs.append(row['sequence'])
    return ids, seqs

def check_schema(data):
    if not isinstance(data, dict):
        return False, "Output is not a JSON object"
    if 'consensus_motif' not in data:
        return False, "Missing key: consensus_motif"
    if 'rotations' not in data:
        return False, "Missing key: rotations"
    if not isinstance(data['consensus_motif'], str):
        return False, "consensus_motif must be a string"
    if not isinstance(data['rotations'], dict):
        return False, "rotations must be an object"
    return True, "Schema OK"

def check_rotations_map(rotations, ids):
    if len(rotations) != len(ids):
        return False, f"Expected {len(ids)} rotations, got {len(rotations)}"
    if set(rotations.keys()) != set(ids):
        missing = set(ids) - set(rotations.keys())
        extra = set(rotations.keys()) - set(ids)
        return False, f"Rotations map key mismatch. Missing: {missing}, Extra: {extra}"
    for pid in ids:
        r = rotations[pid]
        if not isinstance(r, int) or isinstance(r, bool):
            return False, f"Rotation for {pid} is not an integer: {r}"
        if r < 0 or r >= L:
            return False, f"Rotation for {pid} out of range: {r}"
    return True, "Rotations map OK"

def compute_rotated(seq, rot):
    return seq[rot:] + seq[:rot] if rot > 0 else seq

def check_rotation_accuracy(rotations, ids, input_seqs, gold_rotations):
    correct = 0
    for pid, seq in zip(ids, input_seqs):
        t_agent = rotations[pid]
        t_true = gold_rotations[pid]
        idx = (t_true - t_agent) % L
        if idx == 0:
            correct += 1
    return correct / len(ids)

def check_consensus_accuracy(agent_consensus, gold_consensus):
    if len(agent_consensus) != M:
        return 0.0
    for c in agent_consensus:
        if c not in AA_SET:
            return 0.0
    matches = sum(1 for a, b in zip(agent_consensus, gold_consensus) if a == b)
    return matches / M

def check_self_consistency(rotations, ids, input_seqs, agent_consensus):
    if len(agent_consensus) != M:
        return 0.0
    aligned = []
    for pid, seq in zip(ids, input_seqs):
        r = rotations[pid]
        aligned.append(compute_rotated(seq, r))
    min_frac = 1.0
    for j in range(M):
        char = agent_consensus[j]
        count = sum(1 for s in aligned if s[j] == char)
        frac = count / len(aligned)
        if frac < min_frac:
            min_frac = frac
    return min_frac

def main():
    gold = generate_gold()
    gold_rotations = gold['rotations']
    gold_consensus = gold['consensus']

    ids, seqs = load_input()

    out_path = '/app/output/mcsa_results.json'
    if not os.path.exists(out_path):
        score = 0.0
        result = {"score": score, "reward": score, "error": "Output file not found"}
        os.makedirs('/logs/verifier', exist_ok=True)
        with open('/logs/verifier/reward.json', 'w') as f:
            json.dump(result, f)
        return

    try:
        with open(out_path) as f:
            data = json.load(f)
    except Exception as e:
        score = 0.0
        result = {"score": score, "reward": score, "error": f"Invalid JSON: {e}"}
        os.makedirs('/logs/verifier', exist_ok=True)
        with open('/logs/verifier/reward.json', 'w') as f:
            json.dump(result, f)
        return

    schema_ok, schema_msg = check_schema(data)
    if not schema_ok:
        score = 0.0
        result = {"score": score, "reward": score, "error": schema_msg}
        os.makedirs('/logs/verifier', exist_ok=True)
        with open('/logs/verifier/reward.json', 'w') as f:
            json.dump(result, f)
        return

    agent_consensus = data['consensus_motif']
    rotations = data['rotations']

    rot_ok, rot_msg = check_rotations_map(rotations, ids)
    if not rot_ok:
        rot_score = 0.0
        cons_score = check_consensus_accuracy(agent_consensus, gold_consensus)
        self_score = 0.0
        schema_score = 1.0
        map_score = 0.0
        total = 0.10 * schema_score + 0.10 * map_score + 0.45 * rot_score + 0.25 * cons_score + 0.10 * self_score
        result = {"score": total, "reward": total, "error": rot_msg,
                  "rotation_accuracy": 0.0, "consensus_accuracy": cons_score,
                  "self_consistency_min_freq": 0.0, "self_consistency_pass": False}
    else:
        schema_score = 1.0
        map_score = 1.0
        rot_acc = check_rotation_accuracy(rotations, ids, seqs, gold_rotations)
        cons_acc = check_consensus_accuracy(agent_consensus, gold_consensus)
        self_cons = check_self_consistency(rotations, ids, seqs, agent_consensus)
        self_score = 1.0 if self_cons >= 0.70 else 0.0
        total = 0.10 * schema_score + 0.10 * map_score + 0.45 * rot_acc + 0.25 * cons_acc + 0.10 * self_score
        result = {
            "score": total,
            "reward": total,
            "rotation_accuracy": rot_acc,
            "consensus_accuracy": cons_acc,
            "self_consistency_min_freq": self_cons,
            "self_consistency_pass": self_cons >= 0.70,
            "message": "OK"
        }

    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    main()
