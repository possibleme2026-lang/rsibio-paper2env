#!/usr/bin/env python3
import json
import os
import sys
import random

sys.path.append('/tests')
from simulation import binds, AA
from gold_gen import generate_gold

def main():
    gold = generate_gold()
    solution_path = '/app/output/solution.json'
    if not os.path.exists(solution_path):
        print("Solution file not found")
        score = 0.0
    else:
        with open(solution_path) as f:
            sol = json.load(f)
        
        scores = []
        
        # Constraint 1: Ancestral states
        correct = 0
        total = 0
        for node, states in gold['ancestral_states'].items():
            if node in sol.get('ancestral_states', {}):
                for pos, aa in states.items():
                    total += 1
                    if sol['ancestral_states'][node].get(pos) == aa:
                        correct += 1
            else:
                total += len(states)
        c1_score = correct / total if total > 0 else 0.0
        scores.append(c1_score)
        
        # Constraint 2: Extant binding F1
        gold_binding = gold['extant_binding']
        pred_binding = sol.get('extant_binding', {})
        tp = fp = fn = 0
        for species, true_val in gold_binding.items():
            pred_val = pred_binding.get(species, False)
            if true_val and pred_val:
                tp += 1
            elif not true_val and pred_val:
                fp += 1
            elif true_val and not pred_val:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        c2_score = f1
        scores.append(c2_score)
        
        # Constraint 3: Mutation order
        gold_order = gold['mutation_order']
        pred_order = sol.get('mutation_order', [])
        c3_score = 1.0 if pred_order == gold_order else 0.0
        scores.append(c3_score)
        
        # Constraint 4: Binding rule
        random.seed(123)
        test_set = []
        for rule in gold['binding_rule']:
            test_set.append((rule['RAR1_75'], rule['SGT1_100'], rule['SGT1_130'], rule['SGT1_160']))
        aa_seen = set()
        for rule in gold['binding_rule']:
            aa_seen.update([rule['RAR1_75'], rule['SGT1_100'], rule['SGT1_130'], rule['SGT1_160']])
        aa_list = list(aa_seen)
        for _ in range(100):
            test_set.append(tuple(random.choice(aa_list) for _ in range(4)))
        
        pred_rule = sol.get('binding_rule', [])
        pred_set = set()
        for item in pred_rule:
            pred_set.add((item['RAR1_75'], item['SGT1_100'], item['SGT1_130'], item['SGT1_160']))
        
        tp = fp = fn = 0
        for tup in test_set:
            true_bind = binds(tup[0], tup[1], tup[2], tup[3])
            pred_bind = tup in pred_set
            if true_bind and pred_bind:
                tp += 1
            elif not true_bind and pred_bind:
                fp += 1
            elif true_bind and not pred_bind:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_rule = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        c4_score = f1_rule
        scores.append(c4_score)
        
        score = sum(scores) / len(scores)
    
    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({"score": score, "reward": score}, f)
    print(f"Score: {score}")

if __name__ == '__main__':
    main()
