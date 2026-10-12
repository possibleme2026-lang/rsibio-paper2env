import json, math, sys
sys.path.insert(0, '/tests')
from gold_gen import compute_gold

def approx(a, b, tol=1e-3):
    if a is None or b is None:
        return False
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= tol

def main():
    gold = compute_gold()
    try:
        with open('/app/output/result.json') as f:
            agent = json.load(f)
    except Exception as e:
        print('Failed to load agent output:', e)
        with open('/logs/verifier/reward.json', 'w') as f:
            json.dump({'score': 0.0, 'reward': 0.0}, f)
        return
    checks = []
    # Check 1: G4 counts
    ok = True
    for species, gdata in gold['species'].items():
        if species not in agent.get('species', {}):
            ok = False; break
        for comp in ['CDS', 'intron', 'promoter']:
            a = agent['species'][species].get('g4_counts', {}).get(comp)
            b = gdata['g4_counts'][comp]
            if a != b:
                ok = False; break
    checks.append(('G4 counts', ok))
    # Check 2: densities
    ok = True
    for species, gdata in gold['species'].items():
        for comp in ['CDS', 'intron', 'promoter']:
            a = agent['species'][species].get('g4_density_per_kb', {}).get(comp)
            b = gdata['g4_density_per_kb'][comp]
            if not approx(a, b, 1e-3):
                ok = False; break
    checks.append(('G4 densities', ok))
    # Check 3: log2 IRR
    ok = True
    for species, gdata in gold['species'].items():
        for comp in ['CDS', 'intron', 'promoter']:
            a = agent['species'][species].get('log2_IRR', {}).get(comp)
            b = gdata['log2_IRR'][comp]
            if not approx(a, b, 1e-3):
                ok = False; break
    checks.append(('log2 IRR', ok))
    # Check 4: naive ORs
    ok = True
    for key, gdata in gold['species_pairs'].items():
        if key not in agent.get('species_pairs', {}):
            ok = False; break
        a = agent['species_pairs'][key].get('naive_OR')
        b = gdata['naive_OR']
        if not approx(a, b, 1e-2):
            ok = False; break
    checks.append(('naive ORs', ok))
    # Check 5: adjusted ORs
    ok = True
    for key, gdata in gold['species_pairs'].items():
        a = agent['species_pairs'][key].get('adjusted_OR')
        b = gdata['adjusted_OR']
        if not approx(a, b, 1e-2):
            ok = False; break
    checks.append(('adjusted ORs', ok))
    # Check 6: summary and CAI examples
    ok = True
    if agent.get('summary', {}).get('n_pairs_naive_OR_lt_1') != gold['summary']['n_pairs_naive_OR_lt_1']:
        ok = False
    if agent.get('summary', {}).get('n_pairs_adjusted_OR_lt_1') != gold['summary']['n_pairs_adjusted_OR_lt_1']:
        ok = False
    for pid, val in gold['cai_examples'].items():
        a = agent.get('cai_examples', {}).get(pid)
        if not approx(a, val, 1e-3):
            ok = False; break
    checks.append(('summary & CAI', ok))
    passed = sum(1 for _, v in checks if v)
    score = passed / len(checks)
    reward = score
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({'score': score, 'reward': reward}, f)
    for name, v in checks:
        print(f"{name}: {'PASS' if v else 'FAIL'}")
    print(f"Score: {score}")

if __name__ == '__main__':
    main()
