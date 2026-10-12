#!/usr/bin/env python3
import json, os, sys, re, math
import primer3
sys.path.append('/tests')
import gold_gen

RESULT_PATH = '/app/output/result.json'
REWARD_PATH = '/logs/verifier/reward.json'

def reverse_complement(seq):
    comp = {'A':'T','T':'A','C':'G','G':'C','N':'N'}
    return ''.join(comp.get(b,b) for b in reversed(seq))

def gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq) * 100

def no_four_consecutive(seq):
    return not re.search(r'(.)\1{3,}', seq)

def main():
    os.makedirs('/logs/verifier', exist_ok=True)
    config = gold_gen.CONFIG
    ref = gold_gen.generate_reference(gold_gen.SEED, config)
    mut_idx = 292
    gold_freqs = {pop: cfg['freqs'] for pop, cfg in config['populations'].items()}
    gold_carries = {pop: {k: (v > 0.05) for k,v in freqs.items()} for pop, freqs in gold_freqs.items()}
    gold_resistant = {pop: (gold_carries[pop]['R98L'] or gold_carries[pop]['R98Q']) for pop in gold_freqs}

    total = 0
    passed = 0
    def check(cond):
        nonlocal total, passed
        total += 1
        if cond:
            passed += 1
            return True
        return False

    try:
        with open(RESULT_PATH) as f:
            data = json.load(f)
        check(True)
    except Exception:
        check(False)
        score = passed / total if total > 0 else 0.0
        with open(REWARD_PATH, 'w') as f:
            json.dump({"score": score, "reward": score}, f)
        return

    check('populations' in data)
    check('diagnostic_primers' in data)
    pops = data.get('populations', {})
    check(all(p in pops for p in ['S','R1','R2']))
    for p in ['S','R1','R2']:
        pop_data = pops.get(p, {})
        check('variant_frequencies' in pop_data)
        check('carries_R98L' in pop_data)
        check('carries_R98Q' in pop_data)
        check('resistant' in pop_data)
        vf = pop_data.get('variant_frequencies', {})
        check(all(k in vf for k in ['WT_R98','R98L','R98Q']))
    dp = data.get('diagnostic_primers', {})
    check(all(k in dp for k in ['forward','reverse','product_size','target_allele','primer_with_3p_mutation']))

    for p in ['S','R1','R2']:
        pop_data = pops.get(p, {})
        vf = pop_data.get('variant_frequencies', {})
        try:
            s = sum(float(vf.get(k, 0.0)) for k in ['WT_R98','R98L','R98Q'])
            check(abs(s - 1.0) <= 0.01)
        except Exception:
            check(False)
        for k in ['WT_R98','R98L','R98Q']:
            try:
                val = float(vf.get(k, 0.0))
                gold = gold_freqs[p][k]
                check(abs(val - gold) <= 0.05)
            except Exception:
                check(False)
        for k in ['R98L','R98Q']:
            try:
                val = float(vf.get(k, 0.0))
                called = val > 0.05
                check(called == gold_carries[p][k])
            except Exception:
                check(False)
        try:
            check(bool(pop_data.get('resistant')) == gold_resistant[p])
        except Exception:
            check(False)

    fwd = dp.get('forward', '')
    rev = dp.get('reverse', '')
    mut_primer = dp.get('primer_with_3p_mutation', '')
    reported_size = dp.get('product_size', 0)
    check(mut_primer in ['forward','reverse'])
    check(dp.get('target_allele') == 'R98Q')

    if mut_primer == 'forward':
        L = len(fwd)
        if L > 0:
            expected_prefix = ref[mut_idx - L + 1 : mut_idx]
            check(fwd[:-1] == expected_prefix and fwd[-1] == 'A')
        else:
            check(False)
        rev_comp = reverse_complement(rev)
        pos = ref.find(rev_comp)
        check(pos != -1)
        if pos != -1:
            Lr = len(rev)
            f_start = mut_idx - len(fwd) + 1
            r_end = pos + Lr - 1
            calc_size = r_end - f_start + 1
            check(abs(calc_size - reported_size) <= 1)
            check(150 <= calc_size <= 300)
        else:
            check(False)
            check(False)
    elif mut_primer == 'reverse':
        L = len(rev)
        if L > 0:
            rev_comp = reverse_complement(rev)
            check(rev_comp[0] == 'A' and rev_comp[1:] == ref[mut_idx+1 : mut_idx+L])
        else:
            check(False)
        pos = ref.find(fwd)
        check(pos != -1)
        if pos != -1:
            Lf = len(fwd)
            f_start = pos
            r_end = mut_idx + len(rev) - 1
            calc_size = r_end - f_start + 1
            check(abs(calc_size - reported_size) <= 1)
            check(150 <= calc_size <= 300)
        else:
            check(False)
            check(False)
    else:
        for _ in range(4):
            check(False)

    for name, seq in [('forward', fwd), ('reverse', rev)]:
        if seq:
            try:
                tm = primer3.calc_tm(seq)
                check(58 <= tm <= 62)
            except Exception:
                check(False)
            try:
                gc = gc_content(seq)
                check(40 <= gc <= 60)
            except Exception:
                check(False)
            check(no_four_consecutive(seq))
        else:
            check(False)
            check(False)
            check(False)

    score = passed / total if total > 0 else 0.0
    with open(REWARD_PATH, 'w') as f:
        json.dump({"score": score, "reward": score}, f)

if __name__ == '__main__':
    main()
