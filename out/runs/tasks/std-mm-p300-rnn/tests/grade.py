import json, os, sys, numpy as np
sys.path.append('/tests')
from gold_gen import compute_gold

def main():
    output_path = '/app/output/metrics.json'
    report_path = '/logs/verifier/report.json'
    reward_path = '/logs/verifier/reward.json'
    os.makedirs('/logs/verifier', exist_ok=True)
    constraints = []
    details = {}
    try:
        with open(output_path) as f:
            agent = json.load(f)
        schema_ok = all(k in agent for k in ['MMN_sensory', 'P300_frontal', 'P300_latency_delay', 'Noise_accuracy', 'State_space_expansion', 'Attention_amplification', 'status'])
        schema_ok = schema_ok and agent.get('status') == 'ok'
        for k in ['MMN_sensory', 'P300_frontal', 'P300_latency_delay', 'Noise_accuracy', 'State_space_expansion', 'Attention_amplification']:
            v = agent.get(k)
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                schema_ok = False
        constraints.append(('schema', schema_ok))
        details['agent'] = agent
    except Exception as e:
        constraints.append(('schema', False))
        details['error'] = str(e)
        agent = None

    try:
        gold = compute_gold('/app/data')
        details['gold'] = gold
    except Exception as e:
        details['gold_error'] = str(e)
        gold = None

    if agent is not None and gold is not None:
        try:
            v = agent['MMN_sensory']
            g = gold['MMN_sensory']
            ok = (v > 0) and (abs(v - g) <= max(0.01, 0.05 * abs(g)))
            constraints.append(('MMN_sensory', ok))
            details['MMN_sensory'] = {'agent': v, 'gold': g, 'ok': ok}
        except Exception as e:
            constraints.append(('MMN_sensory', False))
        try:
            v = agent['P300_frontal']
            g = gold['P300_frontal']
            mmn = agent['MMN_sensory']
            ok = (v > mmn) and (abs(v - g) <= max(0.01, 0.05 * abs(g)))
            constraints.append(('P300_frontal', ok))
            details['P300_frontal'] = {'agent': v, 'gold': g, 'ok': ok}
        except Exception as e:
            constraints.append(('P300_frontal', False))
        try:
            v = agent['P300_latency_delay']
            g = gold['P300_latency_delay']
            ok = (v > 0) and (abs(v - g) <= 5.0)
            constraints.append(('P300_latency_delay', ok))
            details['P300_latency_delay'] = {'agent': v, 'gold': g, 'ok': ok}
        except Exception as e:
            constraints.append(('P300_latency_delay', False))
        try:
            v = agent['Noise_accuracy']
            g = gold['Noise_accuracy']
            ok = (v >= 0.6) and (abs(v - g) <= 0.05)
            constraints.append(('Noise_accuracy', ok))
            details['Noise_accuracy'] = {'agent': v, 'gold': g, 'ok': ok}
        except Exception as e:
            constraints.append(('Noise_accuracy', False))
        try:
            v = agent['State_space_expansion']
            g = gold['State_space_expansion']
            ok = (v > 0) and (abs(v - g) <= 0.05)
            constraints.append(('State_space_expansion', ok))
            details['State_space_expansion'] = {'agent': v, 'gold': g, 'ok': ok}
        except Exception as e:
            constraints.append(('State_space_expansion', False))
        try:
            v = agent['Attention_amplification']
            g = gold['Attention_amplification']
            ok = (v > 0) and (abs(v - g) <= 0.05)
            constraints.append(('Attention_amplification', ok))
            details['Attention_amplification'] = {'agent': v, 'gold': g, 'ok': ok}
        except Exception as e:
            constraints.append(('Attention_amplification', False))
    else:
        for name in ['MMN_sensory', 'P300_frontal', 'P300_latency_delay', 'Noise_accuracy', 'State_space_expansion', 'Attention_amplification']:
            if not any(c[0] == name for c in constraints):
                constraints.append((name, False))

    total = len(constraints)
    passed = sum(1 for _, ok in constraints if ok)
    score = passed / total if total > 0 else 0.0
    with open(reward_path, 'w') as f:
        json.dump({'score': score, 'reward': score}, f)
    report = {'constraints': {name: ok for name, ok in constraints}, 'details': details, 'score': score}
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
