import json, os, subprocess, sys, tempfile, shutil
sys.path.insert(0, os.path.dirname(__file__))
import gold_gen

def main():
    work = tempfile.mkdtemp(prefix='hidden_')
    try:
        hidden_root = os.path.join(work, 'data')
        os.makedirs(hidden_root, exist_ok=True)
        gold_gen.generate_data(2024, hidden_root)
        gold = gold_gen.compute_gold(hidden_root)
        out_json = os.path.join(work, 'out.json')
        script = '/app/solution/evaluate.py'
        if not os.path.exists(script):
            print('ERROR: /app/solution/evaluate.py not found')
            reward = {'score': 0.0, 'reward': 0.0}
            os.makedirs('/logs/verifier', exist_ok=True)
            json.dump(reward, open('/logs/verifier/reward.json', 'w'))
            return
        try:
            subprocess.run(['python', script, '--input', hidden_root, '--output', out_json],
                           check=True, timeout=300)
        except Exception as e:
            print('ERROR running solution:', e)
            reward = {'score': 0.0, 'reward': 0.0}
            os.makedirs('/logs/verifier', exist_ok=True)
            json.dump(reward, open('/logs/verifier/reward.json', 'w'))
            return
        pred = json.load(open(out_json))
        constraints = []
        def add(name, passed, detail=''):
            constraints.append((name, passed, detail))
        add('output_file_exists', True)
        ok = True
        for task in ['task1', 'task2']:
            if task not in pred:
                ok = False
            else:
                for m in ['mean', 'linear', 'mlp', 'perturbnet']:
                    if m not in pred[task].get('methods', {}):
                        ok = False
        add('methods_present', ok)
        ok = True
        for task in ['task1', 'task2']:
            for m in pred.get(task, {}).get('methods', {}):
                vals = pred[task]['methods'][m]
                if not (0 <= vals.get('mse', -1)): ok = False
                if not (-1 <= vals.get('delta_pearson', 2) <= 1): ok = False
                if not (0 <= vals.get('de_f1', -1) <= 1): ok = False
                if not (0 < vals.get('dist_sim', -1) <= 1): ok = False
        add('numeric_ranges', ok)
        tol = 1e-3
        for task in ['task1', 'task2']:
            for metric in ['mse', 'delta_pearson', 'de_f1', 'dist_sim']:
                ok = True
                detail = ''
                for m in ['mean', 'linear', 'mlp', 'perturbnet']:
                    g = gold[task]['methods'][m][metric]
                    p = pred[task]['methods'][m][metric]
                    if abs(g - p) > tol:
                        ok = False
                        detail += f'{m}: gold={g:.6f} pred={p:.6f}; '
                add(f'{task}_{metric}', ok, detail)
        for task in ['task1', 'task2']:
            ok = True
            detail = ''
            for metric in ['mse', 'delta_pearson', 'de_f1', 'dist_sim']:
                g = gold[task]['ranking'][metric]
                p = pred[task]['ranking'][metric]
                if g != p:
                    ok = False
                    detail += f'{metric}: gold={g} pred={p}; '
            add(f'{task}_ranking', ok, detail)
        for task in ['task1', 'task2']:
            ok = True
            detail = ''
            for m in ['mean', 'linear', 'mlp', 'perturbnet']:
                g = gold[task]['overall_score'][m]
                p = pred[task]['overall_score'][m]
                if abs(g - p) > tol:
                    ok = False
                    detail += f'{m}: gold={g:.6f} pred={p:.6f}; '
            gb = gold[task]['best_method']
            pb = pred[task]['best_method']
            if gb != pb:
                ok = False
                detail += f'best: gold={gb} pred={pb}; '
            add(f'{task}_overall_best', ok, detail)
        total = len(constraints)
        passed = sum(1 for _, p, _ in constraints if p)
        score = passed / total if total > 0 else 0.0
        os.makedirs('/logs/verifier', exist_ok=True)
        reward = {'score': score, 'reward': score}
        json.dump(reward, open('/logs/verifier/reward.json', 'w'))
        report = {'constraints': [{'name': n, 'passed': p, 'detail': d} for n, p, d in constraints], 'score': score}
        json.dump(report, open('/logs/verifier/report.json', 'w'))
        print(f'Score: {score:.4f} ({passed}/{total})')
        for n, p, d in constraints:
            status = '+' if p else '-'
            print(f'  {n}: {status} {d}')
    finally:
        shutil.rmtree(work, ignore_errors=True)

if __name__ == '__main__':
    main()
