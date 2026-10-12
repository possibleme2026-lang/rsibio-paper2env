"""Deterministic generator of the synthetic multi-task-learning behavioural dataset.

Build-time only: the image runs this file and then deletes it, so the agent sees
only the resulting CSV. The grader keeps an authoritative copy in tests/gold_gen.py.

Run:  python gen_inputs.py   ->  writes behavior.csv next to this file
"""
import numpy as np
import pandas as pd

SEED = 20250922
N_PER_CELL = 36
N_PRACTICE = 21
N_LEARN = 80
N_POSTTEST = 40
WARMUP = 5

ORDERS = ['conceptual_first', 'spatial_first']
SIMILARITIES = ['similar', 'distinct']
REGIMES = ['blocked', 'interleaved']

P_SECOND = {
    ('conceptual_first', 'similar', 'blocked'): 0.82,
    ('conceptual_first', 'distinct', 'blocked'): 0.70,
    ('conceptual_first', 'similar', 'interleaved'): 0.68,
    ('conceptual_first', 'distinct', 'interleaved'): 0.80,
    ('spatial_first', 'similar', 'blocked'): 0.73,
    ('spatial_first', 'distinct', 'blocked'): 0.72,
    ('spatial_first', 'similar', 'interleaved'): 0.72,
    ('spatial_first', 'distinct', 'interleaved'): 0.73,
}

P_FIRST = {
    ('conceptual_first', 'similar', 'blocked'): 0.78,
    ('conceptual_first', 'distinct', 'blocked'): 0.68,
    ('conceptual_first', 'similar', 'interleaved'): 0.70,
    ('conceptual_first', 'distinct', 'interleaved'): 0.76,
    ('spatial_first', 'similar', 'blocked'): 0.71,
    ('spatial_first', 'distinct', 'blocked'): 0.70,
    ('spatial_first', 'similar', 'interleaved'): 0.70,
    ('spatial_first', 'distinct', 'interleaved'): 0.71,
}

P_PRACTICE = 0.78
LEARN_START = 0.60
LEARN_END = 0.80


def _logit(p):
    return float(np.log(p / (1.0 - p)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _events(regime, first_task, second_task):
    events = [('practice', first_task, t) for t in range(1, N_PRACTICE + 1)]
    if regime == 'blocked':
        events += [('learn', first_task, t) for t in range(1, N_LEARN + 1)]
        events += [('learn', second_task, t) for t in range(1, N_LEARN + 1)]
    else:
        chunk = 10
        for c in range(N_LEARN // chunk):
            lo = c * chunk + 1
            hi = lo + chunk
            pair = (first_task, second_task) if c % 2 == 0 else (second_task, first_task)
            for tk in pair:
                events += [('learn', tk, t) for t in range(lo, hi)]
    events += [('posttest', first_task, t) for t in range(1, N_POSTTEST + 1)]
    events += [('posttest', second_task, t) for t in range(1, N_POSTTEST + 1)]
    return events


def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    pid = 0
    for order in ORDERS:
        first_task = 'conceptual' if order == 'conceptual_first' else 'spatial'
        second_task = 'spatial' if first_task == 'conceptual' else 'conceptual'
        for similarity in SIMILARITIES:
            for regime in REGIMES:
                p_first = P_FIRST[(order, similarity, regime)]
                p_second = P_SECOND[(order, similarity, regime)]
                for _ in range(N_PER_CELL):
                    pid += 1
                    ability = float(rng.normal(0.0, 0.45))
                    u = float(rng.random())
                    if u < 0.06:
                        post_mode = 'none'
                    elif u < 0.08:
                        post_mode = 'partial'
                    else:
                        post_mode = 'full'
                    for block, task, t in _events(regime, first_task, second_task):
                        if block == 'posttest':
                            if post_mode == 'none':
                                continue
                            if post_mode == 'partial' and t > 20:
                                continue
                            target = p_first if task == first_task else p_second
                            p = _sigmoid(_logit(target) + ability)
                            if t <= WARMUP:
                                p -= 0.12
                        elif block == 'practice':
                            p = _sigmoid(_logit(P_PRACTICE) + ability)
                            if t <= WARMUP:
                                p -= 0.12
                        else:
                            frac = (t - 1) / (N_LEARN - 1)
                            base = LEARN_START + (LEARN_END - LEARN_START) * frac
                            p = _sigmoid(_logit(base) + ability)
                            if t <= WARMUP:
                                p -= 0.12
                        p = float(np.clip(p, 0.02, 0.985))
                        correct = int(rng.random() < p)
                        if correct:
                            rt = float(np.exp(rng.normal(np.log(720.0), 0.30)))
                        else:
                            rt = float(np.exp(rng.normal(np.log(840.0), 0.32)))
                        draw = float(rng.random())
                        if draw < 0.008:
                            rt = float(rng.uniform(85.0, 149.0))
                            correct = 0
                        elif draw < 0.013:
                            rt = float(rng.uniform(4001.0, 6200.0))
                        stimulus = 'S%d' % int(rng.integers(1, 5))
                        correct_key = 'left' if stimulus in ('S1', 'S2') else 'right'
                        if correct:
                            response = correct_key
                        else:
                            response = 'right' if correct_key == 'left' else 'left'
                        rows.append((
                            pid, order, similarity, regime, block, task, t, stimulus,
                            response, correct, round(rt, 1),
                            'correct' if correct else 'incorrect',
                        ))
    return pd.DataFrame(rows, columns=[
        'participant_id', 'order', 'similarity', 'regime', 'block', 'task',
        'trial_in_block', 'stimulus', 'response', 'correct', 'rt_ms', 'feedback',
    ])


if __name__ == '__main__':
    frame = generate()
    frame.to_csv('behavior.csv', index=False)
    print('wrote behavior.csv with %d rows' % len(frame))
