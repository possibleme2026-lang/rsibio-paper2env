import numpy as np
import pandas as pd

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _random_seq(rng, length, gc):
    gc = float(np.clip(gc, 0.0, 1.0))
    out = []
    for _ in range(length):
        if rng.random() < gc:
            out.append(str(rng.choice(['G', 'C'])))
        else:
            out.append(str(rng.choice(['A', 'T'])))
    return ''.join(out)

def _gc_fraction(seq):
    if not seq:
        return 0.0
    return sum(1 for b in seq.upper() if b in 'GC') / float(len(seq))

def generate(n, feat_seed, noise_seed, cohort):
    rng = np.random.default_rng(feat_seed)
    rng_noise = np.random.default_rng(noise_seed) if noise_seed is not None else None
    rows = []
    for i in range(n):
        if cohort == 'L1':
            rtt_gc_target = float(rng.beta(2.5, 2.5))
            spacer_gc_target = float(rng.beta(3.0, 2.0))
            pbs_gc_target = float(rng.beta(2.0, 2.5))
            edit_type = str(rng.choice(['sub', 'ins', 'del'], p=[0.60, 0.20, 0.20]))
            pam_type = str(rng.choice(['NGG', 'NAG', 'NGA'], p=[0.75, 0.18, 0.07]))
            cell_type = 'HEK293T'
        else:
            rtt_gc_target = float(rng.beta(2.0, 3.0))
            spacer_gc_target = float(rng.beta(2.5, 2.5))
            pbs_gc_target = float(rng.beta(2.5, 2.5))
            edit_type = str(rng.choice(['sub', 'ins', 'del'], p=[0.50, 0.25, 0.25]))
            pam_type = str(rng.choice(['NGG', 'NAG', 'NGA'], p=[0.85, 0.10, 0.05]))
            cell_type = 'K562'
        if edit_type == 'del':
            edit_size = int(rng.integers(1, 9))
        else:
            edit_size = int(rng.integers(1, 4))
        nick_offset = float(rng.integers(-3, 4))
        spacer_seq = _random_seq(rng, 20, spacer_gc_target)
        pbs_seq = _random_seq(rng, 12, pbs_gc_target)
        rtt_len = 10 + (edit_size if edit_type == 'del' else 0)
        rtt_len = min(rtt_len, 24)
        rtt_seq = _random_seq(rng, rtt_len, rtt_gc_target)
        target_seq = _random_seq(rng, 30, 0.5)
        spacer_gc = _gc_fraction(spacer_seq)
        pbs_gc = _gc_fraction(pbs_seq)
        rtt_gc = _gc_fraction(rtt_seq)
        rec = {
            'peg_id': f'{cohort}_{i:05d}',
            'cell_type': cell_type,
            'spacer_seq': spacer_seq,
            'pbs_seq': pbs_seq,
            'rtt_seq': rtt_seq,
            'target_seq': target_seq,
            'edit_type': edit_type,
            'edit_size': edit_size,
            'pam_type': pam_type,
            'nick_offset': nick_offset,
            'rtt_len': rtt_len,
        }
        is_del = 1.0 if edit_type == 'del' else 0.0
        is_sub = 1.0 if edit_type == 'sub' else 0.0
        pam_ngg = 1.0 if pam_type == 'NGG' else 0.0
        nick = abs(nick_offset) / 3.0
        z_edit = (-0.8
                  - 3.0 * (rtt_gc - 0.5) ** 2
                  - 0.30 * edit_size
                  - 0.60 * is_del * edit_size
                  + 0.45 * is_sub
                  + 0.60 * spacer_gc
                  + 0.25 * pbs_gc
                  - 0.25 * nick
                  + 0.50 * pam_ngg)
        p_edited = _sigmoid(z_edit)
        z_indel = (-2.0
                   + 2.5 * rtt_gc
                   + 0.15 * edit_size
                   + 0.40 * is_del * edit_size
                   + 0.30 * (1.0 - spacer_gc)
                   - 0.30 * pam_ngg)
        p_indel = _sigmoid(z_indel) * 0.30
        p_unedited = 1.0 - p_edited - p_indel
        p_unedited = float(np.clip(p_unedited, 0.02, 0.98))
        p_vec = np.array([p_edited, p_unedited, p_indel], dtype=float)
        p_vec = np.clip(p_vec, 1e-4, None)
        p_vec = p_vec / p_vec.sum()
        alpha0 = float(np.exp(3.0 + 1.0 * p_edited - 1.2 * rtt_gc - 0.25 * is_del * edit_size + 0.30 * is_sub))
        alpha0 = float(np.clip(alpha0, 8.0, 300.0))
        if rng_noise is not None:
            y = rng_noise.dirichlet(alpha0 * p_vec)
            total = int(np.clip(int(round(np.exp(rng_noise.normal(np.log(1500.0), 0.40)))), 200, 20000))
            counts = rng_noise.multinomial(total, y)
            rec['total_reads'] = total
            rec['n_edited'] = int(counts[0])
            rec['n_unedited'] = int(counts[1])
            rec['n_indel'] = int(counts[2])
            rec['true_edited'] = float(y[0])
            rec['true_unedited'] = float(y[1])
            rec['true_indel'] = float(y[2])
        else:
            rec['total_reads'] = -1
            rec['n_edited'] = -1
            rec['n_unedited'] = -1
            rec['n_indel'] = -1
        rows.append(rec)
    return pd.DataFrame(rows)

def get_gold():
    df = generate(500, 20260102, 987654321, 'L2')
    return df[['peg_id', 'true_edited', 'true_unedited', 'true_indel']].copy()
