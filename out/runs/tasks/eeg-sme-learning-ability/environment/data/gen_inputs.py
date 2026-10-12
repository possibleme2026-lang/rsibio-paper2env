import numpy as np
import pandas as pd

np.random.seed(42)

N_subjects = 80
N_lists = 30
list_length = 20

ability = np.random.normal(0, 1, N_subjects)
intercept = -0.5 + 1.0 * ability

sp = np.arange(1, list_length + 1)
sp_effect = 1.2 * np.exp(-(sp - 1) / 3) + 0.8 * np.exp(-(list_length - sp) / 2)

theta_base = 0.4
alpha_base = -0.5
scale = np.exp(-0.5 * ability)
theta_slope = theta_base * scale + np.random.normal(0, 0.1, N_subjects)
alpha_slope = alpha_base * scale + np.random.normal(0, 0.1, N_subjects)
gamma_slope = np.zeros(N_subjects)

words = [''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 5)) for _ in range(2000)]

rows = []
for subj in range(N_subjects):
    for list_id in range(1, N_lists + 1):
        for pos in range(1, list_length + 1):
            theta = np.random.normal(0, 1)
            alpha = np.random.normal(0, 1)
            gamma = np.random.normal(0, 1)
            logit = (intercept[subj] + sp_effect[pos - 1] +
                     theta_slope[subj] * theta +
                     alpha_slope[subj] * alpha +
                     gamma_slope[subj] * gamma)
            p = 1 / (1 + np.exp(-logit))
            recalled = np.random.binomial(1, p)
            word = words[np.random.randint(0, len(words))]
            rows.append([subj + 1, list_id, pos, word, theta, alpha, gamma, recalled])

df = pd.DataFrame(rows, columns=['subject_id', 'list_id', 'serial_position', 'word',
                                 'theta_power', 'alpha_power', 'gamma_power', 'recalled'])
df.to_csv('/app/data/encoding_events.csv', index=False)
print(f'Generated {len(df)} rows to /app/data/encoding_events.csv')
