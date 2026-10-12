import numpy as np
import pandas as pd

np.random.seed(42)
n_patients = 200
n_biomarkers = 12
t_max = 12.0
dt_sim = 0.01
n_steps = int(t_max / dt_sim) + 1
time_sim = np.linspace(0, t_max, n_steps)

def U(x):
    return 4*(x**2 - 1)**2 + 0.8*x

def dU(x):
    return 16*x*(x**2 - 1) + 0.8

D = 0.5

np.random.seed(123)
a = np.random.uniform(-1, 1, n_biomarkers)
b = np.random.uniform(-1, 1, n_biomarkers)
c = np.random.uniform(-1, 1, n_biomarkers)

patient_data = []
for pid in range(n_patients):
    x = np.zeros(n_steps)
    x[0] = np.random.normal(0, 0.5)
    for i in range(1, n_steps):
        x[i] = x[i-1] - dU(x[i-1])*dt_sim + np.sqrt(2*D*dt_sim)*np.random.normal()
        x[i] = np.clip(x[i], -2.5, 2.5)
    n_visits = np.random.randint(5, 16)
    inter_times = np.random.exponential(scale=1.5, size=n_visits-1)
    visit_times = np.concatenate(([0], np.cumsum(inter_times)))
    if visit_times[-1] > 12:
        visit_times = visit_times / visit_times[-1] * 12
    visit_times = np.sort(visit_times)
    visit_times = np.clip(visit_times, 0, 12)
    x_visits = np.interp(visit_times, time_sim, x)
    for i, t in enumerate(visit_times):
        xv = x_visits[i]
        row = {'patient_id': pid, 'visit_time': t}
        for j in range(n_biomarkers):
            val = a[j] * np.sin(2*xv) + b[j] * xv**2 + c[j] * xv + np.random.normal(0, 0.3)
            val = val * 2 + 10
            row[f'biomarker_{j+1}'] = val
        row['diagnosis'] = 1 if xv > 0.2 else 0
        patient_data.append(row)

df = pd.DataFrame(patient_data)
df.to_csv('/app/data/longitudinal_biomarkers.csv', index=False)
print(f"Generated {len(df)} rows for {n_patients} patients.")
