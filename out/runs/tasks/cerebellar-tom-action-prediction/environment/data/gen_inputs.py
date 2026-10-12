import numpy as np
import pandas as pd

np.random.seed(42)

n = 80
subject_ids = [f"sub-{i:03d}" for i in range(1, n+1)]

age = np.random.normal(48, 6, n).clip(36, 60)
sex = np.random.choice(['M', 'F'], n)
site = np.random.choice([1, 2, 3], n)

tiv = np.random.normal(1_200_000, 80_000, n).clip(1_000_000, 1_500_000)
tiv_z = (tiv - tiv.mean()) / tiv.std()

regions = [
    "Left_CrusI", "Right_CrusI", "Left_CrusII", "Right_CrusII",
    "Left_VIIB", "Right_VIIB", "Vermis_VI", "Vermis_VII",
    "Vermis_VIII", "Left_VIIIa", "Right_VIIIa", "Left_VIIIb",
    "Right_VIIIb", "Left_IX", "Right_IX", "Left_X", "Right_X",
    "Left_I_IV", "Right_I_IV", "Left_V", "Right_V", "Left_VI",
    "Right_VI", "Vermis_IX"
]
n_regions = len(regions)

means = np.random.uniform(3000, 8000, n_regions)

A = np.random.normal(0, 1, (n_regions, n_regions))
cov = A @ A.T
cov = cov / np.max(np.diag(cov)) * 500**2

beta_tiv = np.random.normal(0.6, 0.1, n_regions)
resid = np.random.multivariate_normal(np.zeros(n_regions), cov, n)

gmv_raw = means + np.outer(tiv_z, beta_tiv) * 800 + resid
gmv_raw = np.maximum(gmv_raw, 1000)

gmv_adj = gmv_raw - np.outer(tiv_z, beta_tiv) * 800
gmv_adj_z = (gmv_adj - gmv_adj.mean(axis=0)) / gmv_adj.std(axis=0)

idx = {name: i for i, name in enumerate(regions)}

age_z = (age - age.mean()) / age.std()
sex_effect = np.where(sex == 'M', 0.5, -0.5)
site_effect = np.select([site==1, site==2, site==3], [0.2, -0.1, -0.1])

sex_dummies = pd.get_dummies(sex, drop_first=True).values
site_dummies = pd.get_dummies(site, drop_first=True).values
covs = np.column_stack([age_z, sex_dummies, site_dummies, tiv_z])

tom_true_idx = [idx["Left_CrusII"], idx["Right_CrusII"]]
tom_true_effect = (2.0 * gmv_adj_z[:, tom_true_idx[0]] +
                   1.8 * gmv_adj_z[:, tom_true_idx[1]])
tom_cov_effect = 1.2 * age_z + 0.5 * sex_effect + 0.3 * site_effect
tom_noise = np.random.normal(0, 1.5, n)
other_regions_tom = np.delete(gmv_adj_z, tom_true_idx, axis=1)
X_null_tom = np.column_stack([np.ones(n), covs, other_regions_tom])
beta_tom, _, _, _ = np.linalg.lstsq(X_null_tom, tom_noise, rcond=None)
tom_resid_noise = tom_noise - X_null_tom @ beta_tom
tom_score = 10 + tom_cov_effect + tom_true_effect + tom_resid_noise
tom_score = (tom_score - tom_score.min()) / (tom_score.max() - tom_score.min()) * 20

action_true_idx = [idx["Left_VIIB"], idx["Right_VIIB"], idx["Vermis_VI"], idx["Vermis_VII"]]
action_true_effect = (1.6 * gmv_adj_z[:, action_true_idx[0]] +
                      1.5 * gmv_adj_z[:, action_true_idx[1]] +
                      1.2 * gmv_adj_z[:, action_true_idx[2]] +
                      1.0 * gmv_adj_z[:, action_true_idx[3]])
action_cov_effect = 0.8 * age_z + 0.3 * sex_effect + 0.2 * site_effect
action_noise = np.random.normal(0, 0.8, n)
other_regions_action = np.delete(gmv_adj_z, action_true_idx, axis=1)
X_null_action = np.column_stack([np.ones(n), covs, other_regions_action])
beta_action, _, _, _ = np.linalg.lstsq(X_null_action, action_noise, rcond=None)
action_resid_noise = action_noise - X_null_action @ beta_action
action_score = 0.5 + action_cov_effect + action_true_effect + action_resid_noise
action_score = (action_score - action_score.min()) / (action_score.max() - action_score.min())

default = 0.6 * (tom_true_effect / tom_true_effect.std()) + np.random.normal(0, 0.8, n)
ventral = 0.6 * (action_true_effect / action_true_effect.std()) + np.random.normal(0, 0.8, n)
visual = np.random.normal(0, 1, n)
somatomotor = np.random.normal(0, 1, n)
dorsal = np.random.normal(0, 1, n)
limbic = np.random.normal(0, 1, n)
frontoparietal = np.random.normal(0, 1, n)

def zscore(x):
    return (x - x.mean()) / x.std()

networks = pd.DataFrame({
    'subject_id': subject_ids,
    'Visual': zscore(visual),
    'Somatomotor': zscore(somatomotor),
    'DorsalAttention': zscore(dorsal),
    'VentralAttention': zscore(ventral),
    'Limbic': zscore(limbic),
    'Frontoparietal': zscore(frontoparietal),
    'DefaultMode': zscore(default)
})

gmv_df = pd.DataFrame(gmv_raw, columns=regions)
gmv_df.insert(0, 'subject_id', subject_ids)
behavior_df = pd.DataFrame({
    'subject_id': subject_ids,
    'age_months': age,
    'sex': sex,
    'site': site,
    'explicit_tom_score': tom_score,
    'action_prediction_accuracy': action_score,
    'tiv_mm3': tiv
})
labels_df = pd.DataFrame({
    'region_id': range(1, n_regions+1),
    'region_name': regions,
    'lobule': ['Crus I']*2 + ['Crus II']*2 + ['VIIB']*2 + ['Vermis']*4 + ['VIIIa']*2 + ['VIIIb']*2 + ['IX']*2 + ['X']*2 + ['I-IV']*2 + ['V']*2 + ['VI']*2,
    'hemisphere': ['L','R']*12,
    'prior_network': ['ToM']*4 + ['ActionObservation']*6 + ['Other']*14
})

gmv_df.to_csv('/app/data/cerebellar_gmv.csv', index=False)
behavior_df.to_csv('/app/data/behavior.csv', index=False)
networks.to_csv('/app/data/cortical_network_scores.csv', index=False)
labels_df.to_csv('/app/data/atlas_labels.csv', index=False)
