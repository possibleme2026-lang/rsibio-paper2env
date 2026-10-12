# Cytoplasmic DDM fBm Parameter Estimation

The agent must implement a faithful simplified DDM analysis from the paper. The environment contains synthetic preprocessed DDM effective decay curves E(q,t) = -log(1 - S(q,t)) for five cytoplasmic samples. The curves follow a fractional Brownian motion scaling model with a known offset:
E(q,t) = D_f * q^(2*alpha) * t^alpha + 1.0

Input path: /app/data/ddm_scaling.csv with columns sample_id,q,t,E.
Output path: /app/output.json. It must be valid JSON with double quotes, for example:
{
  "samples": [
    {"id": "sampleA", "D_f": 1.23, "alpha": 0.45}
  ]
}

For every sample id present in the input, output one object with id, D_f, alpha. D_f is the fractional diffusivity amplitude and alpha is the scaling exponent. Fit using least squares in log space after subtracting the offset, or any equivalent method.

Scoring: hidden gold parameters are used. For each sample, d_err=(D_f_pred-D_f_gold)/D_f_gold and a_err=alpha_pred-alpha_gold. RMSE=sqrt(mean(d_err^2+a_err^2)). Score=max(0,1-RMSE). Missing/invalid output gives 0.

Constraints: do not access network; do not read /tests; only use /app/data. The task is deterministic and self-contained.
