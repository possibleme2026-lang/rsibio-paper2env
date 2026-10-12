# Review notes

- Data are synthetic-but-realistic, generated deterministically from a protected seed. True protein labels are hidden from the agent and revealed only in tests/gold_gen.py.
- Shortcut probe considered: agent could try to read the Dockerfile or generator; Dockerfile is not copied into the runtime image and the data generator is not present in /app/data. The hidden labels are only in the verifier phase.
- Tool chain: Python 3.11 with numpy, pandas, scipy (paired t-test, hypergeometric), scikit-learn (logistic regression, AUROC).
- Known limitations: the enrichment pathway annotations are synthetic; the exact p-values depend on the specified universe and hypergeometric test. The classifier predictions are evaluated by AUROC, allowing implementation flexibility.
