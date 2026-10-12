# Review notes

- Synthetic data is generated deterministically from a fixed seed (42) by environment/data/gen_inputs.py. The hidden gold is computed by tests/gold_gen.py using a reference implementation of the same pipeline, not readable by the agent.
- Shortcut probe: agent cannot read tests/gold_gen.py during task phase; gold is regenerated from the fixed inputs. The task requires implementing the full pipeline, not just loading a gold file.
- Tool chain: numpy, pandas, scipy, scikit-learn, POT (Python Optimal Transport) are installed. POT provides the sinkhorn solver for entropy-regularized OT.
- Known limitations: the synthetic dataset is a simplified 2D latent simulation with 24 genes; it does not capture full biological complexity but exercises the core computational steps of PhyloFM.
