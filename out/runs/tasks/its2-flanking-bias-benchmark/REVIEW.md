# Review notes

- Data: synthetic but realistic ITS sequences (840 bp, 3,000 reference records, 1,200 queries) generated deterministically by environment/data/gen_inputs.py; flanking regions carry family signal, ITS2 is genus-specific.
- Shortcut probe: gold metrics are not stored in the environment; the grader recomputes metrics from the agent's own prediction tables and checks scientific thresholds, so hardcoding summary values fails.
- Tool chain: vsearch for best-hit and SINTAX, scikit-learn for HiTaC-style k-mer logistic regression and pretrained MLP classifiers.
- Limitations: the pretrained models are stand-ins (MLPs on flank k-mers) trained to exhibit the flanking-region bias; exact accuracy percentages differ from the paper but the qualitative collapse is reproduced.
