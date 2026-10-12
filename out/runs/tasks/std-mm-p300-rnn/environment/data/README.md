# STD-RNN multiregional model inputs

- config.json: simulation parameters.
- W_S.npy, W_F.npy, W_SF.npy, W_FS.npy: 100x100 weight matrices.
- selectivity.npy: 100x2 sensory input patterns.
- sequences.json: 300 trials (100 baseline, 100 noise, 100 attention).

Generated deterministically by gen_inputs.py.
