# Review notes

- Synthetic log2 expression matrix substitutes unavailable real PBMC RNA-seq; schema inspired by p53-centered and neutron-quality genes.
- Shortcut probe: checked for sample-order and dose leakage; removed dose column, shuffled train labels, and kept gold test labels only in tests/grade.py.
- Known limitations: small synthetic dataset, simplified binary quality discrimination at 6 h, not a full DEG or RBE estimation pipeline.
