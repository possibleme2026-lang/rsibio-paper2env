# Review notes

- Synthetic reads and probes are generated deterministically (seed 42) because real 10x Flex data is not publicly redistributable. The generator simulates realistic error rates, reverse-complement orientations, ambiguous probes, and cell barcode mutations.
- Shortcut probe: The gold standard is not present in the environment. The grader regenerates pristine inputs from tests/gold_gen.py and computes the reference solution internally, so the agent cannot read the expected output or tamper with the input data.
- Tool chain: numpy is installed for efficient vectorized Hamming distance; the task is algorithmic and does not rely on a specific external binary. The agent can use any language.
- Known limitations: The task simplifies the full STAR Suite Flex module by using synthetic reads and omitting transcript alignment; however, it preserves the core computational challenges: probe assignment with Hamming distance, reverse-complement handling, cell barcode correction, and UMI deduplication.
