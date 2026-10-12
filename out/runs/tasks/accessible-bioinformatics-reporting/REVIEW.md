# Review notes

- Synthetic data substitution: the paper's real LLM outputs and interdisciplinary reports are not redistributed; a schema-compatible synthetic analysis-output and candidate-statement dataset is provided.
- Shortcut probe: hidden gold is only in tests/grade.py; selecting all statements is penalized by F1 false positives, and the agent cannot read verifier files during solve.
- Known limitations: this is a deterministic simplified proxy for LLM report generation and comprehension; it does not require or evaluate a live LLM and uses rule-checkable statement predicates.
