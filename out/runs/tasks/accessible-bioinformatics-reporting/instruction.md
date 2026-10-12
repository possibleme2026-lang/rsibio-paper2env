# Accessible Bioinformatics Reporting

You are a report-generation assistant for an interdisciplinary bioinformatics team. Given standardized analysis outputs and candidate report statements, produce a short accessible summary and select the statement IDs that are supported by the data.

Inputs:
- /app/environment/data/analysis_outputs.json: study description, differential expression, enrichment, and figure captions.
- /app/environment/data/candidate_statements.json: each statement has an id, text, and a machine-checkable predicate under check.

Output:
Write /app/output.json as a JSON object with:
- "summary": a non-empty accessible summary string.
- "selected_statement_ids": a list of statement IDs from candidate_statements.json.

Scoring:
A hidden gold set of supported statement IDs is compared to your selected_statement_ids using F1. If the summary is empty, the score is 0.

Constraints:
No internet access. Do not read hidden gold. Solve deterministically from the provided files. Use only IDs that appear in candidate_statements.json.
