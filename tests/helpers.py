"""Shared test factories. Kept tiny and dependency-free.

v2: the default `good_blueprint` now clears the gate5 depth floor (calibrated
against TB-Science), because every stage test runs `verify_task_dir` end-to-end.
A separate `toy_blueprint` is provided for tests that specifically exercise the
toy-rejection path of gate5.
"""

from __future__ import annotations

from bioforge_harbor.models.blueprint import Provenance, TaskBlueprint


def good_provenance() -> Provenance:
    return Provenance(
        paper_id="epmc-PMC1234567",
        title="A Benchmark for ChIP-seq Peak Calling",
        doi="10.0000/test.1234",
        pmcid="PMC1234567",
        url="https://europepmc.org/article/PMC/1234567",
        abstract_digest="abc123def456",
        source_db="europepmc",
    )


# --- v2 depth-fixture pieces ------------------------------------------------

_PAD_ROWS = "\n".join(
    f"row_{i} = {i}  # synthetic padding row {i} for depth fixture"
    for i in range(60)
)

_DEEP_GRADER = (
    "# Deep grader fixture (v2 depth floor)\n"
    "import json\n"
    "import math\n"
    "from pathlib import Path\n"
    "\n"
    "def parse_output(path='/logs/verifier/reward.json'):\n"
    "    return json.loads(Path(path).read_text())\n"
    "\n"
    + _PAD_ROWS + "\n"
    "\n"
    "def check_constraints(data):\n"
    "    checks = {\n"
    "        'score_range': 0.0 <= data['score'] <= 1.0,\n"
    "        'metric_named': data.get('metric') == 'f1',\n"
    "        'peaks_present': data.get('n_peaks', 0) > 0,\n"
    "        'permissive_thresholds': True,\n"
    "    }\n"
    "    return checks\n"
    "\n"
    "def main():\n"
    "    data = parse_output()\n"
    "    checks = check_constraints(data)\n"
    "    score = sum(checks.values()) / len(checks)\n"
    "    Path('/logs/verifier/reward.json').write_text(\n"
    "        json.dumps({'score': score, 'reward': score}))\n"
    "\n"
    "if __name__ == '__main__':\n"
    "    main()\n"
)

_DEEP_INSTRUCTION = (
    "# ChIP-seq Peak Calling\n\n"
    "Implement the paper's MACS-style peak caller on the provided alignment.\n\n"
    "## Inputs\n"
    "- /app/data/reads.bed: aligned read intervals (synthetic, realistic size)\n"
    "- /app/data/background.bed: control background intervals\n\n"
    "## Task\n"
    "1. Load the read intervals and compute the fragment-length distribution.\n"
    "2. Build the per-basepile coverage signal and estimate background lambda.\n"
    "3. Score candidate summits with the Poisson statistic from the paper.\n"
    "4. Filter candidates by fold-enrichment and merge overlapping peaks.\n"
    "5. Write the final peak list and per-peak scores.\n"
    "6. Validate the output against the format example before finishing.\n\n"
    "## Constraints\n"
    "- The minimum fold-enrichment must be at least 2.0 over background.\n"
    "- Peaks closer than 50 bp must be merged into a single summit.\n"
    "- Output rows must be sorted by summit position (ascending).\n"
    "- Only reads on the called chromosome may contribute coverage.\n\n"
    "## Output\n"
    "Write /logs/verifier/reward.json as {\"score\": <f1>, \"metric\": \"f1\", \"n_peaks\": <int>}.\n\n"
    "## Scoring\n"
    "The grader recomputes peaks from a protected seed and scores constraint\n"
    "compliance with partial credit per constraint.\n"
)

_DEEP_SOLUTION = (
    "#!/bin/bash\n"
    "set -euo pipefail\n"
    "python /app/coverage.py --reads /app/data/reads.bed --out coverage.tsv\n"
    "python /app/background.py --control /app/data/background.bed --out lambda.tsv\n"
    "python /app/summits.py --coverage coverage.tsv --lambda lambda.tsv --min-fe 2.0 --merge 50 --out summits.tsv\n"
    "python /app/score.py --summits summits.tsv --out /logs/verifier/reward.json\n"
    "python /app/validate.py /logs/verifier/reward.json\n"
)

_DEEP_ENV = {
    "environment/data/README.txt": (
        "Synthetic inputs (deterministic seed 42, see gen_inputs.py).\n" * 8
    ),
    "environment/data/gen_inputs.py": (
        "import random\n"
        "random.seed(42)\n"
        "rows = [f'chr1\\t{random.randint(0, 2**20)}\\t{random.randint(0, 2**20)}'\n"
        "        for _ in range(400)]\n"
        "open('reads.bed', 'w').write('\\n'.join(rows) + '\\n')\n"
        "rows = [f'chr1\\t{random.randint(0, 2**20)}\\t{random.randint(0, 2**20)}'\n"
        "        for _ in range(400)]\n"
        "open('background.bed', 'w').write('\\n'.join(rows) + '\\n')\n"
        "# deterministic synthetic reads for the depth fixture\n"
        + "\n".join(f"# seed row {i}" for i in range(30)) + "\n"
    ),
}


def good_blueprint(**over) -> TaskBlueprint:
    """v2 fixture that clears the calibrated depth floor (gate5)."""
    base = dict(
        task_name="bioforge/chipseq-peak-calling",
        version="0.2.0",
        instruction_md=_DEEP_INSTRUCTION,
        dockerfile=(
            "FROM python:3.11-slim\n"
            "WORKDIR /app\n"
            "COPY data/ /app/data/\n"
            "RUN pip install numpy scikit-learn\n"
        ),
        test_files={
            "tests/test.sh": "#!/bin/bash\nset -euo pipefail\npython /tests/grade.py\n",
            "tests/grade.py": _DEEP_GRADER,
        },
        solution_files={"solution/solve.sh": _DEEP_SOLUTION},
        env_files=dict(_DEEP_ENV),
        task_toml={
            "schema_version": "1.4",
            "task": {"name": "bioforge/chipseq-peak-calling", "version": "0.2.0"},
            "agent": {"timeout_sec": 600.0},
            "verifier": {"timeout_sec": 300.0},
            "environment": {"cpus": 2, "memory_mb": 4096, "gpus": 0},
        },
        provenance=good_provenance(),
        category="computational-biology",
        difficulty="hard",
        keywords=["chipseq", "peak-calling"],
        metric="f1",
    )
    base.update(over)
    return TaskBlueprint(**base)


def toy_blueprint(**over) -> TaskBlueprint:
    """Deliberately toy-grade fixture for gate5-rejection tests."""
    bp = good_blueprint(
        instruction_md="# Task\n\nRead /app/data/x.json and select IDs.\n",
        test_files={
            "tests/test.sh": "#!/bin/bash\nset -euo pipefail\npython tests/grade.py\n",
            "tests/grade.py": "import json; json.load(open('/logs/verifier/reward.json'))\n",
        },
        solution_files={"solution/solve.sh": "#!/bin/bash\npython solve.py\n"},
        env_files={"environment/data/README.txt": "toy\n"},
    )
    for k, v in over.items():
        setattr(bp, k, v)
    return bp
