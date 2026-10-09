"""Shared test factories. Kept tiny and dependency-free."""

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


def good_blueprint(**over) -> TaskBlueprint:
    base = dict(
        task_name="bioforge/chipseq-peak-calling",
        version="0.1.0",
        instruction_md=(
            "# Task\n\nImplement a peak caller and write F1 to "
            "/app/rewards.json as {\"score\": <f1>}.\n"
        ),
        dockerfile=(
            "FROM python:3.11-slim\n"
            "WORKDIR /app\n"
            "COPY data/ /app/data/\n"
            "RUN pip install numpy scikit-learn\n"
        ),
        test_files={
            "tests/test.sh": "#!/bin/bash\nset -euo pipefail\npython tests/grade.py\n",
            "tests/grade.py": "import json; json.load(open('/app/rewards.json'))\n",
        },
        solution_files={"solution/solve.sh": "#!/bin/bash\npython solve.py\n"},
        env_files={"environment/data/README.txt": "synthetic inputs\n"},
        task_toml={
            "schema_version": "1.4",
            "task": {"name": "bioforge/chipseq-peak-calling", "version": "0.1.0"},
            "agent": {"timeout_sec": 600.0},
            "verifier": {"timeout_sec": 300.0},
            "environment": {"cpus": 2, "memory_mb": 4096, "gpus": 0},
        },
        provenance=good_provenance(),
        category="computational-biology",
        difficulty="medium",
        keywords=["chipseq", "peak-calling"],
        metric="f1",
    )
    base.update(over)
    return TaskBlueprint(**base)
