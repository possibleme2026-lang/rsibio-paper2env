#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
python /tests/grade.py > /logs/verifier/reward.json
