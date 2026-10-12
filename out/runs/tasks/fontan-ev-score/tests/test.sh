#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier
python3 /tests/grade.py 2>&1 | tee /logs/verifier/stdout.log
python3 /tests/_fix_reward.py 2>&1 || python /tests/_fix_reward.py 2>&1 || true
