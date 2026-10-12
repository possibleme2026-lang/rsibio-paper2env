#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier /tmp
python /tests/gold_gen.py
python /tests/grade.py
python3 /tests/_fix_reward.py 2>&1 || python /tests/_fix_reward.py 2>&1 || true
