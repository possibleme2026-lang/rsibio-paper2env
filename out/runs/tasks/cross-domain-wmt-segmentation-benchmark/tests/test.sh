#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier
cd /app
python /tests/grade.py
echo 'grader finished'
python3 /tests/_fix_reward.py 2>&1 || python /tests/_fix_reward.py 2>&1 || true
