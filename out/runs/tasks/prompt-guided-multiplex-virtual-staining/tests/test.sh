#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier
echo '{"score": 0.0, "reward": 0.0}' > /logs/verifier/reward.json
python /tests/grade.py
python3 /tests/_fix_reward.py 2>&1 || python /tests/_fix_reward.py 2>&1 || true
