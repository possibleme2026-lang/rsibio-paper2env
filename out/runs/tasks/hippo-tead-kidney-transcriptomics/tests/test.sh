#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier
python /tests/grade.py
status=$?
if [ ! -f /logs/verifier/reward.json ]; then
  printf '{"score": 0.0, "reward": 0.0}\n' > /logs/verifier/reward.json
fi
echo "grade.py exit status: ${status}"
python3 /tests/_fix_reward.py 2>&1 || python /tests/_fix_reward.py 2>&1 || true
