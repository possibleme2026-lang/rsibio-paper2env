#!/bin/bash
set -uo pipefail
export PYTHONUNBUFFERED=1
python /tests/grade.py
python3 /tests/_fix_reward.py 2>&1 || python /tests/_fix_reward.py 2>&1 || true
