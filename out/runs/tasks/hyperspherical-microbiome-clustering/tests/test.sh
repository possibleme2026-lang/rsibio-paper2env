#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier
python /tests/gold_gen.py --seed 20260728 --out /tmp/gold.npz
python /tests/grade.py
python3 /tests/_fix_reward.py 2>&1 || python /tests/_fix_reward.py 2>&1 || true
