#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p /app/output
python "$DIR/solve.py"
echo "wrote /app/output/edge_z_scores.tsv and /app/output/prioritized_genes.tsv"
