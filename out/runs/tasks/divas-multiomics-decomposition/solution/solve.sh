#!/bin/bash
set -euo pipefail
mkdir -p /app/results
Rscript /solution/solve.R
