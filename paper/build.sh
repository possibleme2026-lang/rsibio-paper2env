#!/usr/bin/env bash
# Build the manuscript.
#
#   ./build.sh              # compile to build/main.pdf
#   TECTONIC=tectonic ./build.sh
#
# Tectonic fetches LaTeX packages on demand, so the first run needs network
# access (set https_proxy if you are behind one). Later runs are offline.
set -euo pipefail

cd "$(dirname "$0")"
TECTONIC="${TECTONIC:-$(cd .. && pwd)/.tools/tectonic.exe}"
if [ ! -x "$TECTONIC" ]; then
  TECTONIC="${TECTONIC_ALT:-tectonic}"
fi

mkdir -p build

# Appendix B is generated from claims.json; refresh it so the PDF can never
# be built from a stale ledger.
python ../tools/gen_claims_tex.py

"$TECTONIC" -X compile main.tex --outdir build
echo "built: $(pwd)/build/main.pdf"
