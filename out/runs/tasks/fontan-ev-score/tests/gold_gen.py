#!/usr/bin/env python3
'''Regenerate the cohort used by the verifier.

The task inputs are produced deterministically at image build time by
/app/data/gen_inputs.py (seed 20260731). This helper re-runs that generator
so reviewers can rebuild the exact same cohort the grader compares against.

Usage: python gold_gen.py [--outdir /tmp/cohort]
'''

import argparse
import subprocess
import sys
from pathlib import Path

GENERATOR = Path('/app/data/gen_inputs.py')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='/tmp/cohort')
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if not GENERATOR.exists():
        sys.exit('generator not found at %s' % GENERATOR)
    subprocess.check_call([sys.executable, str(GENERATOR), '--outdir', str(out)])
    for name in ('clinical.csv', 'mirna_counts.csv'):
        p = out / name
        print('%s: %d bytes' % (p, p.stat().st_size))


if __name__ == '__main__':
    main()
