Synthetic low-divergence local-ancestry benchmark.

Regenerate (deterministic):
  python gen_inputs.py --out . --seed 20260730

Layout
  map.tsv                       snp_id, pos_bp, cM  (3000 SNPs over 150 cM)
  <level>/ref_A.haplotypes.txt  60 haplotypes x 3000 sites, characters 0/1
  <level>/ref_B.haplotypes.txt  60 haplotypes x 3000 sites, characters 0/1
  <level>/admixed_<sample>.txt  admixed haplotype, 3000 characters 0/1
  <level>/truth_<sample>.txt    simulated ancestry, 3000 characters A/B
  simulation_manifest.json      seed, realised FST per level, true switch rate

levels : fst_0002 (FST 0.002), fst_0020 (0.020), fst_0150 (0.150)
samples: hap1 hap2 hap3 hap4 (the same four ancestry paths at every level)
