# Review notes

- Data are synthetic but realistic: Nanopore amplicon reads generated with a deterministic simulator (seed 12345) at build time. The simulator is generic; the secret mutation frequencies and seed are only in the Dockerfile and tests/gold_gen.py, not readable by the agent.
- Shortcut probe: agent could read /app/data/gen_inputs.py, but it contains only a generic simulator without the secret config. The secret config is baked into the Dockerfile RUN command and is not present in the image.
- Tool chain: minimap2, samtools, bcftools, primer3-py installed in the container; grader uses primer3-py to validate primer properties.
- Known limitations: allele frequency tolerance is ±0.05; primer design is evaluated on sequence properties and mutation-site positioning, not on wet-lab validation.
