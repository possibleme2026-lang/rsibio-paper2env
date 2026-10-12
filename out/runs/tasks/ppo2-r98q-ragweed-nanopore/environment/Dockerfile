FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    minimap2 samtools bcftools gzip \
    && rm -rf /var/lib/apt/lists/*
COPY data/ /app/data/
RUN pip install --no-cache-dir primer3-py pysam primer3
RUN python /app/data/gen_inputs.py --seed 12345 --config '{"reference_length":1700,"amplicon_start":1,"amplicon_end":600,"mutation_site":293,"populations":{"S":{"n_reads":2000,"freqs":{"WT_R98":1.0,"R98L":0.0,"R98Q":0.0}},"R1":{"n_reads":2500,"freqs":{"WT_R98":0.05,"R98L":0.75,"R98Q":0.20}},"R2":{"n_reads":2500,"freqs":{"WT_R98":0.03,"R98L":0.02,"R98Q":0.95}}},"error_rate_sub":0.03,"error_rate_ins":0.01,"error_rate_del":0.02}' --outdir /app/data
RUN mkdir -p /app/output
