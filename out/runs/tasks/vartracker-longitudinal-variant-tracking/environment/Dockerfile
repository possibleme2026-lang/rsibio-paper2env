FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends bcftools samtools tabix && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir biopython pandas numpy
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py && rm /app/data/gen_inputs.py
RUN mkdir -p /app/output
