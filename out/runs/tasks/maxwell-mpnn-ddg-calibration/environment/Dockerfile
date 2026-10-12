FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        scipy==1.13.1 \
        scikit-learn==1.5.1 \
        biopython==1.83 \
    && rm -rf /var/lib/apt/lists/*

# Build context is environment/ ; task data is copied as data/ -> /app/data/
COPY data/ /app/data/

# Deterministically materialise the realistic input files, then remove the generator
# so that the ground-truth model is not readable at solve time (it stays in the blueprint).
RUN python3 /app/data/gen_inputs.py && rm -f /app/data/gen_inputs.py

RUN mkdir -p /app/output /logs/verifier
