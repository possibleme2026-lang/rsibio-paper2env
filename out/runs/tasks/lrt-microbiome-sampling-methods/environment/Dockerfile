FROM python:3.11-slim

WORKDIR /app

COPY data/ /app/data/

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates procps \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir numpy pandas scipy scikit-bio statsmodels  skbio\
    && python /app/data/gen_inputs.py \
    && rm -f /app/data/gen_inputs.py \
    && mkdir -p /app/output

ENV PYTHONUNBUFFERED=1
