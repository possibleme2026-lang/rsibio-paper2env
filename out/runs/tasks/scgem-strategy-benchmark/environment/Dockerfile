FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir 'numpy==1.26.4' 'pandas==2.2.2'

COPY data/ /app/data/

# Build the deterministic task inputs (seed 20260802) inside the image.
RUN python /app/data/gen_inputs.py \
 && mkdir -p /app/output /logs/verifier \
 && ls -la /app/data

WORKDIR /app
