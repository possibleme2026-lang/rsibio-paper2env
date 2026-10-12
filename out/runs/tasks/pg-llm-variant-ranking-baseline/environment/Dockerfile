FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        scipy==1.13.1 \
        pandas==2.2.2 \
        biopython==1.83

COPY data/ /app/data/

# Materialise the benchmark inputs deterministically from the fixed seed.
RUN python /app/data/gen_inputs.py && ls -R /app/data | head -40

RUN mkdir -p /app/output /logs/verifier

CMD ["sleep", "infinity"]
