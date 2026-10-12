FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        scipy==1.11.4 \
        pandas==2.1.4 \
        lifelines==0.27.8

COPY data/ /app/data/
RUN python /app/data/gen_inputs.py --outdir /app/data --seed 20260802

RUN mkdir -p /app/output /logs/verifier

CMD ["sleep", "infinity"]
