FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY data/ /app/data/

RUN apt-get update && apt-get install -y --no-install-recommends bash ca-certificates coreutils && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.13.1 pandas==2.2.2 gfapy

RUN cd /app/data && python gen_inputs.py /app/data && rm -f /app/data/gen_inputs.py && mkdir -p /app/results
