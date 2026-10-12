FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 statsmodels==0.14.2

COPY data/ /app/data/

# Build the deterministic inputs once, then delete the generator so the
# simulation seed / causal structure is not readable by the agent at solve time.
RUN mkdir -p /app/output \
    && python /app/data/gen_inputs.py /app/data \
    && rm -f /app/data/gen_inputs.py
