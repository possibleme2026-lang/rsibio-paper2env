FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy pandas scipy
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py --outdir /app/data --truth /tmp/hidden_truth.json && rm /tmp/hidden_truth.json && rm /app/data/gen_inputs.py
