FROM python:3.11-slim
WORKDIR /app
COPY data/ /app/data/
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy pandas scipy
RUN python /app/data/gen_inputs.py && rm /app/data/gen_inputs.py
RUN mkdir -p /app/output
