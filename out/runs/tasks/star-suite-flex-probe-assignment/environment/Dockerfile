FROM python:3.11-slim
WORKDIR /app
COPY data/ /app/data/
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy
RUN cd /app/data && python gen_inputs.py
WORKDIR /app
