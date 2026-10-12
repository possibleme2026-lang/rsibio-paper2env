FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY data/ /app/data/
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.11.4 pandas==2.1.4 scikit-learn==1.4.0 scanpy==1.10.0
RUN python /app/data/gen_inputs.py
RUN mkdir -p /app/output
