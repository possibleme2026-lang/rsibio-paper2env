FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy pandas scipy scikit-learn
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py
RUN mkdir -p /app/output
