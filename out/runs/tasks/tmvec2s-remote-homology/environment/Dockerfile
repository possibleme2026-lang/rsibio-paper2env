FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy scipy scikit-learn biopython pandas
COPY data/ /app/data/
RUN python /app/data/gen_all.py /app/data && rm /app/data/gen_all.py
WORKDIR /app
