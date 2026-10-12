FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxext6 \
    libsm6 \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy pandas scipy scikit-learn rdkit biopython
COPY data/ /app/data/
RUN python /app/data/gen_calibration.py --output /app/data/calibration && rm /app/data/gen_calibration.py
RUN mkdir -p /app/solution /app/output
