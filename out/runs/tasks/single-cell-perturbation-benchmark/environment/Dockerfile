FROM python:3.11-slim
WORKDIR /app
COPY data/ /app/data/
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    pip install --no-cache-dir numpy scipy pandas scikit-learn statsmodels anndata h5py && \
    python /app/data/gen_inputs.py && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /app/solution /app/output
