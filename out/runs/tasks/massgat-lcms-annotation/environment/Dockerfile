FROM python:3.11-slim
WORKDIR /app
COPY data/ /app/data/
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir numpy pandas networkx scikit-learn pyteomics
RUN python /app/data/gen_data.py --seed 42 --n_compounds 30 --n_noise 100 --out_features /app/data/sample_features.csv --out_labels /app/data/sample_labels.csv
