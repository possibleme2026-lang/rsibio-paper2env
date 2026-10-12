FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    emboss \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scipy==1.13.1 \
    biopython==1.83 \
    scikit-learn==1.5.1 \
    rdkit==2024.3.2
COPY data/ /app/data/
RUN python /app/data/gen_assays.py
