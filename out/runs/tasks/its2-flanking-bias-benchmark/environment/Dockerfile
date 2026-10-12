FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends vsearch \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy pandas scikit-learn biopython joblib
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py /app/data
