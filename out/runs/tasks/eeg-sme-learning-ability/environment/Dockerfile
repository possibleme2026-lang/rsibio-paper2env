FROM python:3.11-slim
WORKDIR /app
COPY data/ /app/data/
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    pip install --no-cache-dir numpy pandas scipy statsmodels scikit-learn && \
    python /app/data/gen_inputs.py && rm /app/data/gen_inputs.py && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /app/output
