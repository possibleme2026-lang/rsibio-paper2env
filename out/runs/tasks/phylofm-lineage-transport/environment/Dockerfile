FROM python:3.11-slim
WORKDIR /app
COPY data/ /app/data/
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 scikit-learn==1.5.0 pot==0.9.4 networkx==3.3 \
    && rm -rf /var/lib/apt/lists/*
RUN python /app/data/gen_inputs.py
