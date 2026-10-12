FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.13.0 scikit-image==0.22.0 networkx==3.3 tifffile==2024.5.10 pandas==2.2.2
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py --seed 872634918273 --out-dir /app/data
RUN mkdir -p /app/output
