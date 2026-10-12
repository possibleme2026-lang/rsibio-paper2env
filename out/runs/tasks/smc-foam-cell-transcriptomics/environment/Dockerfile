FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
COPY data/ /app/data/
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 statsmodels==0.14.2 gseapy==1.1.3 scanpy==1.10.3
RUN python /app/data/gen_inputs.py --outdir /app/data --seed 42
