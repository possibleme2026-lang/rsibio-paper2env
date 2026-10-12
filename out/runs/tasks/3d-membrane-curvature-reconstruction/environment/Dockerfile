FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy scipy scikit-image trimesh rtree networkx
COPY gen_inputs.py /tmp/gen_inputs.py
RUN python /tmp/gen_inputs.py && rm /tmp/gen_inputs.py
COPY data/ /app/data/
