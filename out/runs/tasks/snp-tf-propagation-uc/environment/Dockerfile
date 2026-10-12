FROM python:3.11-slim
WORKDIR /app
COPY data/ /app/data/
RUN apt-get update && apt-get install -y --no-install-recommends \
    && pip install --no-cache-dir numpy pandas scipy \
    && python /app/data/gen_inputs.py \
    && mkdir -p /app/output
