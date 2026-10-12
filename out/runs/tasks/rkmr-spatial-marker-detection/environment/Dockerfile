FROM python:3.11-slim
WORKDIR /app
COPY data/ /app/data/
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && pip install --no-cache-dir numpy pandas scikit-learn scipy \
    && python /app/data/make_inputs.py \
    && rm /app/data/make_inputs.py \
    && mkdir -p /app/output
