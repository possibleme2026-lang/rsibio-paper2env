FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
COPY data/ /app/data/
RUN pip install --no-cache-dir numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0 scipy==1.10.1
RUN python /app/data/gen_inputs.py && rm /app/data/gen_inputs.py
RUN mkdir -p /app/output
