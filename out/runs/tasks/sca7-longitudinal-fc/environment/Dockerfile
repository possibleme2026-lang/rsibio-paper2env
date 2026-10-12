FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY data/ /app/data/
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2 scipy==1.13.0 statsmodels==0.14.2 scikit-learn==1.4.2
RUN python /app/data/gen_inputs.py
RUN mkdir -p /app/output
