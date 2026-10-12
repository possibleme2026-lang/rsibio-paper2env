FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        scipy==1.13.1 \
        scikit-learn==1.4.2 \
        xgboost==2.0.3
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py --out /app/data/pond_temperatures.csv \
    && rm -f /app/data/gen_inputs.py
RUN mkdir -p /app/output
WORKDIR /app
