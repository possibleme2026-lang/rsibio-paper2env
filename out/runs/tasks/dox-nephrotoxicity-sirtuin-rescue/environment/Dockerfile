FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev && rm -rf /var/lib/apt/lists/*
COPY data/ /app/data/
RUN pip install --no-cache-dir numpy pandas scipy statsmodels scikit-learn
RUN python /app/data/gen_inputs.py && rm /app/data/gen_inputs.py
RUN mkdir -p /app/output
