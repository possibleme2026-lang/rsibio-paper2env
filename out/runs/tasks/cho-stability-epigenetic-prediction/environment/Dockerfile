FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy pandas scikit-learn shap
COPY data/ /app/data/
RUN cd /app/data && python gen_inputs.py && rm gen_inputs.py
RUN mkdir -p /app/output
