FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends jq bc ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir 'numpy==1.26.4' 'pandas==2.2.3' 'scipy==1.13.1' \
    'scikit-learn==1.5.2' 'statsmodels==0.14.4'
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py --outdir /app/data && mkdir -p /app/outputs
