FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libtiff-tools ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "numpy>=1.26,<2.0" "scipy>=1.11" "pandas>=2.1" "tifffile>=2024.2" "scikit-image>=0.22"
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py && rm -f /app/data/gen_inputs.py && rm -rf /app/data/__pycache__
RUN mkdir -p /app/results
