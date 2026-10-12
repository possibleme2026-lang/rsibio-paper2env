FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2 scipy==1.11.4
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py --outdir /app/data && mkdir -p /app/output
