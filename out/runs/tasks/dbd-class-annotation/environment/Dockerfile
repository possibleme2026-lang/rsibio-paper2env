FROM python:3.11-slim AS datagen
WORKDIR /app
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py --outdir /build/out

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        emboss bedtools samtools ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy biopython
COPY --from=datagen /build/out/ /app/data/
RUN mkdir -p /app/output
