FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        scipy==1.11.4 \
        pandas==2.1.4 \
        scikit-learn==1.3.2

COPY data/ /app/data/

# Generate the deterministic task data and then delete the generator so the
# hidden generating parameters are not readable inside the running container.
RUN cd /app/data && python gen_inputs.py && rm -f gen_inputs.py \
    && mkdir -p /app/output \
    && wc -l /app/data/*.csv
