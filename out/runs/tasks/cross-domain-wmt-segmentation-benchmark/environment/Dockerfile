FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.13.1 nibabel==5.2.1
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py && rm -f /app/data/gen_inputs.py && mkdir -p /app/output
