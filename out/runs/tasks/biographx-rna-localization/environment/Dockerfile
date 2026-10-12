FROM ubuntu:22.04
WORKDIR /app
COPY data/ /app/data/
RUN apt-get update && apt-get install -y --no-install-recommends \
        vienna-rna python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir numpy
RUN python3 /app/data/gen_inputs.py && chmod -R 444 /app/data
RUN mkdir -p /app/output && chmod 777 /app/output
