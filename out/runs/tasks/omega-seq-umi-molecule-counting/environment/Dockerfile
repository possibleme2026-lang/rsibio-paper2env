FROM python:3.11-slim
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        samtools bedtools ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pysam==0.22.0 scipy==1.11.4
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py && rm -f /app/data/gen_inputs.py \
    && mkdir -p /app/output && ls -l /app/data
CMD ["/bin/bash"]
