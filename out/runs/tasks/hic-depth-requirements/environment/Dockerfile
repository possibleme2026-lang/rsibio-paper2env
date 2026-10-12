FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.13.1 pandas==2.2.2
WORKDIR /app
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py /app/data && rm -f /app/data/gen_inputs.py
RUN mkdir -p /app/output
