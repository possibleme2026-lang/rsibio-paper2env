FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        jq \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        scipy==1.13.1 \
        statsmodels==0.14.2 \
        lifelines==0.29.0

WORKDIR /app
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py && mkdir -p /app/output

CMD ["/bin/bash"]
