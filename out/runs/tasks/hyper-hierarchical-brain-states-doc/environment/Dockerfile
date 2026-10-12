FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.11.4 scikit-learn==1.4.2 pandas==2.2.1
COPY data/ /app/data/
RUN mkdir -p /app/outputs && python /app/data/gen_inputs.py --out /app/data && ls /app/data/bold | wc -l
WORKDIR /app
