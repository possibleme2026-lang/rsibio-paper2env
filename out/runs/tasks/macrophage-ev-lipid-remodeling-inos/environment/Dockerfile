FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 statsmodels==0.14.2
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py --seed 20251211 --outdir /app/data && rm /app/data/gen_inputs.py
RUN mkdir -p /app/results
CMD ["bash"]
