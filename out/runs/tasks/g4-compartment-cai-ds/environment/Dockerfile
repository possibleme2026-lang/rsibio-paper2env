FROM python:3.11-slim
WORKDIR /app
COPY data/ /app/data/
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir numpy scipy pandas scikit-learn statsmodels biopython \
    && python /app/data/gen_inputs.py \
    && mkdir -p /app/output
CMD ["sleep", "infinity"]
