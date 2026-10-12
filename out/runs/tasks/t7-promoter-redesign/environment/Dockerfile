FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY data/ /app/data/
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.1 biopython==1.84 \
    && python /app/data/gen_inputs.py \
    && mkdir -p /app/output
