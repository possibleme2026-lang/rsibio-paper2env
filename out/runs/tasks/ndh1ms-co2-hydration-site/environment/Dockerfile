FROM python:3.11-slim
WORKDIR /app
COPY data/ /app/data/
RUN pip install --no-cache-dir biopython numpy scipy \
    && python /app/data/gen_inputs.py \
    && rm /app/data/gen_inputs.py \
    && rm -rf /root/.cache/pip
