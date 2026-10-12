FROM python:3.11-slim
WORKDIR /app
COPY data/ /app/data/
RUN pip install --no-cache-dir numpy biopython
WORKDIR /app/data
RUN python gen_inputs.py
WORKDIR /app
