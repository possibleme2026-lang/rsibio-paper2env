FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy pandas scipy scikit-learn
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py
RUN mkdir -p /app/output && chmod 777 /app/output
