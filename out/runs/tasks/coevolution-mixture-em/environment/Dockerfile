FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy scipy
COPY data/ /app/data/
RUN cd /app/data && python gen_inputs.py
RUN mkdir -p /app/results
