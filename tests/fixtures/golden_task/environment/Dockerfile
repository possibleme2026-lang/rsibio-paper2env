FROM python:3.11-slim
WORKDIR /app
COPY data/ /app/data/
RUN pip install --no-cache-dir numpy
# The gold standard (tests/) is mounted only in the verifier phase, never
# baked into the agent image - copying /app/data files cannot reveal it.
