FROM python:3.11-slim AS builder
WORKDIR /build
COPY data/ /build/data/
RUN cd /build/data && python gen_inputs.py && rm -f gen_inputs.py && rm -rf __pycache__

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends procps ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 scikit-learn==1.5.0
WORKDIR /app
COPY --from=builder /build/data/ /app/data/
RUN mkdir -p /app/output /logs/verifier
CMD ["/bin/bash"]
