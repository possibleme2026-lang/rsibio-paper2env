FROM python:3.11-slim AS data-builder
WORKDIR /build
# The generator lives only in this throw-away stage; it never reaches the final image.
COPY data/ /build/data/
RUN pip install --no-cache-dir "numpy==1.26.4" \
 && python /build/data/generate_data.py /build/generated

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates procps \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
      "numpy==1.26.4" "pandas==2.2.2" "scipy==1.13.1" \
      "scikit-learn==1.4.2" "biopython==1.83" "statsmodels==0.14.2" "dendropy==4.6.1"
COPY --from=data-builder /build/generated/ /app/data/
RUN mkdir -p /app/output
CMD ["sleep", "infinity"]
