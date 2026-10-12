FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
RUN pip install --no-cache-dir \
      numpy==1.26.4 \
      scipy==1.13.1 \
      pandas==2.2.2 \
      biopython==1.83 \
      statsmodels==0.14.2
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py --outdir /app/data \
 && rm -f /app/data/gen_inputs.py \
 && mkdir -p /app/output/trees
