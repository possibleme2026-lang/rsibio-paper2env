FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libglib2.0-0t64 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        scipy==1.13.1 \
        scikit-image==0.23.2 \
        pillow==10.4.0

WORKDIR /app

# task data is copied from the build context (environment/)
COPY data/ /app/data/

# the deterministic simulator writes train/valid/test and is then removed from the
# image so that the agent cannot call the renderer directly
RUN python /app/data/gen_inputs.py --out /app/data \
    && rm -f /app/data/gen_inputs.py \
    && rm -rf /app/data/__pycache__

RUN mkdir -p /app/output

CMD ["bash"]
