FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        scipy==1.11.4 \
        nibabel==5.2.1 \
        dipy==1.9.0

WORKDIR /app

COPY data/ /app/data/

# Generate the synthetic cohort deterministically, then remove the generator so
# that the held-out propagation parameters are not readable at solve time.
RUN python /app/data/gen_inputs.py \
    && rm -f /app/data/gen_inputs.py \
    && mkdir -p /app/output/occupancy /app/output/occupancy_iso /app/output/occupancy_tensor /app/output/streamlines \
    && ls -R /app/data | head -60

CMD ["/bin/bash"]
