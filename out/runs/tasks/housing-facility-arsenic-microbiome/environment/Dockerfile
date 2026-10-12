FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY data/ /app/data/
RUN pip install --no-cache-dir numpy pandas scipy scikit-learn scikit-bio
RUN python /app/data/gen_inputs.py /app/data && rm /app/data/gen_inputs.py
CMD ["/bin/bash"]
