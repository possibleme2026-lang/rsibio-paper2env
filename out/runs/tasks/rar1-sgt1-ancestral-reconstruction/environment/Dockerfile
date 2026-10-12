FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    iqtree \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir biopython numpy scipy
WORKDIR /app
COPY data/ /app/data/
RUN cp /app/data/simulation.py /app/simulation.py && cp /app/data/simulation.py /tests/simulation.py
RUN cd /app/data && python3 gen_inputs.py && rm /app/data/gen_inputs.py /app/data/simulation.py
RUN mkdir -p /app/output
CMD ["/bin/bash"]
RUN ln -sf /usr/bin/iqtree2 /usr/local/bin/iqtree
