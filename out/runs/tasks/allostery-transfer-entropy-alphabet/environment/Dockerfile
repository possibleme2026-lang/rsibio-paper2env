FROM python:3.11-slim
WORKDIR /app
COPY data/ /app/data/
RUN pip install --no-cache-dir numpy scipy pandas scikit-learn biopython
RUN cd /app/data && python gen_inputs.py
