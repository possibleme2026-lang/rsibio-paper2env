FROM python:3.11-slim
WORKDIR /app
COPY data/ /app/data/
RUN pip install --no-cache-dir catboost==1.2.1 pandas numpy scikit-learn && \
    python /app/data/gen_inputs.py && \
    rm /app/data/gen_inputs.py
