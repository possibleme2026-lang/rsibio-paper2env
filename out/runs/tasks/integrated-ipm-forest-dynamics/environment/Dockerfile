FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy pandas scipy scikit-learn statsmodels
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py && rm /app/data/gen_inputs.py
