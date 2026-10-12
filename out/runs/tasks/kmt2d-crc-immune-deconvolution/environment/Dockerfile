FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
COPY data/ /app/data/
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 lifelines==0.28.0 statsmodels==0.14.2 scikit-learn==1.4.2 \
    && python /app/data/gen_inputs.py \
    && rm -f /app/data/gen_inputs.py
WORKDIR /app
