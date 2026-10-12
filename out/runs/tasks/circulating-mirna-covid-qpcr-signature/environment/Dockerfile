FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir 'numpy==1.26.4' 'pandas==2.1.4' 'scipy==1.11.4' 'scikit-learn==1.4.2' 'statsmodels==0.14.1'
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py && rm -f /app/data/gen_inputs.py && mkdir -p /app/output
