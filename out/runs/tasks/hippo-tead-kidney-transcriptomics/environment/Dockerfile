FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir 'numpy>=1.26,<3' 'pandas>=2.1,<3' 'scipy>=1.11' 'scikit-learn>=1.3' 'pydeseq2>=0.5,<0.6'
COPY data/ /app/data/
RUN python /app/data/gen_inputs.py /app/data \
    && rm -f /app/data/gen_inputs.py \
    && ls -l /app/data
RUN mkdir -p /app/output /logs/verifier
