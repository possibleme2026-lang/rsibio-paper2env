FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.13.1 pandas==2.2.2 scikit-learn==1.4.2 statsmodels==0.14.2
COPY data/ /app/data/
COPY gen_data.py /app/gen_data.py
RUN python /app/gen_data.py --seed 20260728 --outdir /app/data && rm -f /app/gen_data.py
RUN mkdir -p /app/output /logs/verifier
CMD ["sleep", "infinity"]
