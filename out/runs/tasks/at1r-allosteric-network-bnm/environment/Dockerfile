FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLBACKEND=Agg
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2 scipy==1.11.4 scikit-learn==1.4.2 networkx==3.3
COPY data/ /app/data/
RUN python /app/data/gen_at1r_data.py --outdir /app/data && mkdir -p /app/output
CMD ["/bin/bash"]
