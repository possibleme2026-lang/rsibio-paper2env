FROM python:3.11-slim
WORKDIR /app
RUN mkdir -p /app/output /app/environment/data
COPY data/ /app/environment/data/
RUN chmod -R a-w /app/environment
CMD sleep infinity
