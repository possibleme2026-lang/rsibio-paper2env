FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy
COPY data/ /app/environment/data/
RUN python /app/environment/data/generate_data.py
CMD ["bash"]
RUN cp /app/data/generate_data.py /app/generate_data.py
