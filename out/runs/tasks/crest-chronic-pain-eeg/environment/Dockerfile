FROM python:3.11-slim
WORKDIR /app
RUN mkdir -p /app/data /app/output /app/environment
COPY data/ /app/environment/data/
RUN python /app/environment/data/generate_spectrograms.py
CMD ["/bin/bash"]
RUN cp /app/data/generate_spectrograms.py /app/generate_spectrograms.py
