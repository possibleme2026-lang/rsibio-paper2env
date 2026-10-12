FROM python:3.11-slim
WORKDIR /app
COPY data/ /app/data/
RUN pip install --no-cache-dir numpy scipy pandas scikit-learn
RUN python /app/data/gen_inputs.py --out /app/data/ && rm -f /app/data/gen_inputs.py
RUN mkdir -p /app/output
CMD ["/bin/bash"]
RUN cp /app/data/tlr_sim.py /app/tlr_sim.py && cp /app/data/tlr_sim.py /tests/tlr_sim.py
