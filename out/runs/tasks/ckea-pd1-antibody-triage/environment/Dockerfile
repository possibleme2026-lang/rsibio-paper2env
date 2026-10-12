FROM continuumio/miniconda3
WORKDIR /app
COPY data/ /app/data/
RUN conda install -y -c conda-forge -c bioconda mdanalysis biopython pandas numpy scikit-learn anarci && \
    pip install --no-cache-dir abnumber && \
    conda clean -afy
RUN python /app/data/gen_md.py && rm /app/data/gen_md.py
RUN cp /app/data/gen_md.py /app/gen_md.py
