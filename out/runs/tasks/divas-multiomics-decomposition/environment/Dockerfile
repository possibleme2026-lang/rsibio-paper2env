FROM rocker/r-ver:4.3.1
WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir numpy pandas scipy
RUN R -e "install.packages('remotes', repos='https://cloud.r-project.org')"
RUN R -e "remotes::install_github('ByronSyun/DIVAS', dependencies=TRUE)"
COPY data/ /app/data/
RUN python3 /app/data/gen_inputs.py --output /app/data && rm /app/data/gen_inputs.py
