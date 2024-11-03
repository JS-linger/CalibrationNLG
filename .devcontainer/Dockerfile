FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and minimal dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# Add conda to path and initialize
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda init bash

# Set working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/JS-linger/CalibrationNLG.git .

# Copy environment file
COPY controlNLG_env.yml .

# Create conda environment
RUN conda env create -f controlNLG_env.yml

# Initialize shell with conda
SHELL ["/bin/bash", "--login", "-c"]

# Add environment activation to bashrc
RUN echo "conda activate controlNLG_env" >> ~/.bashrc && \
    echo "export PATH=/opt/conda/envs/controlNLG_env/bin:$PATH" >> ~/.bashrc

CMD ["/bin/bash", "-c", "source ~/.bashrc && python main.py"]