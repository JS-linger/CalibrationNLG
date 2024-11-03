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

# Add conda to path
ENV PATH="/opt/conda/bin:${PATH}"

# Set working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/JS-linger/CalibrationNLG.git .

# Copy environment file
COPY controlNLG_env.yml .

# Create conda environment
RUN conda env create -f controlNLG_env.yml

# Activate environment
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate controlNLG_env" >> ~/.bashrc
ENV PATH=/opt/conda/envs/controlNLG_env/bin:$PATH

CMD ["python", "main.py"]