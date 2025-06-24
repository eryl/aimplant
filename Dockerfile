# Use NVIDIA PyTorch base image with CUDA
# Due to dependencies of nvidia flare, we 
# get dependency conflicts if we use too new 
# an image here
FROM nvcr.io/nvidia/pytorch:24.04-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    FH_MODEL_DIR=/app/xlm-roberta \
    FH_TRAINING_DATA=/app/data/training_data.txt \
    FH_DEV_DATA=/app/data/dev_data.txt \
    FH_TEST_DATA=/app/data/test_data.txt \
    PYTHONPATH=/app/apps/xlmroberta_mlm

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app/workspace

COPY models/xlm-roberta /app/xlm-roberta
COPY apps /app/apps
COPY requirements_docker.txt /requirements.txt
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

RUN python3 -m pip install -r /requirements.txt

ENTRYPOINT ["/entrypoint.sh"]
