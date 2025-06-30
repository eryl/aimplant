# Use NVIDIA PyTorch base image with CUDA
# Due to dependencies of nvidia flare, we 
# get dependency conflicts if we use too new 
# an image here
FROM nvcr.io/nvidia/pytorch:24.04-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"


# Create a working directory
WORKDIR /app/workspace

COPY models/xlm-roberta /app/xlm-roberta
COPY . /app/workspace
COPY federatedhealth_mlm_job /app/federatedhealth_mlm_job
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

RUN uv sync
RUN uv pip install -e .

ENTRYPOINT ["/entrypoint.sh"]
