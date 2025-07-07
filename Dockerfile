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
ENV PATH="/root/.local/bin:${PATH}"

# Create a working directory
WORKDIR /app/workspace

COPY models/xlm-roberta /app/xlm-roberta
COPY src /app/workspace/src
COPY .python-version /app/workspace
COPY pyproject.toml /app/workspace
COPY local_train.py /app/workspace
COPY federatedhealth_mlm_job /app/federatedhealth_mlm_job
COPY entrypoint.sh /app/workspace/entrypoint.sh

RUN uv sync
RUN --mount=type=cache,target=/root/.cache uv pip install -e .

RUN chmod +x /app/workspace/entrypoint.sh

ENTRYPOINT ["/app/workspace/entrypoint.sh"]
