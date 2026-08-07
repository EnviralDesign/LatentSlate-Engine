ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    LATENTSLATE_ENGINE_HOME=/data

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl ffmpeg git python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /app
COPY . /app
RUN uv sync --extra h3 --no-dev

EXPOSE 8765
VOLUME ["/data", "/root/.cache/huggingface"]
CMD ["uv", "run", "latentslate-engine", "serve", "--host", "0.0.0.0", "--port", "8765"]
