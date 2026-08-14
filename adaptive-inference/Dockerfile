# =============================================================================
# Dockerfile – adaptive-inference / R1.x development image
#
# Design decisions:
#   - Base: python:3.11-slim  (Debian Bookworm, ~130 MB compressed)
#     Chosen over full python:3.11 to keep the image lean.
#     No CUDA/NVIDIA base yet – will be swapped for nvidia/cuda:* in R1.5+.
#   - PyTorch installed via pip with the CPU-only index url so the wheel is
#     ~300 MB instead of ~2 GB (GPU version).  Change --index-url to get CUDA.
#   - All source code lives under /workspace (project root mounted as volume).
#   - Non-root user `appuser` (UID 1000) for safer container execution.
#   - Image is built once and reused; datasets/checkpoints/results are mounted
#     as bind volumes (see docker-compose.yml) so they never get baked in.
# =============================================================================

FROM python:3.11-slim AS base

# ── OS-level dependencies ─────────────────────────────────────────────────────
# build-essential: needed to compile any C extensions that pip may trigger.
# git: useful for experiment tracking scripts that tag commits.
# curl / wget: handy for ad-hoc downloads during research.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd --uid 1000 --create-home --shell /bin/bash appuser

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /workspace

# Copy only requirements first to leverage Docker layer cache.
COPY requirements.txt .

# Install PyTorch (CPU-only) separately so we can point to the right index.
# If you switch to a CUDA base image, change --index-url accordingly:
#   https://download.pytorch.org/whl/cu121   (CUDA 12.1)
#   https://download.pytorch.org/whl/cu118   (CUDA 11.8)
RUN pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements (numpy, scipy, pandas, pytest, …)
RUN pip install --no-cache-dir \
        --extra-index-url https://pypi.org/simple \
        -r requirements.txt

# ── Project source ────────────────────────────────────────────────────────────
# The entire adaptive-inference/ tree is copied into the image so the container
# is self-contained for CI.  Bind mounts in docker-compose.yml overlay the
# data directories (train_set, eval_set, checkpoints, results) at runtime.
COPY . .

# Make sure tasks/ is importable as a package when running from /workspace.
# We add /workspace to PYTHONPATH instead of installing a package so that
# edits to source files are picked up immediately when using a bind mount.
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# ── Runtime defaults ──────────────────────────────────────────────────────────
USER appuser

# Default working directory: project root so all paths (tasks/, training/, …)
# can be addressed with their natural prefix from any command.
WORKDIR /workspace

CMD ["python", "tasks/test_generator.py"]
