# =============================================================================
# Dockerfile — Category Classification FTI Pipeline (Jupyter Server)
# =============================================================================
# This image packages the Feature · Training · Inference notebooks and their
# shared utilities into a ready-to-run Jupyter server.
#
# Build:
#   docker build -t category-classification-jupyter .
#
# Run (interactive, foreground):
#   docker run -p 8888:8888 category-classification-jupyter
#
# Run (detached, with named volume for persistence):
#   docker run -d -p 8888:8888 -v $(pwd)/data:/app/data --name cc-jupyter category-classification-jupyter
#
# Connect from host:
#   Open http://localhost:8888 in your browser (no token required).
# =============================================================================

# -----------------------------------------------------------------------------
# Base image — Python 3.11 provides `tomllib` (built-in) required by notebooks
# -----------------------------------------------------------------------------
FROM python:3.11-slim

LABEL maintainer="FTI Pipeline Team"
LABEL description="Jupyter server for category classification Feature/Training/Inference pipelines"

# -----------------------------------------------------------------------------
# System-level dependencies
# -----------------------------------------------------------------------------
# build-essential  : gcc/g++/make — needed when pip falls back to source builds
#                    (scipy, scikit-learn, lightgbm wheels may require compilation
#                     on less-common platforms or when pre-built wheels are absent)
# libgomp1         : OpenMP runtime — REQUIRED by LightGBM at run time
# ca-certificates  : SSL root certificates — REQUIRED for HTTPS connections to
#                    Hopsworks (Feature Store / Model Registry) and CloudFront
# librdkafka-dev   : C library headers — REQUIRED to build confluent-kafka Python package
# libssl-dev       : OpenSSL headers — REQUIRED by confluent-kafka for TLS/SASL support
# git              : occasionally required by pip for VCS-based dependencies
# curl             : useful for debugging network issues inside the container
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    ca-certificates \
    librdkafka-dev \
    libssl-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Working directory
# -----------------------------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------------------------
# Python packaging tools — upgrade pip and install wheel/setuptools first so
# that subsequent packages can build from source cleanly if a wheel is missing.
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# Python dependencies
# -----------------------------------------------------------------------------
# Copy requirements.txt alone first so Docker layer-caching works:
# the layer below only rebuilds when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Application files
# -----------------------------------------------------------------------------
# Copy all files explicitly named in the project contract.
# These are the 8 files that constitute the reproducible FTI pipeline.
COPY category_classification.py \
     fa0_data_acquisition.py \
     category_classification_f.ipynb \
     category_classification_t.ipynb \
     category_classification_i.ipynb \
     category_classification_fti.toml \
     category_metadata.json \
     ./

# -----------------------------------------------------------------------------
# Data directory
# -----------------------------------------------------------------------------
# The notebooks reference ./data for downloaded parquet/CSV files.
# Pre-create it and open permissions so the notebook kernel can write here.
RUN mkdir -p /app/data && chmod 777 /app/data

# -----------------------------------------------------------------------------
# Jupyter configuration
# -----------------------------------------------------------------------------
# Bind to 0.0.0.0 so the server is reachable from the Docker host.
# Disable browser auto-launch (no GUI inside the container).
# Allow root because the container runs as root by default.
# Empty token/password makes local development convenient; for production,
# replace the token with a strong secret or use JupyterHub.
# -----------------------------------------------------------------------------
ENV JUPYTER_ENABLE_LAB=no

EXPOSE 8888

CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
