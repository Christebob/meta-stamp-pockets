# =============================================================================
# META-STAMP V3 Backend Dockerfile
# =============================================================================
# Multi-stage Docker build for production-ready FastAPI backend service
# Uses Python 3.11-slim base image with Poetry for dependency management
# Optimized for minimal image size and enhanced security
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Dependencies (Builder)
# -----------------------------------------------------------------------------
# This stage installs Poetry and all production dependencies into a virtual
# environment that will be copied to the final runtime stage.
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Set environment variables for Poetry
# POETRY_NO_INTERACTION: Disable interactive prompts
# POETRY_VIRTUALENVS_IN_PROJECT: Create .venv in project directory for easy copy
# POETRY_VIRTUALENVS_CREATE: Ensure venv is created
# POETRY_CACHE_DIR: Set cache directory for Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_HOME="/opt/poetry"

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install system dependencies required for building Python packages
# - curl: Required to download Poetry installer
# - build-essential: Required for compiling some Python packages (e.g., numpy)
# - libffi-dev: Required for cryptography package
# - gcc: C compiler for native extensions
# - g++: C++ compiler for some machine learning packages
# - libsndfile1-dev: Required for librosa audio processing
# - libgomp1: OpenMP runtime for numpy/scipy parallel operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libffi-dev \
    gcc \
    g++ \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry using the official installer script
# MUST match the version used to generate poetry.lock (2.3.3)
RUN curl -sSL https://install.python-poetry.org | python3 - --version 2.3.3

# Set working directory for the application
WORKDIR /app

# Copy dependency specification files
# poetry.lock* uses glob pattern to handle case where lock file doesn't exist yet
COPY pyproject.toml poetry.lock* ./

# Create a placeholder README.md so Poetry doesn't error on missing readme
# (pyproject.toml declares readme = "README.md" but .dockerignore excludes it)
RUN touch README.md

# Install production dependencies only (no dev dependencies)
# --no-root: Don't install the project itself, just dependencies
# --only main: Only install main dependencies (excludes dev group)
# After install, remove the cache to reduce layer size
RUN poetry install --only main --no-root \
    && rm -rf $POETRY_CACHE_DIR

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
# This stage creates the final minimal image with only the necessary runtime
# dependencies and application code. No build tools are included.
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Set Python environment variables for production
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files
# PYTHONUNBUFFERED: Ensures stdout/stderr are sent straight to terminal
# PYTHONFAULTHANDLER: Enable fault handler for better debugging on crashes
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# Install minimal runtime system dependencies
# - libgomp1: OpenMP runtime required by numpy/scipy for parallel operations
# - libsndfile1: Audio file I/O library required by librosa
# - ffmpeg: Multimedia framework for audio/video processing (used by librosa/opencv)
# - libgl1: OpenGL library required by opencv-python-headless
# - libglib2.0-0: GLib library required by opencv
# - libmagic1: Required by python-magic for file type detection
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libsndfile1 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create a non-root user for security best practices
# Running containers as non-root reduces the impact of potential security vulnerabilities
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# Copy the virtual environment from builder stage
# This is the key optimization - we only copy the installed packages,
# not the build tools or cache files
COPY --from=builder /app/.venv /app/.venv

# Add virtual environment to PATH so python and pip point to venv versions
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source code
# Separate COPY commands allow better caching - if only app code changes,
# the dependency layer doesn't need to be rebuilt
COPY app/ /app/app/
COPY main.py /app/

# Change ownership of application files to non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user for runtime
USER appuser

# Expose the FastAPI application port
# This is documentation - the actual port binding happens at runtime
EXPOSE 8000

# Health check configuration for container orchestration
# This allows Docker and orchestration tools (Kubernetes, Docker Compose)
# to verify the container is healthy and ready to receive traffic
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request, os; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT','8000') + '/health', timeout=5)" || exit 1

# Set the default command to run the FastAPI application with Uvicorn
# --host 0.0.0.0: Listen on all network interfaces (required for container networking)
# --port: Uses PORT env var from Railway (defaults to 8000)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
