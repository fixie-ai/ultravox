# CUDA Version
ARG CUDA_VERSION=12.8.1
ARG BASE_IMAGE="nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04"

FROM ${BASE_IMAGE}

ENV \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.7.1 \
    CUDA_VERSION=${CUDA_VERSION} \
    POETRY_HOME=/opt/poetry \
    PROJECT_ROOT=/app \
    VENV_PATH=/app/.venv \
    POETRY_VIRTUALENVS_IN_PROJECT=true

# Set working directory
WORKDIR ${PROJECT_ROOT}

# Install Python 3.11 and venv (3.12 conflicts with mecab-ko)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common build-essential git vim openssh-client emacs \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Install poetry in its own venv.
RUN python3.11 -m venv $POETRY_HOME
ENV PATH="${VENV_PATH}/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${POETRY_HOME}/bin"
RUN $POETRY_HOME/bin/pip install poetry==$POETRY_VERSION

# Copy poetry files
COPY pyproject.toml $PROJECT_ROOT/
COPY poetry.lock $PROJECT_ROOT/

# make sure poetry is available in the PATH
RUN poetry install --no-interaction --no-ansi --without dev --no-cache --no-root && \
    rm -rf ~/.cache/pypoetry
