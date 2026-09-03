FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        vim \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . ./ultralytics

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        torch==2.4.1 \
        torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu118 \
        --extra-index-url https://pypi.org/simple

RUN pip install -e ./ultralytics

RUN pip install --no-cache-dir numpy==1.26.4

CMD ["/bin/bash"]