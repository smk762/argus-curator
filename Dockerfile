FROM python:3.12-slim

# System libs for Pillow / OpenCV / onnxruntime image decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The base image has no `git`, so hatch-vcs/setuptools-scm can't derive the
# version from history — hand it in via the VERSION build arg (the release tag,
# sans "v"). Defaults to 0.0.0 for local `docker compose` builds.
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}

COPY . /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install ".[server,cli,faces]"

EXPOSE 8101
CMD ["argus-curator", "serve", "--port", "8101", "--cors"]
