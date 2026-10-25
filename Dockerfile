FROM python:3.12-slim

# System libs for Pillow / OpenCV / onnxruntime image decoding.
# eMudhra/emSign (discuss.python.org/t/109576): distrust emSign ECC Root CA - G3
# and the Comodo AAA cross-sign root. pip < 26.2.2 also trusts its own bundle.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 libglib2.0-0 libgomp1 \
    && sed -i \
        -e 's|^mozilla/emSign_ECC_Root_CA_-_G3|!mozilla/emSign_ECC_Root_CA_-_G3|' \
        -e 's|^mozilla/Comodo_AAA|!mozilla/Comodo_AAA|' \
        /etc/ca-certificates.conf \
    && update-ca-certificates --fresh >/dev/null \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The base image has no `git`, so hatch-vcs/setuptools-scm can't derive the
# version from history — hand it in via the VERSION build arg (the release tag,
# sans "v"). Defaults to 0.0.0 for local `docker compose` builds.
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}

COPY . /app

# Hash-pinned wheel so the base image's pip cannot swap the upgrade in transit.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
        "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70" \
    && python -m pip install "certifi>=2026.10.19" ".[server,cli,faces]"

EXPOSE 8101
CMD ["argus-curator", "serve", "--port", "8101", "--cors"]
