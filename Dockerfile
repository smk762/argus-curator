FROM gpu_base

WORKDIR /app

COPY dist/*.whl /tmp/wheels/
RUN --mount=type=cache,target=/root/.cache/pip \
    set -- /tmp/wheels/*.whl && \
    pip install --upgrade pip && \
    pip install "$1[server,gpu]" && \
    rm -rf /tmp/wheels

EXPOSE 8101
CMD ["argus-curator", "serve", "--port", "8101", "--cors"]
