# Docker and the GPU

## Docker

```bash
docker compose up --build
```

Builds the CUDA image and gives it the GPU (it needs the NVIDIA container
runtime). `CURATOR_GPU=0` in `.env` builds the CPU image instead. Releases
publish both: `ghcr.io/smk762/argus-curator:<version>` (CPU) and
`:<version>-gpu` / `:latest-gpu`. The GPU image installs `[gpu]`, with the CUDA
13 and cuDNN 9 libraries as NVIDIA's pip wheels on the python-slim base, so the
host driver must support CUDA 13. It is about 5.5 GB, against 1.6 GB for CPU.

Bind-mounts a dataset into `/data/images`, an output dir into `/data/out`, and
persists the scan cache + InsightFace model downloads across rebuilds. A
converted MiVOLO model under `~/.cache/argus_curator/models/` is
picked up through the same cache mount.

## Sharing the GPU

On CUDA, curator takes argus-cortex's host-wide GPU lease (`argus_cortex.gpu`)
around each image it runs a model on: InsightFace with MiVOLO,
Places365 (per batch) and Fashionpedia, plus a larger ask when a model loads.
It is what the GPU image (`:<version>-gpu`, the compose default) runs.
The lease is taken per image, so argus-lens, or a `POST /faces` from
argus-darkroom, can use the card between a scan's images. It is never taken on
CPU, or when `device="cuda"` but onnxruntime has no CUDA provider (sessions
would only run on CPU anyway). InsightFace's pack download happens before the
load lease, so a cold start does not hold the GPU while weights hit the disk.

- `ARGUS_GPU_COORDINATOR=lease` turns it on. `none`, the library default,
  gates nothing; `docker-compose.yaml` sets `lease`. Every argus process on the
  host must name the same `ARGUS_GPU_LEASE_PATH` on a shared directory.
  Containers have their own `/tmp`, so compose mounts `ARGUS_GPU_LOCK_DIR`
  (`/tmp/argus-gpu` on the host) at `/run/argus-gpu` and uses
  `/run/argus-gpu/argus-gpu.lock`; argus-lens must mount the same host
  directory and use the same file.
- With the `[gpu]` extra (which brings NVML), the lease also waits until the card
  has the memory a model needs free, counted across every process on it, so
  ComfyUI or an Ollama-backed SingGuard on the same GPU are included. The
  amounts are rough, per model (`runtime.VRAM_MB`).
- `ARGUS_GPU_LEASE_TIMEOUT_S` (300) caps the wait. A scan that cannot get the GPU
  in time fails, rather than skipping an image's faces, and the server answers
  503 with `Retry-After: 5` (the streaming scan drops any partial cache).
  `POST /faces` waits at most 30 s and then answers the same 503. `GET /health`
  reports `gpu_coordinator`.
- `pip install "argus-curator[gpu]"` can still pull CPU `onnxruntime` via
  InsightFace. If `CUDAExecutionProvider` is missing, uninstall `onnxruntime`
  and reinstall `onnxruntime-gpu` (the GPU image does this and fails the build
  if CUDA is still absent).
