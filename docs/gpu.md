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

## The argus-iris GPU broker

[argus-iris](https://github.com/smk762/argus-iris) is the suite's GPU lease
broker. It decides whether a GPU job may start now, on which node, and what
must be unloaded first. It never runs the work. With
`ARGUS_GPU_COORDINATOR=broker`, curator asks iris for each lease instead of
taking the host lock (argus-cortex's `BrokerCoordinator`, from cortex 0.6):

| Variable | Value |
|---|---|
| `ARGUS_GPU_COORDINATOR` | `broker` |
| `ARGUS_GPU_BROKER_URL` | `http://<GPU_HOST>:8120` |
| `ARGUS_GPU_BROKER_TOKEN` | the suite's shared caller token. Put it in the host's `.env`, never in a repo. |
| `ARGUS_GPU_TENANT_MANIFEST` | `/app/gpu-tenant.yaml` in the image (compose's default), or set `ARGUS_GPU_TENANT=argus-curator` instead |
| `ARGUS_GPU_BROKER_FALLBACK` | `lease` (default) or `none` |

[`gpu-tenant.yaml`](../gpu-tenant.yaml) declares curator's kinds of GPU work:
`faces`, `scenes` and `garments`, with each one's footprint from
`runtime.VRAM_MB` (`vram_mb` is the load ask plus one image's run ask,
`model_mb` the load ask). Each lease names its model as the kind. Scans take
the kinds' `batch` priority. `POST /faces`, which darkroom waits on, asks for
`interactive`. A test checks that the manifest validates as cortex's
`TenantManifest`. The manifest pins `node` to the host curator runs on.
Change it if curator runs on another node.

The broker client fails open. If iris cannot be reached or answers 5xx, the
lease comes from `ARGUS_GPU_BROKER_FALLBACK`: the host lock described above, so
compose's lock mount is still needed. Today iris has no lease routes, so every
lease falls back. A refusal from iris (`GpuLeaseRefused`) fails a scan or
`POST /faces` with the same 503 and `Retry-After: 5` as a timeout.
`POST /faces` caps both the broker's wait and its fallback's at 30 s.
`GET /health` reports `gpu_coordinator: broker`.

Curator does not offer the resident contract yet (`/admin/unload|wake`), so
iris cannot unload its models to free memory.
