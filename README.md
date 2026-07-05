# Argus Curator

[![PyPI](https://img.shields.io/pypi/v/argus-curator)](https://pypi.org/project/argus-curator/)
[![Python](https://img.shields.io/pypi/pyversions/argus-curator)](https://pypi.org/project/argus-curator/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/smk762/argus-curator/actions/workflows/ci.yml/badge.svg)](https://github.com/smk762/argus-curator/actions/workflows/ci.yml)

The LoRA data-prep front-end — curate by quality and by face, then caption with [argus-lens](https://github.com/smk762/argus-lens).

`argus-curator` is the **curation stage** of the Argus suite. Upstream,
[argus-quarry](https://github.com/smk762/argus-quarry) acquires provenance-clean
images; curator decides *which images, of whom, at what quality* belong in a
LoRA training set; argus-lens then decides *what's in each image*. Curator and
lens share one `TargetProfile`, so a manifest written here is captioned
downstream with no remapping.

```
   argus-quarry               argus-curator (:8101)      argus-lens (:8100)      imogen / kohya
   ─ download   ─┐            ─ scan + score ─┐          ─ caption ─┐            ─ train ─
   ─ provenance ─┤  images    ─ face-cluster ─┤ manifest ─ buckets ─┤  dataset   ─ LoRA  ─
   ─ verify ─────┴─────────►  ─ select ───────┴────────► (identity/ ─┴────────►  ───────►
                              ─ export ───────►          wardrobe/…)
```

## Why not FiftyOne / fastdup / Immich?

| | Generic CV curation (FiftyOne, fastdup) | Consumer galleries (Immich, PhotoPrism) | **Argus Curator** |
|---|---|---|---|
| Quality scoring | Generic | None | **Training-suitability** (target-aware sharpness/res/artifact + face-count fit) |
| Dedup | pHash | Basic | pHash, keeps best representative, **reports** the rest |
| Faces | Plugin | Clustering for browsing | Clustering **for dataset filtering** (export by identity) |
| Dataset export | Manual | None | Structure-preserving copy/symlink/move + **manifest** |
| Captioner handoff | None | None | One shared `TargetProfile` → argus-lens |

It's LoRA-native, identity-aware, and caption-integrated — the reason the suite exists.

## The shared TargetProfile (the moat)

Both services speak the same taxonomy. This single schema is the contract:

```python
from argus_curator import TargetProfile

TargetProfile(
    target_style="photo",        # "photo" | "anime"
    target_backend="sdxl",       # "sdxl" | "flux-dev-1" | ...
    checkpoint=None,
    target_category="identity",  # identity | wardrobe | pose_composition | setting
)
```

The curator uses it to weight scoring and label exports; argus-lens inherits it verbatim.

## Installation

```bash
pip install argus-curator                 # engine only (Pillow, numpy, ImageHash, pydantic)
pip install "argus-curator[server,cli]"   # FastAPI server + CLI
pip install "argus-curator[faces]"        # + InsightFace identity clustering (CPU onnxruntime)
pip install "argus-curator[gpu]"          # + onnxruntime-gpu for GPU face detection
pip install "argus-curator[all]"          # everything
```

System libraries for the face stack (Ubuntu/Debian):

```bash
sudo apt install -y libgl1 libglib2.0-0 libgomp1
```

## Usage

### Python

```python
from argus_curator import scan_folder, TargetProfile, ScanConfig, FaceConfig

summary = scan_folder(
    "/data/images",
    profile=TargetProfile(target_category="identity"),
    cfg=ScanConfig(min_short_side=512, blur_threshold=100.0, cluster_distance=10),
    faces_cfg=FaceConfig(enabled=True, model="buffalo_l", cluster_eps=0.5),
)

print(summary.passed, "passed,", summary.duplicates, "near-dupes")
for fc in summary.face_clusters:
    print(fc.cluster_id, fc.size, "faces, rep:", fc.representative_rel_path)
```

### CLI

```bash
# Report only — see the score distribution before committing to a threshold
argus-curator scan /data/images --csv report.csv

# Identity curation with face clustering, copy 0.65+ keepers (structure preserved)
argus-curator scan /data/images \
    --target-category identity --faces --device cuda \
    --min-score 0.65 --copy-to /data/curated

# Export only specific identities, capped to a diverse 200
argus-curator scan /data/images --faces \
    --face-clusters face_1,face_2 --max-keep 200 --copy-to /data/curated

# Pick a pose-balanced subset (head-on + 3/4 only, drop side profiles)
argus-curator scan /data/images --faces \
    --pose frontal,three_quarter --copy-to /data/curated

# Which detectors are available?
argus-curator detectors
```

### HTTP server (:8101, peer to argus-lens)

```bash
pip install "argus-curator[server,faces]"
argus-curator serve --cors --port 8101 --source-root /data/images
```

| Route | Description |
|---|---|
| `GET  /health` | Liveness |
| `GET  /detectors` | `{ torch, cuda, clip, insightface, onnxruntime }` |
| `GET  /folders?path=<rel>` | Browse Docker-mounted folders under the source root (for the UI picker) |
| `POST /scan/folder` | Scan + score + dedup + face-cluster → `ScanSummary` |
| `GET  /scan/{scan_id}` | Cached summary, paginated via `?offset=&limit=` |
| `GET  /thumb?path=<rel>&scan_id=<id>` | `image/webp` thumbnail from the mount |
| `POST /upload` | Multipart image upload (`files` + `folder`) into a folder under the source root |
| `POST /export` | Structure-preserving transfer + manifest → `ExportResult` |
| `POST /scan/folder/stream` | Same as `/scan/folder`, streaming live progress over SSE |
| `POST /export/stream` | Same as `/export`, streaming per-file transfer progress over SSE |

The `*/stream` variants emit `event: progress` frames (`{phase, done, total}`)
while the work runs, then a single `event: complete` frame carrying the same
payload the non-streaming endpoint returns (or `event: error`).

`POST /scan/folder` body:

```jsonc
{
  "folder": "/data/images",
  "target_profile": { "target_style": "photo", "target_category": "identity" },
  "config": { "min_short_side": 512, "max_aspect_ratio": 3.0, "blur_threshold": 100.0,
              "cluster_distance": 10, "max_workers": 4 },
  "faces":  { "enabled": true, "model": "buffalo_l", "min_det_score": 0.5, "cluster_eps": 0.5 }
}
```

`POST /export` body:

```jsonc
{
  "scan_id": "...",                 // or inline "selection": ["rel_path", ...]
  "dest": "/data/out",
  "mode": "copy",                   // "copy" | "symlink" | "move"
  "preserve_structure": true,
  "min_score": 0.6, "include_rejected": false, "keep_similar": false,
  "face_clusters": ["face_2"],      // optional: export only these identities
  "face_poses": ["frontal", "three_quarter"],  // optional: export only these head poses
  "write_manifest": true,
  "caption_url": null               // optional: POST manifest to argus-lens
}
```

### Docker

```bash
docker compose up --build
```

Bind-mounts a dataset into `/data/images`, an output dir into `/data/out`, and
persists the scan cache + InsightFace model downloads across rebuilds.

## Handoff to argus-lens

Export writes a JSONL manifest (one row per **exported** image — rows exist
only for files whose transfer actually succeeded):

```jsonc
{ "manifest_version": "2.0", "rel_path": "...", "abs_path": "...",
  "exported_path": "...", "target_profile": { ... },
  "primary_face_cluster": "face_2", "primary_face_pose": "three_quarter",
  "score": 0.87, "similar_group": 3 }
```

The row shape is published in the wire schema as `ManifestRow`.
`exported_path` is the path actually written under the export root — consumers
must use it rather than re-deriving a location from `rel_path`. Flattened
exports (`preserve_structure: false`) de-collide duplicate basenames with a
short hash suffix, so the two can differ; collisions are detected
case-insensitively (and Unicode-normalised) so the result is safe on
case-insensitive destination filesystems, and the export fails loudly rather
than overwrite if a unique name cannot be generated. The same
`rel_path -> exported_path` mapping is returned as `exported_paths` on the
export response, so it is available even with `write_manifest: false`. Note
that each export call plans in isolation: re-exporting into the same
destination overwrites files (and rewrites the manifest).

argus-lens batch-captions this manifest — categories are already shared, so no
remapping. Set `caption_url` on the export request to POST it straight to lens
for a one-click curate→caption run.

## How scoring works (per image)

1. **Hard filters** — min short side, max aspect ratio, blur (Laplacian-edge variance floor).
2. **Composite score** — target-aware weighted blend of sharpness / resolution / artifact, plus a small composition bonus that depends on `target_category` (e.g. identity rewards a single centred face; setting rewards wide framing).
3. **Face-count fit** — with `[faces]`, identity targets penalise 0 or 2+ faces; other categories are progressively more tolerant.
4. **Near-duplicate dedup** — pHash clustering keeps the highest-scoring representative and *reports* the rest (never silently dropped).
5. **Selection (at export)** — score threshold + optional diversity cap (`max_keep` / `diversity_weight`) + optional face-cluster filter. Every excluded image carries a `keep_reason`.

## State

Scans are cached on disk (keyed by `scan_id`, default `~/.cache/argus_curator/scans`,
override with `CURATOR_CACHE_DIR`). This is what makes paginated `GET /scan/{id}`
and export-by-id work without recomputing.

## Related projects

- [**argus-quarry**](https://github.com/smk762/argus-quarry) — provenance-first acquisition of public-domain / CC0 portraits (the suite's input stage).
- [**argus-lens**](https://github.com/smk762/argus-lens) — intent-aware, multi-model captioning (consumes the manifest this package exports).
- [**argus-studio**](https://github.com/smk762/argus-studio) — the suite's Next.js web UI (its `/curate` view drives this server).

## License

MIT — matches the rest of the Argus suite.
