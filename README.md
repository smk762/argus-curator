<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/smk762/argus-cortex/dev/brand/packages/curator/argus-curator-lockup-dark.svg">
    <img alt="Argus Curator" src="https://raw.githubusercontent.com/smk762/argus-cortex/dev/brand/packages/curator/argus-curator-lockup-light.svg" height="52">
  </picture>
</h1>

[![PyPI](https://img.shields.io/pypi/v/argus-curator)](https://pypi.org/project/argus-curator/)
[![Python](https://img.shields.io/pypi/pyversions/argus-curator)](https://pypi.org/project/argus-curator/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/smk762/argus-curator/actions/workflows/ci.yml/badge.svg)](https://github.com/smk762/argus-curator/actions/workflows/ci.yml)

Image curation and moderation: score, de-duplicate, and tag images by face, apparent age, scene and garment, then filter on those tags with a reason recorded for every exclusion. Use it on its own to audit or moderate an image collection, or as the LoRA data-prep front-end of the Argus suite, captioned downstream by [argus-lens](https://github.com/smk762/argus-lens).

Two ways to use it:

- **Standalone moderation and dataset auditing.** Scan a folder or call the
  HTTP API, review the per-image report, and keep, drop or quarantine images
  on criteria you choose. Nothing has to be trained or captioned. See
  [Standalone moderation and dataset auditing](docs/moderation.md).
- **The curation stage of the Argus suite**, described below.

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

A scan only reads the source folder (unless you ask for XMP sidecars beside
the originals). It records, per image, its quality
score, near-duplicate group, faces and identities, and whatever optional checks
you turn on: apparent age and gender, scene, garments, nudity and a moderation
policy verdict. An export then filters on that record and gives every excluded
image a `keep_reason`, so the result doubles as an audit trail.

## Why not FiftyOne / fastdup / Immich?

| | Generic CV curation (FiftyOne, fastdup) | Consumer galleries (Immich, PhotoPrism) | **Argus Curator** |
|---|---|---|---|
| Quality scoring | Generic | None | **Training-suitability** (target-aware sharpness/res/artifact + face-count fit) |
| Dedup | pHash | Basic | pHash, keeps best representative, **reports** the rest |
| Faces | Plugin | Clustering for browsing | Clustering **for dataset filtering** (export by identity) |
| Dataset export | Manual | None | Structure-preserving copy/symlink/move + **manifest** |
| Captioner handoff | None | None | One shared `TargetProfile` → argus-lens |
| Moderation / audit | Manual queries | None | Apparent-age, scene and garment filters, with a `keep_reason` for every exclusion |

It's LoRA-native, identity-aware, and caption-integrated — the reason the suite exists.

## The shared TargetProfile (the moat)

Both services speak the same taxonomy. This single schema is the contract:

```python
from argus_curator import TargetProfile

TargetProfile(
    target_style="photo",        # "photo" | "anime"
    target_backend="sdxl",       # "sdxl" | "flux-dev-1" | ...
    checkpoint=None,
    target_category="identity",  # identity | wardrobe | pose_composition | setting | concept
)
```

The curator uses it to weight scoring and label exports; argus-lens inherits it verbatim.
The style and category values are argus-cortex's (`argus_cortex.taxonomy`), the one
copy the whole suite shares. `concept` is a style or visual-concept LoRA: face count
never lowers its score, and clean, high-resolution images get the composition bonus.

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

Face attributes, scenes and garments read ONNX models that are converted once
(`mivolo-convert`, `fairface-convert`, `places365-convert`,
`garments-convert`); each feature's page says how. The nudity, policy and
watermark checks talk to their own servers over HTTP. A Docker image is
published for CPU and GPU: see [Docker and the GPU](docs/gpu.md).

## Quick start

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

# Predict age in years and gender per face (MiVOLO), then keep adults up to 35
# whose gender prediction is confident
argus-curator scan /data/images --mivolo \
    --gender female --min-attr-conf 0.6 --min-age 25 --max-age 35 --copy-to /data/curated

# Which detectors are available?
argus-curator detectors
```

The full list of commands and options is in the [CLI reference](docs/cli.md).

### HTTP server

```bash
pip install "argus-curator[server,faces]"
argus-curator serve --cors --port 8101 --source-root /data/images --export-root /data/out
```

Scans and exports then run over HTTP (`POST /scan/folder`, `POST /export`);
see the [HTTP API](docs/http-api.md).

## Documentation

**Using curator**

- [Standalone moderation and dataset auditing](docs/moderation.md): what each check records, a scan-review-act pass, and its limits
- [How scoring works](docs/scoring.md): hard filters, the composite score, near-duplicates, selection
- [CLI reference](docs/cli.md): every command and option (generated)
- [HTTP API](docs/http-api.md): the :8101 server, its routes, streaming, request bodies and scan cache
- [Docker and the GPU](docs/gpu.md): the CPU and GPU images, and sharing a card with the rest of the suite

**Checks and filters**

- [Face attributes](docs/face-attributes.md): MiVOLO age and gender, FairFace race, gender and age bracket, set age
- [Face sources](docs/face-sources.md): head pose, face sharpness, eyes and mouth, identity distance, the argus-mask preset
- [Scene tags](docs/scene-tags.md): Places365 scenes and indoor/outdoor
- [Garments](docs/garments.md): Fashionpedia garments, colours and cut-off
- [Nudity detection](docs/nudity.md): NudeNet, the EraX second pass, content rating
- [Policy check](docs/policy-check.md): SingGuard verdicts, policies and context
- [Prefilter](docs/prefilter.md): drop images before the slow checks
- [Crops](docs/crops.md): watermarks and borders, given or detected

**Export and handoff**

- [Manifest reference](docs/manifest.md): the `manifest.jsonl` contract, its fields, versions and compatibility rule
- [Sidecars and run stats](docs/sidecars.md): carrying captions and XMP with an export
- [XMP sidecars](docs/xmp.md): face regions, person names and content rating
- [Synthetic source portraits](docs/synthetic-face-sets.md): darkroom, curator and lens for argus-mask

## Handoff to argus-lens

Export writes `manifest.jsonl`, one row per exported image, carrying the
shared `TargetProfile`, the real exported path, and the primary face's
measurements. argus-lens batch-captions it with no remapping; set
`caption_url` on the export request to POST it straight to lens for a
one-click curate-then-caption run. The rows, their versions (currently 2.3)
and the rule for consumers (refuse an unknown major) are in the
[manifest reference](docs/manifest.md).

## Related projects

- [**argus-quarry**](https://github.com/smk762/argus-quarry) — provenance-first acquisition of public-domain / CC0 portraits (the suite's input stage).
- [**argus-lens**](https://github.com/smk762/argus-lens) — intent-aware, multi-model captioning (consumes the manifest this package exports).
- [**argus-studio**](https://github.com/smk762/argus-studio) — the suite's Next.js web UI (its `/curate` view drives this server).
- [**argus-darkroom**](https://github.com/smk762/argus-darkroom) — guarded image generation; its records carry with an export under `--carry-sidecars`.
- [**argus-mask**](https://github.com/smk762/argus-mask) — a talking face. [docs/synthetic-face-sets.md](docs/synthetic-face-sets.md) shows how darkroom, curator and lens make its synthetic source portraits.

## License

MIT — matches the rest of the Argus suite.
