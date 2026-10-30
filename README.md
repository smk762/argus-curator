# Argus Curator

[![PyPI](https://img.shields.io/pypi/v/argus-curator)](https://pypi.org/project/argus-curator/)
[![Python](https://img.shields.io/pypi/pyversions/argus-curator)](https://pypi.org/project/argus-curator/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/smk762/argus-curator/actions/workflows/ci.yml/badge.svg)](https://github.com/smk762/argus-curator/actions/workflows/ci.yml)

Image curation and moderation: score, de-duplicate, and tag images by face, apparent age, scene and garment, then filter on those tags with a reason recorded for every exclusion. Use it on its own to audit or moderate an image collection, or as the LoRA data-prep front-end of the Argus suite, captioned downstream by [argus-lens](https://github.com/smk762/argus-lens).

Two ways to use it:

- **Standalone moderation and dataset auditing.** Scan a folder or call the
  HTTP API, review the per-image report, and keep, drop or quarantine images
  on criteria you choose. Nothing has to be trained or captioned. See
  [Standalone: moderation and dataset auditing](#standalone-moderation-and-dataset-auditing).
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

## Standalone: moderation and dataset auditing

Everything curator records about an image is usable without the rest of the
suite. A scan never modifies the source folder; it produces a per-image record
(`--json`, `--csv`, or `GET /scan/{scan_id}`), and an export applies filters
to it, giving every excluded image a `keep_reason`. That record doubles as an
audit trail: you can show which images were removed and on what grounds.

| Check | What is recorded | Filter on it |
|---|---|---|
| Technical quality | sharpness, resolution, aspect ratio, compression artifacts | `--min-score`, hard-filter `reject_reason` |
| Near-duplicates | pHash groups, best representative kept | `--keep-similar` to keep the rest |
| People | face count, identity clusters, head pose | `--face-clusters`, `--pose` |
| Apparent age | MiVOLO age in years and its bracket, per face (needs `mivolo-convert`) | `--min-age` / `--max-age`, `--age` |
| Apparent gender | MiVOLO gender with its probability, pooled per identity | `--gender`, `--min-attr-conf` |
| Setting | Places365 scene top-k, indoor/outdoor | `--scene`, `--environment` |
| Clothing | Fashionpedia garments: type, colour, cut off at the frame edge | `--garment`, `--exclude-garment`, `--uncropped` |
| Nudity | NudeNet detections per body-part class, exposed score (needs a [NudeNet server](#nudity-detection-nudenet)) | `--nsfw-threshold`, `--nsfw-label` |
| Policy | SingGuard verdict and the rules that fired: sexual content, crime, gore and self-harm, … (needs a [SingGuard server](#policy-check-singguard)) | `--guard-unsafe`, `--guard-rules` |

A typical moderation pass is scan, review, act:

```bash
# 1. Scan and report only: nothing is copied, moved or deleted.
argus-curator scan /data/inbox --mivolo --scenes --garments --csv report.csv --json scan.json

# 2. Review report.csv: one row per image with every tag and its confidence.

# 3. Keep only images whose primary face reads as clearly adult. --min-score 0,
#    --include-rejected and --keep-similar stop quality and duplicates from
#    removing anything, so age is the only criterion. Images where no age could
#    be estimated (no face found, or too blurry to pass the hard filters that
#    face detection runs behind) are excluded as `face-attributes-missing`: the
#    filter fails closed rather than letting unknowns through.
argus-curator scan /data/inbox --mivolo --min-age 25 \
    --min-score 0 --include-rejected --keep-similar \
    --copy-to /data/approved --csv decisions.csv
```

`decisions.csv` then lists every image with `keep` and its `keep_reason`
(`face-age-out-of-range`, `face-attributes-missing`, `garment-excluded`, …),
which is the list to send for human review or to quarantine. The same scans and
filters are available over HTTP (`POST /scan/folder`, `GET /scan/{scan_id}`,
`POST /export`), so curator can sit in front of an upload or moderation queue.

**Limits — read before relying on it for moderation:**

- **Content checks need their servers, and both miss things.** NudeNet
  detects exposed body parts only, and misses stylised, partial or unusual
  poses. SingGuard judges a whole policy (violence, gore, self-harm, …) but
  gives a verdict without a confidence, so nothing is "near the threshold" to
  review. Set NSFW thresholds conservatively and have a person review flagged
  and borderline images.
- **Age and gender are model estimates from appearance**, not facts
  about anyone. Age estimates are typically off by a few years, more so from
  the face alone (the default). `--mivolo-body` adds each face's body from
  InsightFace's person detector; a person with no face still gets a body-only
  estimate. Filter with a margin above any legal threshold (`--min-age 25`
  rather than 18) and send borderline images to a person.
- Only the **primary** face of each image is filtered on; other faces are
  recorded per face (`faces[].attributes`) but do not exclude the image.
- Scene and garment models make mistakes too; see the measured accuracy in
  [Scene tags](#scene-tags-places365) and [Garments](#garments-fashionpedia).

Curator is a filter and an audit trail to put in front of human review, not a
replacement for it.

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

# Predict age in years and gender per face (MiVOLO), then keep adults up to 35
# whose gender prediction is confident
argus-curator scan /data/images --mivolo \
    --gender female --min-attr-conf 0.6 --min-age 25 --max-age 35 --copy-to /data/curated

# Which detectors are available?
argus-curator detectors
```

### Face attributes (MiVOLO)

[MiVOLO v2](https://huggingface.co/iitolstykh/mivolo_v2) (`--mivolo`, which
implies `--faces`) reads each detected face:

| | MiVOLO v2 |
|---|---|
| Predicts | age in **years**, gender |
| Confidence | gender probability; age is a regression (none) |
| Input | 384 px face box, letterboxed; plus the body with `--mivolo-body` |
| Weights | Apache-2.0, Hugging Face |

The age bracket (0-2, 3-9, 10-19, …, 60-69, 70+) is derived from the years,
for the `--age` filter. `gender_source` / `age_source` name the model that
produced each value. Results land per face (`faces[].attributes`),
on the primary face (`primary_face_attributes`, also in the CSV report), and
pooled per identity cluster (`face_clusters[].attributes`): probabilities are
averaged and ages take the median over every member face, so one odd frame
cannot flip or drag an identity's label.

MiVOLO is built to read a face *and* a person crop, and its published accuracy
is for the pair. By default it runs face-only (a supported mode, slightly less
accurate). `--mivolo-body` (`"faces": {"mivolo_body": true}`; `POST
/faces?body=true`) gives each face its body from InsightFace 2.1's person
detector, the one model of the `cheetah_l` pack (30 MB; fetched to
`~/.insightface` on first use). Under evaluation, so off by default.

- Faces get bodies by one-to-one assignment, scoring each pair as the mean of
  their IoU and the share of the face inside the body (`body_face_match`,
  0.0001).
- As MiVOLO does, the body crop has every face on it blacked out (its own too:
  the face channel carries that) and every other person on it, with a masked
  person edge within 30% of the crop's edge stretched to it.
- A body under `body_min_score` (0.5) or `body_min_size` (50 px), or with less
  than `body_min_visible` (40%) left after masking, is not used, and that face
  runs face-only. In a packed group photo most bodies fail that last test, as
  they do in MiVOLO. The thresholds are request fields beside `mivolo_body`,
  so a cached scan records the ones its ages were read with.
- `faces[].attributes.age_body` says whether an age was read with a body.
- Person boxes land on `results[].people` (bbox, the matched `face_index`, the
  age). A person with no face still gets a body-only MiVOLO age (absent face
  channels). NudeNet detections are assigned to the overlapping person, else
  face, so `minor-exposed` can tell a younger person from an adult in the
  same frame.

`scripts/age_eval.py labels.csv` compares the two modes on images of known
age: mean and median error, share within 5 years, and how many subjects under
18 were read as 18 or over.

InsightFace loads only the models curator reads: detection, recognition, and
the 3D-68 landmarks behind head pose. The pack's own gender/age model (MiVOLO
replaces it) and its 106-point landmarks are skipped, about 28% less time per
image on CPU with identical embeddings and pose. The Cheetah models carry
InsightFace's non-commercial licence, like `buffalo_l`.

These are a model's guess from appearance, not facts about anyone; expect
errors on ambiguous faces, heavy makeup or poor lighting, and check
`gender_conf` before relying on a gender.

**Filters** (all on the primary face): `--gender`, `--age` (brackets),
`--min-age` / `--max-age` (years). `--min-attr-conf` /
`min_attribute_conf` additionally requires a filtered label's confidence to
reach the floor (a MiVOLO age has none, so it is not held to it). Excluded
images say why in `keep_reason`: `face-<attr>-excluded`,
`face-<attr>-low-confidence`, `face-<attr>-missing`,
`face-age-out-of-range`, `face-age-years-missing`, or
`face-attributes-missing`.

**The model** is not on PyPI; it is converted to ONNX once, after which
inference runs on the `[faces]` onnxruntime stack with no torch:

```bash
# Downloads the pinned checkpoint (commit + sha256 checked)
pip install "argus-curator[mivolo-convert]"
argus-curator mivolo-convert          # -> ~/.cache/argus_curator/models/mivolo_v2_d1_384.onnx
```

Point elsewhere with `--mivolo-model` or `ARGUS_CURATOR_MIVOLO_MODEL`. Model
paths are server configuration, never part of a request; a server without the
model refuses the scan with 400 before scoring anything, and `GET /detectors`
reports `mivolo`.

#### An age for the frames with no face

MiVOLO reads a face, so a frame that shows none has no age, and in a set
those are often the most explicit frames. `--set-age` (or
`"faces": {"set_age": true}` on the request; implies `--mivolo`) gives each
passing frame with no face a `set_age` read off the set, when the scan looks
like one subject: the largest identity must hold at least 90% of the scan's
faces (`set_age_min_share`) and have at least 5 MiVOLO ages
(`set_age_min_faces`). Its median age is the `years`, with the `cluster_id`,
the number of `faces` it is read from and the identity's `share`. The summary
carries the same `set_age`, or `set_age_skipped` saying why none was given
(faces split across identities, too few ages), so a mixed folder gets nothing
rather than someone else's age. With `ARGUS_CURATOR_SINGGUARD_CONTEXT=age`
the frame's SingGuard context says that no face is visible and what the rest
of the set shows. It is an inference about the set, not a reading of the
frame: a viewer shows it as such, and where a wrong age is the costlier
mistake treats it as it treats a measured one. `/health` reports `set_age`.

### Scene tags (Places365)

`--scenes` (or `"scenes": {"enabled": true}` on the scan request) runs a
[Places365](https://github.com/CSAILVision/places365)-CNN (Zhou et al., TPAMI
2017) on every passing image and records its top-k scene categories (365 of
them: `beach`, `bedroom`, `gas_station`, `forest_path`, …) with their softmax
probabilities, plus an indoor/outdoor call. `environment_conf` is the
probability mass on categories of that environment, not upstream's top-10
vote. Results land on `results[].scene` and in the CSV report.

Top-1 accuracy on Places365 is about 55% (top-5 about 85%), and neighbouring
categories overlap (`lagoon` / `beach` / `coast`), which is why filters match
any of the recorded top-k rather than only the first:

- `--scene beach,coast` / `scenes`: keep images with one of these among their
  top-k at probability ≥ `--min-scene-conf` / `min_scene_conf` (default 0.1, so
  a label lingering in the tail at 3% does not count). Unknown labels are
  rejected with 422 rather than silently matching nothing.
- `--environment outdoor` / `scene_environments`: indoor or outdoor, held to the
  same floor.
- `--diversify-scenes` / `diversify_scenes` (with `--max-keep`): the diversity
  cap also spreads the selection across backdrops, so an identity set is not
  two hundred bathroom selfies.

Excluded images say why: `scene-excluded`, `scene-low-confidence`,
`scene-environment-excluded`, `scene-environment-low-confidence`, or
`scene-missing`.

The weights are converted once to ONNX, like the face models:

```bash
pip install "argus-curator[places365-convert]"   # torch, only for this step
argus-curator places365-convert                  # resnet18 (default) or --arch resnet50
# -> ~/.cache/argus_curator/models/places365_resnet18.onnx
```

Upstream serves the checkpoints over plain HTTP, so each is pinned by sha256.
ResNet-50 is twice the size for about one point of top-1 (55.6% vs 54.5%).
Point elsewhere with `--places365-model` / `ARGUS_CURATOR_PLACES365_MODEL`.

### Garments (Fashionpedia)

`--garments` (or `"garments": {"enabled": true}`) runs a YOLOS-small detector
fine-tuned on [Fashionpedia](https://fashionpedia.github.io/home/)
([`valentinafevu/yolos-fashionpedia`](https://huggingface.co/valentinafevu/yolos-fashionpedia),
MIT; dataset CC BY 4.0) on every passing image. Each detection records its
garment type (27 classes: `dress`, `pants`, `shoe`, `bag_wallet`, `hat`, …),
probability, box, whether the box touches the frame edge (likely cut off), and
a dominant colour from the box's centre. Results land on `results[].garments`
(`[]` = ran, found nothing) and in the CSV as `label:conf:#colour`, with `*`
marking a cut-off garment.

Measured on 300 Fashionpedia validation images (CPU, ~0.3 s/image):

| | |
|---|---|
| Garments, AP50 / mAP | 0.327 / 0.193 |
| Strong (P / R at 0.5) | shoe 0.96 / 0.97 · pants 0.83 / 0.87 · dress 0.75 / 0.91 · hat 0.86 / 0.78 · jacket 0.86 / 0.58 |
| Weak | skirt 0.38 / 0.31 · coat 0.43 / 0.15 · shirt/blouse recall 0.31 |
| "Is garment X in the image?", all classes | P 0.64 / R 0.76 at 0.3 · **0.77 / 0.66 at 0.5** · 0.86 / 0.56 at 0.7 |

Fashionpedia's 19 garment *parts* (sleeve, pocket, zipper, …) scored AP50
0.105 and are not recorded; neither are its 294 fine-grained attributes, which
no published checkpoint predicts. Detections are recorded from 0.3 so reviewers
see the doubtful ones; filters count them from `--min-garment-conf` (0.5). A
floor below what the scan recorded cannot be honoured, so the server refuses it
with 400; the CLI instead records from the lower floor in the first place.

- `--garment dress,skirt` / `garments`: wearing at least one of these.
- `--exclude-garment bag_wallet,umbrella` / `exclude_garments`: none of these.
  Exclusion wants recall, so consider a lower `--min-garment-conf` for it.
- `--uncropped` / `garments_uncropped`: the matching garment is not cut off.
- `--diversify-outfits` / `diversify_outfits` (with `--max-keep`): the
  diversity cap also spreads the selection across outfits (garment types and
  their colours), so an identity LoRA does not learn one outfit as part of the
  person.

Exclusions say why: `garment-not-found`, `garment-low-confidence` (present,
below the floor), `garment-excluded`, `garment-cropped`, `garments-missing`.

```bash
pip install "argus-curator[garments-convert]"   # torch + transformers, only for this step
argus-curator garments-convert                  # pinned commit + sha256
# -> ~/.cache/argus_curator/models/yolos_fashionpedia.onnx
```

Point elsewhere with `--garments-model` / `ARGUS_CURATOR_GARMENTS_MODEL`.

### Nudity detection (NudeNet)

Curator can check every scanned image with [NudeNet](https://github.com/notAI-tech/NudeNet),
a detector for 18 body-part classes (`FEMALE_BREAST_EXPOSED`, `BUTTOCKS_COVERED`,
`FACE_FEMALE`, …). It talks to a **NudeNet server over HTTP**; curator never
imports NudeNet, so it gains no dependency and stays MIT. NudeNet itself is
AGPL-3.0, so running its server is a choice for whoever deploys it.

Configure the server in the environment (or `.env`), never per request:

```bash
ARGUS_CURATOR_NUDENET_URL=http://192.168.1.50:8080   # its origin; /infer is appended
ARGUS_CURATOR_NUDENET_TIMEOUT=30                     # seconds per image (default 30)
ARGUS_CURATOR_NUDENET_WORKERS=4                      # concurrent requests (default 4)
```

Any host running `ghcr.io/notai-tech/nudenet` works, including one elsewhere on
your LAN. `docker compose up` starts one alongside curator (pinned by digest)
and uses it unless `.env` says otherwise.

**It is opt-out.** When a URL is configured, every scan checks its images,
including ones that failed the quality filters, unless the scan opts out
(`--no-nsfw`, or `"nsfw": {"enabled": false}`). `--nsfw` / `"enabled": true`
makes the check mandatory: the scan is refused if no server is configured. Set
the URL empty to switch the check off for a deployment. The scan summary
records whether it ran (`nsfw_config.enabled`). Images are sent to that server,
so keep it on infrastructure you control.

Each image records every detection, the highest score per class
(`nsfw.labels`), and an **exposed score**: the highest score among the five
exposed-nudity classes (`BUTTOCKS_EXPOSED`, `FEMALE_BREAST_EXPOSED`,
`FEMALE_GENITALIA_EXPOSED`, `MALE_GENITALIA_EXPOSED`, `ANUS_EXPOSED`). Faces and
covered body parts are detected too but never count towards it. The CSV
report has `nsfw_exposed_score` and `nsfw_labels`.

Filters drop an image when a score is **at or above** its threshold:

```bash
# Total: drop anything whose exposed score is 0.5 or more
argus-curator scan /data/inbox --nsfw-threshold 0.5 --copy-to /data/clean

# Per class, alongside or instead of the total
argus-curator scan /data/inbox \
    --nsfw-label FEMALE_BREAST_EXPOSED=0.4,MALE_GENITALIA_EXPOSED=0.3,BELLY_EXPOSED=0.8 \
    --copy-to /data/clean
```

Over HTTP these are `nsfw_threshold`, `nsfw_label_thresholds` (`{"CLASS": score}`)
and `nsfw_exposed_labels` (which classes the total is taken over). The server
reports nothing under 0.2, so thresholds must be 0.2–1. An image that was never
checked, or whose check failed, is **excluded** whenever an NSFW filter is set
(`nsfw-unchecked`): the filter fails closed. Other reasons are `nsfw-exposed`
and `nsfw-<class>` (e.g. `nsfw-belly-exposed`). Exporting with NSFW thresholds
from a scan that did not run NudeNet is refused (400) rather than excluding
everything. `GET /detectors` reports `nudenet` when a URL is configured.

Each checked image also carries a **`content_rating`**, read off the same
detections: `explicit` when an exposed class was reported (the five above, from
NudeNet's 0.2 floor), else `suggestive` when a covered intimate class was
(lingerie, swimwear: covered breast, genitalia, buttocks or anus), else `safe`.
An unchecked image has none. It is on every result and in the CSV report, and
`content_ratings` (`--content-ratings safe,suggestive`) keeps only those
ratings; unchecked images are excluded (`nsfw-unchecked`), others get
`content-rating:<rating>`. The rating always uses the five default exposed
classes; `--nsfw-exposed-labels` / `--nsfw-label` can still drop an image
the rating calls `safe` (for example a high `BELLY_EXPOSED`).

### XMP sidecars

`write_xmp: true` (`--xmp`) writes an XMP sidecar beside each exported file,
`photo.jpg.xmp`, through argus-cortex's shared writer, the one argus-lens uses:

- `mwg-rs:Regions`: every detected face, with the name given for its cluster.
- `Iptc4xmpExt:PersonInImage`: the names, from `face_names`
  (`--face-names face_1=Ada,face_3=Ada`; two clusters can be one person). An
  unnamed cluster gets an unnamed region and no person entry.
- `argus:ContentRating`: the content rating above.

The source's own sidecar (lens's captions, a photo app's stars and edits) is
the starting point and everything curator does not set is kept; the source's
sidecar itself is never changed. Age estimates are never written: a MiVOLO
guess in a photo library reads as a fact. A file curator knows nothing about
gets no sidecar; `xmp_written` counts the ones written. A transfer can succeed
while a sidecar fails (`copied` greater than `xmp_written`); the image still
landed, and the failure is logged rather than undoing it.

`--xmp-sources` (CLI only) merges the same fields into the sidecars beside
**every scanned original** curator has something for — including rejected
images — independent of `--xmp` / `--copy-to`. That writes into the scanned
folder; the server's write paths stay under the export root. Regions are
replaced as a whole, so regions drawn by hand on an image with detected faces
are lost.

### Policy check (SingGuard)

Where NudeNet scores nudity per body part, a [SingGuard](https://huggingface.co/inclusionAI/SingGuard-4b-GGUF)
server judges each image against a whole moderation policy and answers with a
**verdict and the rule that fired**. Its default rules:

| Rule | Covers |
| --- | --- |
| A. Sexual Content Risk | sexual content |
| B. Real-World Crimes & Public Safety | violent crime, weapons, public-safety threats |
| C. Unethical Behavior | hate, harassment, self-harm, gore and shock content |
| D. Cybersecurity | data leaks, hacking, surveillance abuse |
| E. Agent Safety | attempts to expose model safeguards |
| F. Politically Sensitive | political advocacy, unrest, attacks on political figures |
| G. Animal Abuse | cruelty to animals |

It gives **no scores**, so it complements NudeNet rather than replacing it. It
runs a vision language model: about a second per image in `fast` mode, several
in `fast-slow` (which also writes why a rule fired). The default is `cascade`
(`fast` for every image, `slow` for what `fast` flags). `cascade` and `slow`
are modes of argus-cull's and agent-composer's `singguard` services; a stock
SingGuard server answers 400 to them. Curator then retries in `fast`, and once
`fast` works it judges the rest of the run in `fast` and logs
`singguard_mode_unsupported`; each such verdict records `thinking: "fast"`.
Configure it like NudeNet:

```bash
ARGUS_CURATOR_SINGGUARD_URL=http://192.168.1.198:9150   # its origin; /api/guard/check is appended
ARGUS_CURATOR_SINGGUARD_THINKING=cascade                 # or fast / fast-slow / slow
ARGUS_CURATOR_SINGGUARD_POLICY_FILE=/etc/curator/policy.txt  # optional: replaces the default rules
ARGUS_CURATOR_SINGGUARD_TIMEOUT=90                       # seconds per image (default 90)
ARGUS_CURATOR_SINGGUARD_WORKERS=2                        # concurrent requests (default 2)
ARGUS_CURATOR_SINGGUARD_CONTEXT=age,folder,caption,scene # optional: what the scan knows, sent as request text
```

By default SingGuard sees the image alone, and under the adult policies it
then reasons about an age and a consent it cannot see. `ARGUS_CURATOR_SINGGUARD_CONTEXT`
sends what the scan already knows as the request text, opening with a line
that says it is recorded context and not a request (SingGuard reads a bare
description as a request to produce it): `age` is each face's MiVOLO estimate
(and a body-only estimate when no face matched); `folder` is the image's folder
under the scanned root; `caption` is an argus-lens `<stem>.txt` or XMP sidecar
(`photo.jpg.xmp`, else Lightroom's `photo.xmp`), cut to 600 characters;
`scene` is Places365; `nsfw` is the NudeNet content rating; `garments` is
Fashionpedia. Compose defaults to `age,folder,caption,scene`. The library
sends nothing until the variable is set. A wrong
age leans the verdict the wrong way, so measure it on labelled images
(`scripts/guard_eval.py --images`) before relying on it. The under-18 rule
(`minor-exposed`) does not read this text.

It is opt-out in the same way (`--no-guard`, `"guard": {"enabled": false}`;
`--guard` makes it mandatory). The default rule A flags any sexual content,
adult nudity included. For an adult dataset, pick the built-in **`adult`**
policy per scan (`--guard-policy adult`, `"guard": {"policy": "adult"}`). Its
rule A covers only apparent minors, non-consent and sexual violence, and its
Safe rule names consensual adult nudity, SingGuard's own "Adult Content
(Non-criminal)" category, a single adult posing for the camera, and that
consent is not read from the image: a posed, camera-aware adult counts as
consenting unless hidden recording, restraint, coercion or an unconscious
person is shown. What passes is said only under Safe, never as a "not this
rule" line under a risk heading, which SingGuard reads as a match. Rules B to G cover the same areas
under the same letters, so `--guard-rules B,C` means the same thing under
both. They are written out as text, though, rather than left to SingGuard's
built-in wording, so verdicts can differ: a group photo of a poker game that
the default rule B flags passes under `adult`. `default` is the policy file when one is set, else SingGuard's own
rules. A request names a policy and never sends policy text. `GET /detectors`
lists the names in `singguard_policies`, and `GET /guard/policies` serves each
one's text: its `source` (`builtin`, `file`, or `singguard` when `default`
leaves SingGuard on its own rules and `text` is null) and the name results
record it under (`recorded`), so a verdict can be traced to its wording. The
policy file's text is served too, to anyone who can read from the server.

`adult-granular` (a letter per hard case: A minors, H non-consent, I sexual
violence) and `adult-cautious` (doubt about age or consent counts as rule A)
are under evaluation against `adult`; don't build on them yet.

To compare policies on your own images, judge one scan under several:
`"guard": {"policy": "adult", "variants": ["adult-granular", "adult-cautious"]}`
(CLI `--guard-variants`). `guard` stays the official verdict, and
`guard_variants` holds every policy's verdict by name, the official one
included. A missing name means that check failed. Each variant is one more
SingGuard pass per image. The filters read the official verdict unless
`guard_policies` lists others (CLI `--guard-policies`). An image is then
dropped if any listed verdict fails, or is missing, with the policy named in
the keep reason (`guard-unsafe:adult-granular`). The CSV report adds a
`guard_variants` column.
`scripts/guard_eval.py --url <singguard>` scores every policy on the text
probes in `eval/guard_probes.jsonl` and on an optional folder of images
sorted by expected verdict.

Each image records `guard.safe`, the rules that fired, the explanation, the
model and the policy: `default`, `adult`, or `custom-<sha256 prefix>` of the
file. The CSV report has `guard_safe` and `guard_rules`.

```bash
# Drop anything judged unsafe
argus-curator scan /data/inbox --guard-unsafe --copy-to /data/clean

# Drop only crime and unethical-behaviour verdicts, by letter or full name
argus-curator scan /data/inbox --guard-rules B,C --copy-to /data/clean
```

Over HTTP these are `guard_unsafe` and `guard_rules`. As with NudeNet, an
unchecked image is excluded whenever a SingGuard filter is set
(`guard-unchecked`); other reasons are `guard-unsafe` and `guard-rule-<rule>`
(e.g. `guard-rule-c`), or `guard-rule-unknown` for an unsafe verdict that names
no rule. `GET /detectors` reports `singguard`.

### HTTP server (:8101, peer to argus-lens)

```bash
pip install "argus-curator[server,faces]"
argus-curator serve --cors --port 8101 --source-root /data/images --export-root /data/out
```

Request-supplied paths are treated as untrusted: scan folders resolve under
`--source-root` (env: `ARGUS_CURATOR_SCAN_ROOT` / `CURATOR_SOURCE_PATH`) and
export destinations under `--export-root` (`ARGUS_CURATOR_EXPORT_ROOT` /
`CURATOR_EXPORT_PATH`). Traversal escapes are refused, and scan/export refuse
outright when their root is not configured. Both roots are needed to export:
the destination resolves under the export root, and the scan being exported
must itself lie under the source root — a `scan_id` is data from a store the
CLI also writes to, so its folder is re-checked rather than trusted (the same
check applies to `/thumb?scan_id=`). The destructive `mode: "move"` is
rejected unless the server is started with `--allow-move`
(`CURATOR_ALLOW_MOVE=1`). `--cors` allows only the localhost dev frontend
(:3000); allow-list other origins with `--cors-origin <origin>`
(`CURATOR_CORS_ORIGINS`), or use `--cors-any` for a credential-less wildcard
on public demos. A literal `*` in the allow-list is treated as `--cors-any`
(wildcard without credentials), never as origin reflection.

CORS is not a write boundary — `POST /upload` sends `multipart/form-data`, which
browsers deliver with no preflight — so state-changing requests are additionally
gated on `Origin`: it must be absent (curl, the CLI, server-to-server),
same-origin, or on the allow-list, else 403. `--cors-any` therefore permits
anonymous reads from anywhere but no cross-site writes; name an origin with
`--cors-origin` to let it write.

| Route | Description |
|---|---|
| `GET  /health` | Liveness. `faces` is true when `POST /faces` can run (InsightFace and MiVOLO). |
| `GET  /detectors` | `{ torch, cuda, clip, insightface, onnxruntime, mivolo, bodies, places365, garments, nudenet, singguard, … }` |
| `POST /faces` | One image in memory (raw bytes or multipart). `{ source: mivolo, faces: [{ age_estimate_years, bbox }] }`. Writes nothing. |
| `GET  /guard/policies` | Each SingGuard policy a scan can name: `{ singguard, policies: [{ name, recorded, source, text, error }] }` |
| `GET  /folders?path=<rel>` | Browse Docker-mounted folders under the source root (for the UI picker) |
| `POST /scan/folder` | Scan + score + dedup + face-cluster → `ScanSummary` |
| `GET  /scan/{scan_id}` | Cached summary, paginated via `?offset=&limit=` |
| `GET  /thumb?path=<rel>&scan_id=<id>` | `image/webp` thumbnail from the mount |
| `GET  /image?path=<rel>&scan_id=<id>` | Original image bytes, same containment as `/thumb` |
| `POST /scan/{scan_id}/select` | Dry-run of the export filter → `{total, kept, rows: [{rel_path, abs_path, keep_reason}]}`. Writes nothing |
| `POST /upload` | Multipart image upload (`files` + `folder`) into a folder under the source root |
| `POST /export` | Structure-preserving transfer + manifest → `ExportResult` |
| `POST /scan/folder/stream` | Same as `/scan/folder`, streaming live progress over SSE |
| `POST /export/stream` | Same as `/export`, streaming per-file transfer progress over SSE |
| `GET  /scans` | Cached scans, newest first, plus `active`: the one running, with its phase, counts, `pending_checks` and `cancelling` |
| `POST /scan/{scan_id}/cancel` | Stop the running scan (404 if that is not the one running). The stream ends with `event: cancelled`; the partial summary is dropped |
| `POST /scans/purge` | Stop the running scan, if any, wait for it to let go, and delete every cached scan → `{cancelled, deleted}`. Source images are not touched |

The `*/stream` variants emit `event: progress` frames (`{phase, done, total}`)
while the work runs, then a single `event: complete` frame carrying the same
payload the non-streaming endpoint returns (or `event: error`). A scan runs
`scoring` → `faces` and `nsfw` together → `identities` → `scenes` →
`garments` → `guard` → `clustering`: ages and the exposed score come first, so
a viewer can lift a blur as soon as both are known, and their progress events
interleave while both run; identities follow at once, so a set age is known
before the guard pass reads it as context. Scan progress and `event: chunk` frames carry
`pending_checks`, the checks among `faces`, `nsfw` and `guard` whose results
have not all landed yet; a check that could not run is not listed.

A scan stops cooperatively: `POST /scan/{id}/cancel` raises a flag the
scanner reads before each image and between phases, the checks already with
a model or a remote server finish, the ones still queued are dropped, and the
slot frees. Until then `GET /scans` reports the scan as `cancelling`. In the
library, pass `cancelled=` (a callable returning True to stop) to
`scan_folder()`; it raises `ScanCancelled`. `GET /health` says
`scan_control: true` on a server that has these routes.

`POST /scan/folder` body:

```jsonc
{
  "folder": "shoots/2026-07",         // relative to the server's --source-root ("" = the root)
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
  "dest": "training-set-v1",        // relative to the server's --export-root
  "mode": "copy",                   // "copy" | "symlink" | "move" (move needs --allow-move)
  "preserve_structure": true,
  "min_score": 0.6, "include_rejected": false, "keep_similar": false,
  "face_clusters": ["face_2"],      // optional: export only these identities
  "face_poses": ["frontal", "three_quarter"],  // optional: export only these head poses
  "face_genders": null,             // optional: gender filter on the primary face
  "face_ages": null,                // age brackets ("20-29", ...), from MiVOLO's years
  "min_age_years": 18, "max_age_years": 35,     // optional: MiVOLO age range
  "min_attribute_conf": 0.6,        // filtered attributes must be at least this confident
  "scenes": ["beach", "coast"],     // optional: Places365 scenes among the image's top-k
  "scene_environments": ["outdoor"],
  "min_scene_conf": 0.1, "diversify_scenes": false,
  "garments": ["dress"], "exclude_garments": ["bag_wallet"],  // optional: Fashionpedia
  "min_garment_conf": 0.5, "garments_uncropped": false, "diversify_outfits": true,
  "write_manifest": true,
  "caption_url": null               // optional: POST manifest to argus-lens
}
```

### Docker

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

### Sharing the GPU

On CUDA, curator takes argus-cortex's host-wide GPU lease (`argus_cortex.gpu`)
around each image it runs a model on: InsightFace with MiVOLO,
Places365 (per batch) and Fashionpedia, plus a larger ask when a model loads.
It is what the GPU image (`:<version>-gpu`, the compose default) runs.
The lease is taken per image, so argus-lens, or a `POST /faces` from
argus-darkroom, can use the card between a scan's images. It is never taken on
CPU, or when `device="cuda"` but onnxruntime has no CUDA provider (sessions
would only run on CPU anyway). InsightFace's pack download happens before the
load lease, so a cold start does not hold the GPU while weights hit the disk.

- `ARGUS_GPU_COORDINATOR=lease` turns it on (`none`, the default, gates nothing).
  Every argus process on the host must name the same `ARGUS_GPU_LEASE_PATH` on a
  shared directory. Containers have their own `/tmp`, so mount one; see the
  commented lines in `docker-compose.yaml`.
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

## Handoff to argus-lens

Export writes a JSONL manifest (one row per **exported** image — rows exist
only for files whose transfer actually succeeded):

```jsonc
{ "manifest_version": "2.1", "rel_path": "...", "abs_path": "...",
  "exported_path": "...", "exported_abs_path": "...", "target_profile": { ... },
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

`exported_abs_path` is that file's absolute location, and `exported_abs_paths`
the matching mapping on the export response. Prefer them: joining the export
root onto `exported_path` means reconstructing the server's layout rule
somewhere that does not own it, and the de-collision suffix above is not
reproducible by a client anyway. A manifest is also frequently read somewhere
other than the root it was written into — argus-lens receives it as an upload —
and nothing in a row names that root.

`abs_path` is where the image can be **read**, which is not always where it came
from: `mode: "move"` deletes the source during transfer, so for a moved export
`abs_path` equals `exported_abs_path`. Use `rel_path` for the original location
within the scan.

`GET /health` reports `manifest_version` alongside the package `version`, so a
consumer can check the contract it is about to speak — and refuse an unknown
major — rather than inferring it from which fields a response happens to carry.
The two versions move independently.

argus-lens batch-captions this manifest — categories are already shared, so no
remapping. Set `caption_url` on the export request to POST it straight to lens
for a one-click curate→caption run.

## How scoring works (per image)

1. **Hard filters** — min short side, max aspect ratio, blur (Laplacian-edge variance floor, measured with the long side at 1024px so the threshold does not depend on the file's resolution).
2. **Composite score** — target-aware weighted blend of sharpness / resolution / artifact, plus a small composition bonus that depends on `target_category` (e.g. identity rewards a single centred face; setting rewards wide framing).
3. **Face-count fit** — with `[faces]`, identity targets penalise 0 or 2+ faces; other categories are progressively more tolerant.
4. **Near-duplicate dedup** — pHash clustering keeps the highest-scoring representative and *reports* the rest (never silently dropped).
5. **Selection (at export)** — score threshold + optional diversity cap (`max_keep` / `diversity_weight`) + optional face-cluster filter. Every excluded image carries a `keep_reason`.

## State

Scans are cached on disk (keyed by `scan_id`, default `~/.cache/argus_curator/scans`,
override with `CURATOR_CACHE_DIR`). This is what makes paginated `GET /scan/{id}`
and export-by-id work without recomputing. `POST /scans/purge` empties it (and
stops a scan that is writing to it first); nothing else on disk is touched.

## Related projects

- [**argus-quarry**](https://github.com/smk762/argus-quarry) — provenance-first acquisition of public-domain / CC0 portraits (the suite's input stage).
- [**argus-lens**](https://github.com/smk762/argus-lens) — intent-aware, multi-model captioning (consumes the manifest this package exports).
- [**argus-studio**](https://github.com/smk762/argus-studio) — the suite's Next.js web UI (its `/curate` view drives this server).

## License

MIT — matches the rest of the Argus suite.
