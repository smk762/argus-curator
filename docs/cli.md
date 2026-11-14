# argus-curator CLI reference

LoRA-native dataset curation: score, dedup, face-cluster, export.

**Usage**:

```console
$ argus-curator [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `scan`: Scan a folder, rank for training...
* `serve`: Start the argus-curator micro-server...
* `detectors`: Report which optional detectors/backends...
* `fairface-convert`: Convert the FairFace PyTorch weights to...
* `mivolo-convert`: Fetch, verify and convert MiVOLO v2 to the...
* `places365-convert`: Fetch, verify and convert a Places365-CNN...
* `garments-convert`: Fetch, verify and convert the Fashionpedia...
* `schema`: Emit the wire-contract JSON Schema...

## `argus-curator scan`

Scan a folder, rank for training suitability, and optionally export keepers.

**Usage**:

```console
$ argus-curator scan [OPTIONS] {source}
```

**Arguments**:

* `source <path>`: Root folder to scan (recurses into sub-folders)  [required]

**Options**:

* `--min-score <float>`: Score threshold for keepers (0..1)  [default: 0.6]
* `--copy-to <path>`: Copy keepers here (structure preserved)
* `--move-to <path>`: Move keepers here instead of copying
* `--symlink-to <path>`: Symlink keepers here
* `--keep-similar`: Keep non-representative near-duplicates
* `--no-cluster`: Disable near-duplicate grouping
* `--cluster-distance <int>`: pHash Hamming distance for near-duplicates  [default: 10]
* `--max-keep <int>`: Cap keepers to a diverse N
* `--diversity-weight <float>`: 0=pure score, 1=pure spread  [default: 0.4]
* `--min-short-side <int>`: Minimum short-side pixels (512 at least)  [default: 512]
* `--max-aspect-ratio <float>`: Maximum long:short ratio  [default: 3.0]
* `--blur-threshold <float>`: Minimum Laplacian variance  [default: 100.0]
* `--max-workers <int>`: Parallel worker threads  [default: 4]
* `--target-style <str>`: photo or anime  [default: photo]
* `--target-backend <str>`: Diffusion backend family  [default: sdxl]
* `--checkpoint <str>`: Target checkpoint
* `--target-category <str>`: identity|wardrobe|pose_composition|setting|concept  [default: identity]
* `--faces`: Run InsightFace detection + identity clustering
* `--faces-model <str>`: InsightFace model name  [default: buffalo_l]
* `--min-det-score <float>`: Minimum face detection score  [default: 0.5]
* `--cluster-eps <float>`: Cosine-distance threshold for face clustering  [default: 0.5]
* `--device <str>`: auto|cpu|cuda for face detection  [default: auto]
* `--fairface`: Predict race/gender/age bracket per face with FairFace (implies --faces)
* `--mivolo`: Predict age in years + gender per face with MiVOLO (implies --faces)
* `--fairface-model <path>`: Converted FairFace ONNX model (ARGUS_CURATOR_FAIRFACE_MODEL)
* `--set-age`: Give frames with no face the set&#x27;s median MiVOLO age when one identity holds the scan (implies --mivolo)
* `--mivolo-body`: Also give MiVOLO each face&#x27;s body, from InsightFace&#x27;s person detector (implies --mivolo; under evaluation)
* `--mivolo-model <path>`: Converted MiVOLO ONNX model (ARGUS_CURATOR_MIVOLO_MODEL)
* `--scenes`: Tag each image&#x27;s scene + indoor/outdoor with Places365
* `--scene-top-k <int>`: Scene categories recorded per image  [default: 5]
* `--places365-model <path>`: Converted Places365 ONNX model (ARGUS_CURATOR_PLACES365_MODEL)
* `--garments`: Detect garments (type, colour, cut-off) with Fashionpedia
* `--garments-model <path>`: Converted garment-detector ONNX model (ARGUS_CURATOR_GARMENTS_MODEL)
* `--face-clusters <str>`: Comma-separated cluster ids to export
* `--face-reference <list[str]>`: An image in the scan (relative path) whose face is the identity reference; repeatable. Each face then gets a reference_distance (implies --faces)
* `--face-source`: Export only face-source images (one identity, frontal and level, large, sharp, eyes open, mouth closed, adult; needs --mivolo), ranked best first
* `--landmarks`: Write the primary face&#x27;s landmarks into the manifest
* `--pose <str>`: Export only these primary-face poses (comma-separated: frontal,three_quarter,profile)
* `--race <str>`: Export only these primary-face races (comma-separated: white,black,latino_hispanic,east_asian,southeast_asian,indian,middle_eastern)
* `--gender <str>`: Export only these primary-face genders (male,female)
* `--age <str>`: Export only these primary-face age brackets (comma-separated, e.g. 20-29,30-39)
* `--min-age <float>`: Export only primary faces at least this old (MiVOLO)
* `--max-age <float>`: Export only primary faces at most this old (MiVOLO)
* `--min-attr-conf <float>`: Minimum confidence for a filtered attribute to count as a match (0..1)  [default: 0.0]
* `--scene <str>`: Export only images showing one of these Places365 scenes (comma-separated, e.g. beach,coast)
* `--environment <str>`: Export only indoor or outdoor images (indoor,outdoor)
* `--min-scene-conf <float>`: Minimum probability for a scene / environment to count as a match  [default: 0.1]
* `--diversify-scenes`: With --max-keep, also spread the selection across scenes
* `--garment <str>`: Export only images wearing one of these garments (comma-separated, e.g. dress,skirt)
* `--exclude-garment <str>`: Drop images showing any of these garments (e.g. bag_wallet,umbrella)
* `--min-garment-conf <float>`: Minimum probability for a garment to count (filters + outfit diversity)  [default: 0.5]
* `--uncropped`: With --garment, the garment must not be cut off by the frame edge
* `--diversify-outfits`: With --max-keep, also spread the selection across outfits
* `--nsfw / --no-nsfw`: NudeNet nudity check (default: on when a NudeNet URL is configured; --no-nsfw opts out)
* `--nudenet-url <str>`: NudeNet server origin, e.g. http://nudenet:8080 (ARGUS_CURATOR_NUDENET_URL)
* `--erax-url <str>`: EraX-NSFW server origin, e.g. http://erax:8080 (ARGUS_CURATOR_ERAX_URL)
* `--watermark-url <str>`: Watermark detector origin, e.g. http://watermark:8080 (ARGUS_CURATOR_WATERMARK_URL)
* `--nsfw-threshold <float>`: Drop images whose exposed-nudity score is at or above this (0.2..1)
* `--nsfw-label <str>`: Per-class thresholds, comma-separated LABEL=score (e.g. FEMALE_BREAST_EXPOSED=0.5,BELLY_EXPOSED=0.8)
* `--nsfw-exposed-labels <str>`: NudeNet classes --nsfw-threshold is taken over (default: the 5 exposed)
* `--content-ratings <str>`: Keep only these NudeNet content ratings, comma-separated: safe, suggestive, explicit
* `--guard / --no-guard`: SingGuard policy check (default: on when a SingGuard URL is configured; --no-guard opts out)
* `--singguard-url <str>`: SingGuard server origin, e.g. http://guard:9150 (ARGUS_CURATOR_SINGGUARD_URL)
* `--guard-policy <str>`: SingGuard policy: default (the policy file, else SingGuard&#x27;s rules) or adult (allows adult nudity)  [default: default]
* `--guard-variants <str>`: Also judge every image under these policies, comma-separated (one more SingGuard pass each)
* `--guard-policies <str>`: Which policies&#x27; verdicts --guard-unsafe/--guard-rules read, comma-separated; any failing drops (default: the official one)
* `--guard-unsafe`: Drop images SingGuard judged unsafe, under any rule
* `--guard-rules <str>`: Drop images where one of these SingGuard rules fired, comma-separated (e.g. A,C)
* `--drop-nsfw`: Reject images NudeNet reads as exposed (&gt;= 0.2) before scenes, garments and SingGuard
* `--drop-no-face`: Reject images with no face before scenes, garments and SingGuard (needs --faces)
* `--drop-no-age`: Reject images with no MiVOLO, body-only or set age before scenes, garments and SingGuard (needs --mivolo)
* `--crops`: Crop each image that has a &lt;image&gt;.crop.json sidecar ({left, top, right, bottom}, displayed pixels) in memory before every check, and export it cropped (--copy-to only). Sources are never changed.
* `--detect-watermarks`: Find watermarks with the detector (--watermark-url) and cut them away at the frame&#x27;s edges; implies --crops. A watermark no edge cut removes rejects the image (crop:overlay).
* `--max-cut <float>`: Most of a side one edge cut may take (--detect-watermarks)  [default: 0.15]
* `--keep-overlays`: Keep images whose watermark cannot be cut away, uncropped, instead of rejecting
* `--xmp`: Write an XMP sidecar beside each exported file (face regions, names, content rating)
* `--xmp-sources`: Merge face regions, names and content rating into sidecars beside every scanned original (including rejected ones; independent of --xmp / --copy-to). Writes into the scanned folder
* `--carry-sidecars`: Transfer each exported image&#x27;s &lt;stem&gt;.txt / &lt;file&gt;.xmp / &lt;stem&gt;.xmp / &lt;stem&gt;.json with it, in the same mode
* `--run-stats <path>`: JSON file copied (pretty-printed) to &lt;dest&gt;/run_stats.json with the export
* `--face-names <str>`: Person names for the sidecars by cluster, comma-separated (face_1=Ada,face_2=Bo)
* `--csv <path>`: Write the per-image CSV report here
* `--json <path>`: Write the full ScanSummary JSON here
* `-v, --verbose`: Print per-image details
* `--help`: Show this message and exit.

## `argus-curator serve`

Start the argus-curator micro-server (FastAPI) on :8101.

Scan folders and export destinations from request bodies are resolved
relative to --source-root / --export-root; the endpoints refuse when their
root is not configured.

**Usage**:

```console
$ argus-curator serve [OPTIONS]
```

**Options**:

* `-p, --port <int>`: Port to listen on  [default: 8101]
* `--host <str>`: Host to bind to  [default: 0.0.0.0]
* `--cors`: Enable CORS for the localhost dev frontend (:3000)
* `--cors-origin <list[str]>`: Allowed CORS origin (repeatable; implies CORS; or CURATOR_CORS_ORIGINS)
* `--cors-any`: Allow ANY origin, credential-less (public demos only; implies CORS)
* `--source-root <str>`: Root that scan/thumb/upload paths resolve under (ARGUS_CURATOR_SCAN_ROOT)
* `--export-root <str>`: Root that export destinations resolve under (ARGUS_CURATOR_EXPORT_ROOT)
* `--allow-move`: Permit destructive mode=&quot;move&quot; exports (default: rejected)
* `--allow-delete`: Let POST /scan/{id}/delete remove any scanned image (CURATOR_ALLOW_DELETE / CURATOR_DEVMODE; default: only under-18 exposed images)
* `--help`: Show this message and exit.

## `argus-curator detectors`

Report which optional detectors/backends are available.

**Usage**:

```console
$ argus-curator detectors [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `argus-curator fairface-convert`

Convert the FairFace PyTorch weights to the ONNX model --fairface runs.

Download the weights from the FairFace repo (github.com/dchen236/FairFace,
CC BY 4.0). Needs: pip install argus-curator

**Usage**:

```console
$ argus-curator fairface-convert [OPTIONS] {weights}
```

**Arguments**:

* `weights <path>`: Official FairFace 7-race weights (fairface_alldata_20191111.pt)  [required]

**Options**:

* `-o, --output <path>`: Where to write the ONNX model (default: model cache)
* `--help`: Show this message and exit.

## `argus-curator mivolo-convert`

Fetch, verify and convert MiVOLO v2 to the ONNX model --mivolo runs.

Weights: huggingface.co/iitolstykh/mivolo_v2 (Apache-2.0), pinned by commit
and sha256. Needs: pip install argus-curator

**Usage**:

```console
$ argus-curator mivolo-convert [OPTIONS]
```

**Options**:

* `--weights <path>`: Local model.safetensors (default: download the pinned iitolstykh/mivolo_v2)
* `-o, --output <path>`: Where to write the ONNX model (default: model cache)
* `--help`: Show this message and exit.

## `argus-curator places365-convert`

Fetch, verify and convert a Places365-CNN to the ONNX model --scenes runs.

Weights: places2.csail.mit.edu (CC BY), pinned by sha256 since upstream
serves them over plain HTTP. Needs: pip install argus-curator

**Usage**:

```console
$ argus-curator places365-convert [OPTIONS]
```

**Options**:

* `--arch <str>`: resnet18 (default, 45 MB) or resnet50 (97 MB, ~1 pt better top-1)  [default: resnet18]
* `--weights <path>`: Local &lt;arch&gt;_places365.pth.tar (default: download it; sha256-checked)
* `-o, --output <path>`: Where to write the ONNX model (default: model cache)
* `--help`: Show this message and exit.

## `argus-curator garments-convert`

Fetch, verify and convert the Fashionpedia garment detector for --garments.

Weights: huggingface.co/valentinafevu/yolos-fashionpedia (MIT; trained on
Fashionpedia, CC BY 4.0), pinned by commit and sha256.
Needs: pip install argus-curator

**Usage**:

```console
$ argus-curator garments-convert [OPTIONS]
```

**Options**:

* `--checkpoint <path>`: Local dir with config.json + pytorch_model.bin (default: download the pinned one)
* `-o, --output <path>`: Where to write the ONNX model (default: model cache)
* `--help`: Show this message and exit.

## `argus-curator schema`

Emit the wire-contract JSON Schema consumers codegen against.

Run without flags to (re)write the committed schema; run with --check in CI
to fail if the models have drifted from the committed artifact.

**Usage**:

```console
$ argus-curator schema [OPTIONS]
```

**Options**:

* `-o, --output <path>`: Where to write the JSON Schema  [default: schema/curator-wire.schema.json]
* `--check`: Exit non-zero if the committed schema is stale (for CI)
* `--help`: Show this message and exit.
