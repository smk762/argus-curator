# Scene tags (Places365)

`--scenes` (or `"scenes": {"enabled": true}` on the scan request) runs a
[Places365](https://github.com/CSAILVision/places365)-CNN (Zhou et al., TPAMI
2017) on every passing image and records its top-k scene categories (365 of
them: `beach`, `bedroom`, `gas_station`, `forest_path`, …) with their softmax
probabilities, plus an indoor/outdoor call. `environment_conf` is the
probability mass on categories of that environment, not upstream's top-10
vote. Results land on `results[].scene` and in the CSV report.

## Filters

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

## Installing the model

The weights are converted once to ONNX, like the face models:

```bash
pip install "argus-curator[places365-convert]"   # torch, only for this step
argus-curator places365-convert                  # resnet18 (default) or --arch resnet50
# -> ~/.cache/argus_curator/models/places365_resnet18.onnx
```

Upstream serves the checkpoints over plain HTTP, so each is pinned by sha256.
ResNet-50 is twice the size for about one point of top-1 (55.6% vs 54.5%).
Point elsewhere with `--places365-model` / `ARGUS_CURATOR_PLACES365_MODEL`.
