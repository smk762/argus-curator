# Garments (Fashionpedia)

`--garments` (or `"garments": {"enabled": true}`) runs a YOLOS-small detector
fine-tuned on [Fashionpedia](https://fashionpedia.github.io/home/)
([`valentinafevu/yolos-fashionpedia`](https://huggingface.co/valentinafevu/yolos-fashionpedia),
MIT; dataset CC BY 4.0) on every passing image. Each detection records its
garment type (27 classes: `dress`, `pants`, `shoe`, `bag_wallet`, `hat`, …),
probability, box, whether the box touches the frame edge (likely cut off), and
a dominant colour from the box's centre. Results land on `results[].garments`
(`[]` = ran, found nothing) and in the CSV as `label:conf:#colour`, with `*`
marking a cut-off garment.

## Accuracy

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

## Filters

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

## Installing the model

The detector is converted to ONNX once:

```bash
pip install "argus-curator[garments-convert]"   # torch + transformers, only for this step
argus-curator garments-convert                  # pinned commit + sha256
# -> ~/.cache/argus_curator/models/yolos_fashionpedia.onnx
```

Point elsewhere with `--garments-model` / `ARGUS_CURATOR_GARMENTS_MODEL`.
