# Face attributes (MiVOLO, FairFace)

Curator can read apparent age and gender from each detected face with
[MiVOLO v2](#mivolo), and race, gender and an age bracket with
[FairFace](#fairface). Both run on the faces InsightFace already found, so
either one implies `--faces`.

These are a model's guess from appearance, not facts about anyone; expect
errors on ambiguous faces, heavy makeup or poor lighting, and check
`gender_conf` before relying on a gender.

## MiVOLO

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

### Reading the body as well

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

### Installing the model

The model is not on PyPI; it is converted to ONNX once, after which
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

## An age for the frames with no face

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
of the set shows (see [Policy check](policy-check.md)). It is an inference
about the set, not a reading of the frame: a viewer shows it as such, and
where a wrong age is the costlier mistake treats it as it treats a measured
one. `/health` reports `set_age`. Deleting images re-reads the set age from
the faces left, which can change it, remove it or give one to a frame that had
none; a check that already ran is not run again.

## FairFace

[FairFace](https://github.com/dchen236/FairFace) (`--fairface`, or
`"faces": {"fairface": true}`; implies `--faces`) runs the FairFace ResNet-34
(Kärkkäinen and Joo, WACV 2021; weights CC BY 4.0) on each detected face. It
predicts three things, each a softmax over its classes, so every label carries
its probability as a confidence:

| | FairFace |
|---|---|
| `race` | 7 classes: `white`, `black`, `latino_hispanic`, `east_asian`, `southeast_asian`, `indian`, `middle_eastern` |
| `gender` | `male`, `female` |
| `age` | bracket: `0-2`, `3-9`, `10-19`, `20-29`, `30-39`, `40-49`, `50-59`, `60-69`, `70+` |

`race` comes only from FairFace. When MiVOLO runs as well, it supersedes
FairFace's gender and age: `gender_source` and `age_source` name the model
each value came from. Results land in the same places as MiVOLO's (per face,
on the primary face, pooled per identity).

FairFace was trained on dlib face chips. Curator has InsightFace's 5 keypoints
rather than dlib's landmarks, so it maps those onto dlib's mean-face template
under the same padding: an approximation of the training crop, not a
bit-exact one.

**The model** is not downloaded for you. Fetch the official 7-race weights
(`fairface_alldata_20191111.pt`) from the FairFace repository, then convert
them to ONNX once:

```bash
pip install "argus-curator[fairface-convert]"     # torch + torchvision, only for this step
argus-curator fairface-convert fairface_alldata_20191111.pt
# -> ~/.cache/argus_curator/models/fairface_res34_race7.onnx
```

Point elsewhere with `--fairface-model` or `ARGUS_CURATOR_FAIRFACE_MODEL`
(`fairface-convert --output` writes somewhere else). As with MiVOLO, a server
without the model refuses a FairFace scan with 400, and `GET /detectors`
reports `fairface`.

## Filters

All filters read the primary face:

| CLI | Request field | Keeps |
|---|---|---|
| `--race` | `face_races` | these FairFace races |
| `--gender` | `face_genders` | these genders |
| `--age` | `face_ages` | these age brackets |
| `--min-age` / `--max-age` | `min_age_years` / `max_age_years` | a MiVOLO age in this range (years) |
| `--min-attr-conf` | `min_attribute_conf` | filtered labels at least this confident |

`--min-attr-conf` requires a filtered label's confidence to reach the floor (a
MiVOLO age has none, so it is not held to it). Excluded images say why in
`keep_reason`: `face-<attr>-excluded`, `face-<attr>-low-confidence`,
`face-<attr>-missing`, `face-age-out-of-range`, `face-age-years-missing`, or
`face-attributes-missing`.

```bash
# Predict age in years and gender per face (MiVOLO), then keep adults up to 35
# whose gender prediction is confident
argus-curator scan /data/images --mivolo \
    --gender female --min-attr-conf 0.6 --min-age 25 --max-age 35 --copy-to /data/curated
```
