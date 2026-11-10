# Synthetic source portraits for argus-mask

This guide covers the suite's tools for making a **synthetic source portrait** for argus-mask: an image of a person who does not exist, made with an image model, that a photoreal face route builds a talking head from.

It follows argus-mask's [Making a synthetic face for input](https://github.com/smk762/argus-mask/blob/main/docs/synthetic-faces.md), which stays the authority on two things: the rule, and what a good portrait looks like. Read that first. This guide says which suite tool handles each step, and what each tool doesn't check.

```
argus-darkroom ──▶ argus-curator ──▶ (argus-lens) ──▶ you ──▶ argus-mask
generate, guarded   shortlist          reading aid      pick     inspect, serve
records every run   age, pose, faces   captions         look-alike check
```

**Real people:** a real person's face needs their documented consent ([argus-mask#19](https://github.com/smk762/argus-mask/issues/19)). Curator support for consented real-person sets is tracked in [argus-curator#133](https://github.com/smk762/argus-curator/issues/133). This guide covers synthetic faces only.

**Not used:**
- **argus-quarry** collects public-domain and CC0 portraits of real people. A licence to use a photo isn't the person's consent to be made into a talking face, so quarry images must never be a mask source, even as a reference for an identity adapter.
- **argus-forge and argus-proof** are for LoRA training, which this workflow doesn't need.

## What the tools check, and what they don't

| The rule (mask guide) | Checked by a tool? |
|---|---|
| No real person named in the prompt | **No.** Darkroom's pre-flight judges the prompt for harm, not likeness. It's up to you. |
| No real photo as an input (img2img, IP-Adapter, InstantID, face-swap) | **No.** Darkroom renders text-to-image templates, but a custom workflow could take an image. Use the shipped SDXL templates. |
| No LoRA of a real person | **No.** Darkroom pins each LoRA by name and sha256 in the record, so you can audit it, but it can't tell what the LoRA was trained on. |
| Look-alike check | **No.** Run a reverse image search and look at the result yourself, then record it. |
| 18 and over | **Partly.** Darkroom's SingGuard minors rule and curator's MiVOLO estimate both apply. An estimate is a tool's guess, so discard anything that reads as under 18, whatever the number says. |
| Generator licence | **Recorded, not judged.** The darkroom record names the checkpoint and its sha256, and you add the licence. |

## 1. Generate with argus-darkroom

[argus-darkroom](https://github.com/smk762/argus-darkroom) checks the prompt before rendering, renders through ComfyUI, scans each output, and keeps a record of every attempt. Use an SFW policy, either `default` or one of the `sfw-*` policies. Under these, darkroom refuses a prompt that any pre-flight check flags, and omits any exposed output.

These are the mask guide's SDXL settings, written as darkroom flags:

```bash
for seed in $(seq 1001 1024); do
  argus-darkroom generate "$(cat prompt.txt)" -n "$(cat negative.txt)" \
      --checkpoint sdxl/sd_xl_base_1.0.safetensors --policy default \
      --sampler dpmpp_2m --scheduler karras --steps 35 --cfg 6 \
      --width 896 --height 1152 --seed "$seed"
done
```

- **Prompt and negative prompt:** use the mask guide's templates. Keep the adult age in the prompt, and `child, minor` in the negative.
- **One at a time:** darkroom runs one generation at a time and answers 409 to a second, which is why the loop runs sequentially.
- **Checkpoint path:** the checkpoint goes under ComfyUI's `checkpoints/sdxl/` so darkroom picks the `sdxl` workflow template.

Each attempt leaves `<id>.json` in darkroom's `OUTPUT_ROOT`, and released images also leave `<id>.png`. Shortlist only `passed` records:

```bash
mkdir -p candidates
for rec in darkroom-output/*.json; do
  [ "$(jq -r .outcome "$rec")" = passed ] || continue
  id=$(jq -r .id "$rec")
  cp "darkroom-output/$id.png" "darkroom-output/$id.json" candidates/
done
```

`review` images had a SingGuard rule fire. Leave them out, even if they look fine.

### What the darkroom record covers

The mask guide asks for a `generation.json` beside the portrait. Most of it is already in the darkroom record:

| `generation.json` needs | Darkroom record |
|---|---|
| Model, with file name and sha256 | `request.checkpoint`, `request.vae` |
| LoRAs, with sha256 | `request.loras` (name, sha256, weight, trigger words) |
| Prompt and negative prompt | `request.final_prompt`, `request.negative_prompt` (and `user_prompt`, plus `expanded_prompt` if `--expand` was used) |
| Seed, sampler, steps, CFG and size | `request.seed`, `request.sampling` |
| Tool versions | `darkroom_version`, `request.engine_version` |
| Workflow as run | `request.workflow` and `request.workflow_sha256`. Keep a copy of the template file, as the record holds only its hash. |
| Licences | **not recorded:** add them |
| Edits after generation | **not recorded:** add them |
| Look-alike check and its date | **not recorded:** add it |

## 2. Shortlist with argus-curator

Curator doesn't decide what's a good portrait, but it narrows a batch of 16 to 32 down to the ones worth looking at:

```bash
argus-curator scan candidates/ \
    --mivolo --drop-no-face --min-age 18 --pose frontal \
    --garments --exclude-garment glasses,hat,headband_hair_accessory \
    --keep-similar --carry-sidecars \
    --csv shortlist.csv --copy-to shortlist/
```

| Flag | Why |
|---|---|
| `--mivolo` | Detects faces (InsightFace) and estimates age in years (MiVOLO). It also gives head pose: yaw, pitch and a pose bucket. |
| `--drop-no-face` | A portrait without a detected face is unusable. |
| `--min-age 18` | Exports only primary faces MiVOLO reads as 18 or over. |
| `--pose frontal` | Exports only frontal faces, meaning `abs(yaw) <= 15°` (`frontal_max_yaw`). Pitch plays no part in the bucket: it's on each face in the `--json` summary, not in the CSV. Roll isn't measured, so check "head level" by eye. |
| `--garments --exclude-garment …` | Drops portraits where Fashionpedia sees glasses, a hat, or a headband or hair accessory. It doesn't see hands or hair over the face. |
| `--keep-similar` | A seed sweep of one prompt gives near-duplicates. By default curator keeps only the best of each group, but here you want to compare them all. |
| `--carry-sidecars` | Copies each image's `<id>.json` darkroom record into `shortlist/` with it. |
| `--csv` | One row per image, with `primary_face_yaw`, `primary_face_age_years`, `sharpness`, `score` and `reject_reason`, for comparing them side by side. |

`--mivolo` and `--garments` need their converted models: run `argus-curator mivolo-convert` and `argus-curator garments-convert` once (see the [README](../README.md#installation)). Without them the scan refuses up front and names the missing model.

With NudeNet configured (`ARGUS_CURATOR_NUDENET_URL`), the scan also scores exposure. A head-and-shoulders portrait should score near zero, so add `--drop-nsfw` to reject anything that doesn't.

**Sharpness is measured over the whole image.** The guide's plain mid-grey background has little texture, so a portrait with a sharp face can still score under the default `--blur-threshold 100` and be rejected as blurry. If good portraits are missing from the shortlist, look for `blurry` in the CSV's `reject_reason` column and lower the threshold.

Don't use `--detect-watermarks` here. It crops a watermark away, and the guide wants the chosen image unedited. Discard a portrait with a watermark or text instead, and regenerate.

## 3. Read the captions (optional): argus-lens

Several rows of the guide's table have no detector:
- neutral expression with lips closed;
- eyes open and looking into the lens;
- even light;
- a plain background;
- no hands over the face.

[argus-lens](https://github.com/smk762/argus-lens) captions can point you at problems, though they don't settle them:

```bash
argus-lens caption shortlist/ --format txt --backend hybrid
```

Lens files caption words under categories: `smile`, `expression`, `glasses` and `makeup` under identity, and `rim light`, `harsh light`, `shadows`, `dramatic lighting` and `backlighting` under lighting. A caption with one of these words is a portrait to look at again. A caption without them is no guarantee the portrait is clean.

## 4. Pick, and check for look-alikes

Choose against the mask guide's table and checklist by eye. Then run the look-alike check on the one you keep, and record what you searched and when. If the face looks like someone you could name, discard it and change the seed.

## 5. More than one view

Some splat routes train on several views. The mask guide gives two synthetic-only ways to make them. Either way, every view has to stay the same person.

Curator can check that coarsely. Scan the views with faces on, and a view in a different face cluster from your frontal portrait has drifted:

```bash
argus-curator scan views/ --faces --cluster-eps 0.35 --csv views.csv
```

Each face gets a `face_<n>` cluster, built by cosine distance between InsightFace embeddings. All the views should land in one cluster. The default `--cluster-eps 0.5` is meant for grouping one person across a varied dataset, so set it tighter for a drift check. The 0.35 above is a starting point, not a calibrated value. Curator reports cluster membership, not each view's distance from the frontal portrait, so a view that is drifting but still inside the threshold won't show up.

Keep lighting and background identical across views. No tool here compares them.

## 6. Hand off to argus-mask

Lay out the source as the mask guide says:

```
assets/<face-id>/source/
  portrait.png          # the chosen image, unedited (darkroom's <id>.png)
  portrait.final.png    # after any inpaint and upscale
  generation.json       # the darkroom record, plus licences, edits and the look-alike check
  workflow.json         # the darkroom workflow template the record names
```

Start `generation.json` from the darkroom `<id>.json` that came with the image, and add what it lacks: licences, edits and the look-alike check. Then build and inspect the face with argus-mask's `inspect --kind synthetic`, as its guide shows.

## Gaps

These parts of the guide have no tool support yet:

| Wanted | Today |
|---|---|
| Head level (roll) | Not measured. InsightFace returns roll, but curator drops it. |
| Face large enough in pixels | Face bbox recorded, no filter on its size. |
| Sharp face | Sharpness is measured over the whole image. |
| Neutral expression, lips closed, eyes open | Lens captions only, no detector. |
| Even light, plain background | Lens captions only. |
| Identity distance per view | Cluster membership only. |
| Licence and look-alike check in the record | Added by hand. |

These issues track the gaps:

- **[argus-curator#130](https://github.com/smk762/argus-curator/issues/130):** roll, face size, face sharpness, eyes open and mouth closed (from landmarks), and identity distance per view.
- **[argus-darkroom#33](https://github.com/smk762/argus-darkroom/issues/33):** licences in the record, a check for named real people in the prompt, and adding the look-alike check to a record afterwards.

Lighting and background stay checks by eye.
