# How scoring works

Every scanned image goes through these steps. The first four run at scan time;
selection runs at export (or on a dry run with `POST /scan/{scan_id}/select`).

1. **Hard filters** — min short side, max aspect ratio, blur (Laplacian-edge variance floor, measured with the long side at 1024px so the threshold does not depend on the file's resolution).
2. **Composite score** — target-aware weighted blend of sharpness / resolution / artifact, plus a small composition bonus that depends on `target_category` (e.g. identity rewards a single centred face; setting rewards wide framing; concept rewards clean, high-resolution images and ignores faces). With embeddings, an opt-in [zero-shot profile fit and junk score](#profile-fit-and-junk-zero-shot) adds to or takes from that bonus.
3. **Face-count fit** — with `[faces]`, identity targets penalise 0 or 2+ faces; other categories are progressively more tolerant.
4. **Near-duplicate dedup** — pHash clustering (plus, opt-in, [CLIP similarity](#semantic-near-duplicates)) keeps the highest-scoring representative and *reports* the rest (never silently dropped).
5. **Selection (select and export)** — score threshold + filters + optional diversity cap (`max_keep` / `diversity_weight`), then a ranking (`rank`) and an optional train/held-out `split`; see [HTTP API](http-api.md#ranking-max_keep-and-splits). Every excluded image carries a `keep_reason`.

A hard reject (step 1) is never kept or exported, whatever its score, and gets
no face, age, scene or garment reading. The [prefilter](prefilter.md) and
[crops](crops.md) add their own rejects on top of these.

## Score weights

The composite score's weights are `ScanConfig` fields, on the scan request's
`config`:

| Field | Default |
|---|---|
| `weight_sharpness` | 0.35 |
| `weight_resolution` | 0.30 |
| `weight_artifact` | 0.15 |
| `weight_subject` | 0.20 |
| `sharpness_ref` | 800.0 (the sharpness that scores full marks) |
| `resolution_ref` | 1024 (the short side that scores full marks) |

The hard-filter floors are `min_short_side` (512, which is also the lowest it
can be set), `max_aspect_ratio` (3.0) and `blur_threshold` (100.0); near-duplicate
grouping uses `cluster_distance` (pHash Hamming distance, 10; -1 turns grouping
off).

## Semantic near-duplicates

pHash misses a mirrored frame, a small crop or a second scan of the same print:
they land 15 to 35 bits apart. With `embeddings.enabled` on, setting
`config.semantic_duplicates` (CLI `--semantic-dupes`) also links two images
whose CLIP embeddings have a cosine similarity of at least
`config.semantic_duplicate_cosine` (default 0.95, CLI `--semantic-dupes-cosine`).
A scan asking for it without embeddings is refused (422), never run pHash-only.

The two tests share one grouping. Passing images are taken best score first;
each joins the first representative it is a near-duplicate of by either test,
or becomes a representative itself. So a group has one representative, the
best-scoring image, and the same `similar-to:<rel_path>` keep reason. It is not
a transitive closure: `duplicate_of` always names an image the duplicate itself
matched, never one reached through a chain (CLIP similarity chains across a
whole shoot), and a scan with `semantic_duplicates` off groups exactly as
before. `ImageResult.duplicate_kind` says which test tied the image to
`duplicate_of`: `phash`, `semantic` or `both` (also a column in the CSV report).

Why 0.95: on the DataComp ViT-L/14 embeddings, every pair of distinct shots
measured scored at most 0.944 (Churchill portraits from different sittings;
different paintings of cathedral interiors: 0.924; one photographer's 205
Flickr photos: 0.85), while rescans of the same photo or print scored 0.98 to
1.0 and mirrored frames a median 0.989. Crops sit near the line (a 90% crop
median 0.966, 80% median 0.950), and so do the hardest negatives: one prompt
rendered by different SDXL/Flux checkpoints (99th percentile 0.943, max 0.981).
Lower it for crops; raise it for generated sets.

Only the vectors are memory-only. The group fields (`duplicate_of`,
`duplicate_kind`, `similar_group`) are plain values, saved with the scan. Rows
dropped later (a delete or a move) are regrouped with the embeddings the server
still holds; after a restart they are gone, so the semantic links the rows
recorded stand in for them. A semantic duplicate whose representative was
dropped then cannot be compared afresh, and `ScanSummary.semantic_groups_stale`
turns true: rescan with embeddings on to regroup it.

## Profile fit and junk (zero-shot)

Nothing in the sharpness, resolution and artifact terms says whether an image
is the kind of image the `TargetProfile` asks for, or is junk that passes
them: a screenshot, a collage, a page of text, a meme. With `embeddings.enabled`
on, two opt-in scores compare each passing image's CLIP embedding with text
prompts embedded by the same checkpoint's text tower:

| Field (`config`) | Default | CLI | What it does |
|---|---|---|---|
| `profile_fit` | off | `--profile-fit` | scores the fit; adds `profile_fit_weight` x fit to the target bonus |
| `profile_fit_weight` | 0.10 | | |
| `junk_check` | off | `--junk-check` | scores the junk probability; takes `junk_weight` x junk off the target bonus |
| `junk_weight` | 0.15 | | |
| `junk_threshold` | 0.8 | `--junk-threshold` | the junk probability that names a `junk_kind` |

The scores land on `ImageResult.zero_shot`: `profile_fit` and `junk`
(probabilities, None when not asked for) and `junk_kind` (`screenshot`,
`collage`, `text` or `meme`, set only once `junk` reaches `junk_threshold`).
They are also CSV report columns, and `score_breakdown` lists their parts of
the target bonus as `zero_shot_profile_fit` and `zero_shot_junk` (the bonus
itself, `target_bonus`, includes them, so it can go negative). An image with no
embedding has no `zero_shot` and neither term. Neither score rejects an image:
the opt-in [`prefilter.drop_junk`](prefilter.md) (`--drop-junk`) drops the ones
with a `junk_kind`, as `prefilter:junk`. A scan asking for any of the three
without embeddings is refused (422, CLI exit 2), never run without them.

- **Profile fit** is the probability of the profile's category in one softmax
  over the four framing prompts (identity, wardrobe, pose_composition,
  setting). `concept`, a style LoRA, has no framing: its fit is the probability
  of its style (photo or anime). The style is left out of the other
  categories' fit, because it was too unsure to weigh (below).
- **Junk** is the summed probability of the four junk labels in one softmax
  over them and nine ordinary-image labels (photo, portrait, people, place, old
  photograph, print, art, anime, render). Each label's vector averages a few
  prompts (`zeroshot.py` lists them).

The prompt vectors ship precomputed as `data/clip_prompts.json`, pinned by
sha256 (`zeroshot.PROMPTS_SHA256`), so no text model or tokenizer runs at scan
time: the taxonomy has two styles and five categories, so every profile's
prompts are in the file. `argus-curator embeddings-convert --prompts`
regenerates it (byte-identical across runs) from the pinned checkpoint; a test
fails if the prompts in the code and the file differ. Like the semantic
duplicates, only the plain scores are saved with the scan; the vectors stay in
memory.

### Calibration

Measured on local images with the DataComp ViT-L/14 image tower and the
committed prompt file (logit scale 100). Ordinary images: 196 Wikimedia
Commons portraits (Einstein, Curie, Twain, Lovelace, Churchill; archival
photos, prints and scans, some files counted twice), 30 Commons wardrobe
images (kimono, military uniform), 14 Commons interiors and gardens, 11 Commons
Art Nouveau and aurora images, 371 SDXL/Flux renders from argus-mint and 400
frames from two photographers' model shoots: 1,022 in all. Junk: 37 real
browser screenshots of public sites (desktop and phone viewports), 18 real
screenshots of text pages (RFCs, Project Gutenberg, the GPL, PEP 8), 41
collages (40 grids of 4 to 20 of the photos above, and one real contact sheet)
and 40 memes (one of the photos above with bold outlined captions). The
collages and memes are built from real photos; there is no anime set, so the
anime side is not calibrated.

| `junk_threshold` | ordinary images flagged | precision | screenshot | text | collage | meme |
|---|---|---|---|---|---|---|
| 0.5 | 11 / 1,022 (1.1%) | 0.92 | 0.97 | 1.00 | 0.95 | 0.85 |
| 0.7 | 10 (1.0%) | 0.93 | 0.95 | 1.00 | 0.93 | 0.85 |
| **0.8** | **8 (0.8%)** | **0.94** | 0.95 | 1.00 | 0.93 | 0.82 |
| 0.9 | 8 (0.8%) | 0.94 | 0.92 | 1.00 | 0.85 | 0.80 |

Every ordinary image flagged at 0.8 is a Commons portrait: archival scans
with printed captions and plates, read as `text`, and one 2025 upload, read as
`meme` (two of the eight are files counted twice); none of the renders or shoot
frames is, and the render and shoot sets' highest junk probability is 0.13 and
0.58. The memes it misses carry one short caption. The kind is right for every
collage, 37 of 40 memes (3 read as `text`), 12 of 18 text pages (6 as
`screenshot`) and 22 of 37 screenshots (8 as `text`, 7 as `meme`), so read
`junk_kind` as a hint and `junk` as the signal. The ordinary-image labels
matter: with only "a photo" and "a painting" to weigh junk against, 7.8% of
ordinary images (10% of the renders) scored 0.5 or more.

Profile fit, median probability of the right category: Commons portraits 0.91
under `identity` (64% have it as the likeliest of the four; most of the rest
read as `wardrobe`), the wardrobe set 0.77 (67%), interiors and gardens 1.00
(all 14). Full-body shoot frames split between `wardrobe` (37%) and
`pose_composition` (61%), which frame alike. Style: real photos read as photo
(anime above 0.5 for 1% of the portraits, 7% of the shoot frames), but so did
too little else to fold into the fit: 53% of the wardrobe set (Edo-period
kimono prints among them) and 20% of the photoreal renders read as anime.

Turning `drop_junk` on is safe for photo sets at 0.8; for an archive of old
prints, review the `junk_kind` rows first, or raise the threshold.
