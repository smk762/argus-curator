# How scoring works

Every scanned image goes through these steps. The first four run at scan time;
selection runs at export (or on a dry run with `POST /scan/{scan_id}/select`).

1. **Hard filters** — min short side, max aspect ratio, blur (Laplacian-edge variance floor, measured with the long side at 1024px so the threshold does not depend on the file's resolution).
2. **Composite score** — target-aware weighted blend of sharpness / resolution / artifact, plus a small composition bonus that depends on `target_category` (e.g. identity rewards a single centred face; setting rewards wide framing; concept rewards clean, high-resolution images and ignores faces).
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
