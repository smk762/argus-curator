# How scoring works

Every scanned image goes through these steps. The first four run at scan time;
selection runs at export (or on a dry run with `POST /scan/{scan_id}/select`).

1. **Hard filters** — min short side, max aspect ratio, blur (Laplacian-edge variance floor, measured with the long side at 1024px so the threshold does not depend on the file's resolution).
2. **Composite score** — target-aware weighted blend of sharpness / resolution / artifact, plus a small composition bonus that depends on `target_category` (e.g. identity rewards a single centred face; setting rewards wide framing; concept rewards clean, high-resolution images and ignores faces).
3. **Face-count fit** — with `[faces]`, identity targets penalise 0 or 2+ faces; other categories are progressively more tolerant.
4. **Near-duplicate dedup** — pHash clustering keeps the highest-scoring representative and *reports* the rest (never silently dropped).
5. **Selection (at export)** — score threshold + optional diversity cap (`max_keep` / `diversity_weight`) + optional face-cluster filter. Every excluded image carries a `keep_reason`.

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
