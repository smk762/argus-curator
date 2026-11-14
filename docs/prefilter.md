# Prefilter: drop before the slow checks

SingGuard takes seconds an image, and scenes and garments add their own pass.
When some images are of no use whatever the later checks say, the prefilter
drops them first, so those passes never run on them. Each rule is opt-in, on
the request as `"prefilter": {...}` and on the CLI as a flag:

| Rule | Drops a passing image when | Needs |
|---|---|---|
| `drop_nsfw` (`--drop-nsfw`) | its exposed NudeNet reading is 0.2 or more (an image NudeNet and EraX could not score is kept) | the NudeNet check |
| `drop_no_face` (`--drop-no-face`) | no face was detected (an image whose detection failed, with `face_error` set, is kept) | face detection |
| `drop_no_age` (`--drop-no-age`) | no face has a MiVOLO age, no body-only age was read, and it has no [set age](face-attributes.md#an-age-for-the-frames-with-no-face) | MiVOLO |

The rules run once, right after identities (so a set age counts), on the
images that passed the hard filters. A dropped image stays in the results as
a reject with `reject_reason` `prefilter:nsfw`, `prefilter:no-face` or
`prefilter:no-age` (the first that applies, in that order); hard rejects keep
their own reason. A hard reject gets no face or age reading, so the under-18
rule could not speak for it, and it is not sent to NudeNet, EraX or SingGuard
either; `min_short_side` cannot go below 512 for the same reason. A dropped
image has been through faces, NudeNet and EraX, since the rules read them, but
not SingGuard. No rejected image, dropped or hard, is kept or exported, whatever
its score, and not when named in an export's `selection`; `include_rejected` is
ignored. Its keep reason (`/scan/{id}/select`, the CSV report) is its
`reject_reason`, e.g. `prefilter:nsfw`. A dropped image no longer counts
toward its identity: `face_clusters[].size` and the representative leave it
out, an identity with only dropped images is removed from `face_clusters`, and
its rows and `set_age.cluster_id` keep naming it. The pooled attributes and the
set age were read before the drop and still include it. A streaming scan sends
the dropped images in a `prefilter` chunk, with the recounted `face_clusters`.
A rule without the check it needs
(`drop_nsfw` with NudeNet off or no `ARGUS_CURATOR_NUDENET_URL`) is refused
with HTTP 400 (CLI exit 2). The summary records the rules as
`prefilter_config`, and `/health` reports `prefilter`.

```bash
# Faces with an age only, and no SingGuard pass on nudity NudeNet already saw
argus-curator scan /data/inbox --mivolo --drop-nsfw --drop-no-age --csv report.csv
```
