# Nudity detection (NudeNet)

Curator can check every scanned image with [NudeNet](https://github.com/notAI-tech/NudeNet),
a detector for 18 body-part classes (`FEMALE_BREAST_EXPOSED`, `BUTTOCKS_COVERED`,
`FACE_FEMALE`, …). It talks to a **NudeNet server over HTTP**; curator never
imports NudeNet, so it gains no dependency and stays MIT. NudeNet itself is
AGPL-3.0, so running its server is a choice for whoever deploys it.

## Configuration

Configure the server in the environment (or `.env`), never per request. `<LAN_HOST>` is a placeholder for the host that runs the server:

```bash
ARGUS_CURATOR_NUDENET_URL=http://<LAN_HOST>:8130   # its origin; /infer is appended
ARGUS_CURATOR_NUDENET_TIMEOUT=30                     # seconds per image (default 30)
ARGUS_CURATOR_NUDENET_WORKERS=4                      # concurrent requests (default 4)
ARGUS_CURATOR_ERAX_URL=http://<LAN_HOST>:8131      # optional; empty turns EraX off
ARGUS_CURATOR_ERAX_TIMEOUT=30
ARGUS_CURATOR_ERAX_WORKERS=4
```

Use the suite's shared server, [argus-retina](https://github.com/smk762/argus-retina)'s
`ghcr.io/smk762/argus-retina-nudenet` (NudeNet's **640m** model; EraX on :8131).
`docker compose up` here starts that same image as the bundled `nudenet`
service, and uses it unless `.env` points elsewhere.

Any host speaking NudeNet's `/infer` API works, but the model matters: **320n
(the upstream `ghcr.io/notai-tech/nudenet` image) scores the same image
differently from 640m, so a threshold tuned on one doesn't carry to the other.**
Ask a server which model it runs with argus-cortex's
`NudeNetClient(url).info()` (`model`, `resolution`), and record it alongside any
scores you tune against.

### Choosing a threshold

**Use 0.2**, any exposure NudeNet reports (`DEFAULT_NSFW_THRESHOLD`, the same
default as argus-darkroom). It's conservative on purpose and can drop the odd
image that isn't exposed; raise it per class with `--nsfw-label` if that bites.
The CLI's scan summary counts exposed images at 0.2 unless `--nsfw-threshold`
says otherwise. Curator never filters by default: a request that sets no
threshold drops nothing for nudity.

How 0.2 was chosen (2026-11-16, #156). 300 random images of an adult test set
were scored on both models, and the 38 images where they disagreed at 0.5 were
reviewed by hand:

| | 320n | 640m |
|---|---|---|
| Median exposed score | 0.43 | 0.54 |
| Dropped at 0.5 | 133 | 153 |
| Right on the 38 disagreements | 9 | 30 |

- 640m was right on all 29 images only it dropped: 320n had missed them.
- Of the 9 only 320n dropped, 8 were exposed. 640m scored them 0.27–0.49 (or 0,
  where EraX fills in), so a 0.5 cutoff let them through. 0.35 would catch 7,
  and 0.2 catches all 8.
- On 300 random public-domain images from argus-quarry's pool, 640m scored none
  at 0.2 or above, so the lower cutoff cost nothing on clean images there.

Run EraX beside 640m: it fills in where NudeNet reports no exposure at all.

## Opt-out by default

**It is opt-out.** When a URL is configured, every scan checks its images,
including ones that failed the quality filters, unless the scan opts out
(`--no-nsfw`, or `"nsfw": {"enabled": false}`). `--nsfw` / `"enabled": true`
makes the check mandatory: the scan is refused if no server is configured. Set
the URL empty to switch the check off for a deployment. The scan summary
records whether it ran (`nsfw_config.enabled`). Images are sent to that server,
so keep it on infrastructure you control.

## Scores and filters

Each image records every detection, the highest score per class
(`nsfw.labels`), and an **exposed score**: the highest score among the five
exposed-nudity classes (`BUTTOCKS_EXPOSED`, `FEMALE_BREAST_EXPOSED`,
`FEMALE_GENITALIA_EXPOSED`, `MALE_GENITALIA_EXPOSED`, `ANUS_EXPOSED`). Faces and
covered body parts are detected too but never count towards it. The CSV
report has `nsfw_exposed_score` and `nsfw_labels`.

Filters drop an image when a score is **at or above** its threshold:

```bash
# Total: drop any image with a reported exposure (the recommended 0.2)
argus-curator scan /data/inbox --nsfw-threshold 0.2 --copy-to /data/clean

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

## EraX: a second detector

An optional [EraX-NSFW](https://huggingface.co/erax-ai/EraX-NSFW-V1.0) server
(`ARGUS_CURATOR_ERAX_URL`) is a second detector for frames NudeNet does not
score. It speaks the same `/infer` API with its own classes (`anus`,
`make_love`, `nipple`, `penis`, `vagina`). When NudeNet's exposed score is 0
or the check failed, those hits are mapped onto NudeNet's exposed labels
(`nipple` → `FEMALE_BREAST_EXPOSED`, `make_love` → both genitalia classes)
and recorded on `nsfw`, with the raw EraX result on `erax` and
`erax_filled: true`. An image NudeNet already scored as exposed is left
alone. A `nipple` inside a box NudeNet called `MALE_BREAST_EXPOSED` is not
mapped (a bare male chest is fine); one NudeNet did not box still counts.
When NudeNet's check failed, the filled image also gets `nudenet_failed:
true` and the NudeNet filters exclude it as `nsfw-unchecked` unless the EraX
hits reach a threshold (EraX has no buttocks class). EraX runs beside
NudeNet on every image that passed the hard filters, and each nsfw row is
streamed as soon as it is decided: at once when NudeNet scored it exposed, otherwise when EraX has
answered too. `--no-nsfw` turns both off. `GET /detectors` reports `erax`.

## Content rating

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

The content rating is also written into XMP sidecars as `argus:ContentRating`
(see [XMP sidecars](xmp.md)). For a whole-policy verdict rather than per-part
scores, see [Policy check (SingGuard)](policy-check.md).
