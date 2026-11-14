# Nudity detection (NudeNet)

Curator can check every scanned image with [NudeNet](https://github.com/notAI-tech/NudeNet),
a detector for 18 body-part classes (`FEMALE_BREAST_EXPOSED`, `BUTTOCKS_COVERED`,
`FACE_FEMALE`, …). It talks to a **NudeNet server over HTTP**; curator never
imports NudeNet, so it gains no dependency and stays MIT. NudeNet itself is
AGPL-3.0, so running its server is a choice for whoever deploys it.

## Configuration

Configure the server in the environment (or `.env`), never per request. `<LAN_HOST>` is a placeholder for the host that runs the server:

```bash
ARGUS_CURATOR_NUDENET_URL=http://<LAN_HOST>:8080   # its origin; /infer is appended
ARGUS_CURATOR_NUDENET_TIMEOUT=30                     # seconds per image (default 30)
ARGUS_CURATOR_NUDENET_WORKERS=4                      # concurrent requests (default 4)
ARGUS_CURATOR_ERAX_URL=http://<LAN_HOST>:8081      # optional; empty turns EraX off
ARGUS_CURATOR_ERAX_TIMEOUT=30
ARGUS_CURATOR_ERAX_WORKERS=4
```

Any host speaking NudeNet's `/infer` API works, including the official
`ghcr.io/notai-tech/nudenet` image (320n) or a 640m sidecar. `docker compose up`
in argus-cull starts a 640m server and uses it unless `.env` says otherwise.

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
# Total: drop anything whose exposed score is 0.5 or more
argus-curator scan /data/inbox --nsfw-threshold 0.5 --copy-to /data/clean

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
