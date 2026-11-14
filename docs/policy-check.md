# Policy check (SingGuard)

Where NudeNet scores nudity per body part, a [SingGuard](https://huggingface.co/inclusionAI/SingGuard-4b-GGUF)
server judges each image against a whole moderation policy and answers with a
**verdict and the rule that fired**. Its default rules:

| Rule | Covers |
| --- | --- |
| A. Sexual Content Risk | sexual content |
| B. Real-World Crimes & Public Safety | violent crime, weapons, public-safety threats |
| C. Unethical Behavior | hate, harassment, self-harm, gore and shock content |
| D. Cybersecurity | data leaks, hacking, surveillance abuse |
| E. Agent Safety | attempts to expose model safeguards |
| F. Politically Sensitive | political advocacy, unrest, attacks on political figures |
| G. Animal Abuse | cruelty to animals |

## Modes

It gives **no scores**, so it complements NudeNet rather than replacing it. It
runs a vision language model: about a second per image in `fast` mode, several
in `fast-slow` (which also writes why a rule fired). The default is `cascade`
(`fast` for every image, `slow` for what `fast` flags; a frame with an exposed
NudeNet or EraX reading of 0.2 or more skips `fast` and is judged `slow`).
`cascade` and `slow` are modes of argus-cull's and agent-composer's `singguard` services; a stock
SingGuard server answers 400 to them. Curator then retries in `fast`, and once
`fast` works it judges the rest of the run in `fast` and logs
`singguard_mode_unsupported`; each such verdict records `thinking: "fast"`.
## Configuration

Configure it like [NudeNet](nudity.md), in the environment. `<AGENT_HOST>` is a placeholder for the host that runs the server:

```bash
ARGUS_CURATOR_SINGGUARD_URL=http://<AGENT_HOST>:9150   # its origin; /api/guard/check is appended
ARGUS_CURATOR_SINGGUARD_THINKING=cascade                 # or fast / fast-slow / slow
ARGUS_CURATOR_SINGGUARD_POLICY_FILE=/etc/curator/policy.txt  # optional: replaces the default rules
ARGUS_CURATOR_SINGGUARD_TIMEOUT=90                       # seconds per image (default 90)
ARGUS_CURATOR_SINGGUARD_WORKERS=2                        # concurrent requests (default 2)
ARGUS_CURATOR_SINGGUARD_CONTEXT=age,folder,caption,scene # optional: what the scan knows, sent as request text
```

## Context sent with each image

By default SingGuard sees the image alone, and under the adult policies it
then reasons about an age and a consent it cannot see. `ARGUS_CURATOR_SINGGUARD_CONTEXT`
sends what the scan already knows as the request text, opening with a line
that says it is recorded context and not a request (SingGuard reads a bare
description as a request to produce it): `age` is each face's MiVOLO estimate
(and a body-only estimate when no face matched); `folder` is the image's folder
under the scanned root; `caption` is an argus-lens `<stem>.txt` or XMP sidecar
(`photo.jpg.xmp`, else Lightroom's `photo.xmp`), cut to 600 characters;
`scene` is Places365; `nsfw` is the NudeNet content rating; `garments` is
Fashionpedia. Compose defaults to `age,folder,caption,scene`. The library
sends nothing until the variable is set. A wrong
age leans the verdict the wrong way, so measure it on labelled images
(`scripts/guard_eval.py --images`) before relying on it. The under-18 rule
(`minor-exposed`) does not read this text.

## Policies

It is opt-out in the same way (`--no-guard`, `"guard": {"enabled": false}`;
`--guard` makes it mandatory). The default rule A flags any sexual content,
adult nudity included. For an adult dataset, pick the built-in **`adult`**
policy per scan (`--guard-policy adult`, `"guard": {"policy": "adult"}`). Its
rule A covers only apparent minors, non-consent and sexual violence, and its
Safe rule names consensual adult nudity, SingGuard's own "Adult Content
(Non-criminal)" category, a single adult posing for the camera, and that
consent is not read from the image: a posed, camera-aware adult counts as
consenting unless hidden recording, restraint, coercion or an unconscious
person is shown. What passes is said only under Safe, never as a "not this
rule" line under a risk heading, which SingGuard reads as a match. Rules B to G
cover the same areas under the same letters, so `--guard-rules B,C` means the same thing under
both. They are written out as text, though, rather than left to SingGuard's
built-in wording, so verdicts can differ: a group photo of a poker game that
the default rule B flags passes under `adult`. `default` is the policy file
when one is set, else SingGuard's own rules. A request names a policy and never sends policy text. `GET /detectors`
lists the names in `singguard_policies`, and `GET /guard/policies` serves each
one's text: its `source` (`builtin`, `file`, or `singguard` when `default`
leaves SingGuard on its own rules and `text` is null) and the name results
record it under (`recorded`), so a verdict can be traced to its wording. The
policy file's text is served too, to anyone who can read from the server.

`adult-granular` (a letter per hard case: A minors, H non-consent, I sexual
violence), `adult-cautious` (doubt about age or consent counts as rule A), and
`adult-cinematic` (adult, but a depiction of crime or violence is Safe unless it
is sexual, shows gore or dismemberment, or teaches how to do it, with no sign of
fiction needed; legal adult entertainment is Safe) are under evaluation against
`adult`; don't build on them yet. `adult-cinematic` has its own letters: B to F
are Unethical Behavior, Cybersecurity, Agent Safety, Politically Sensitive and
Animal Abuse, and G is Criminal Instructions, so `--guard-rules B` picks a
different rule there than under `adult`. A letter is read against the policy's
current wording, so this also applies to `adult-cinematic` verdicts recorded
before argus-cortex 0.4.6.

## Comparing policies

To compare policies on your own images, judge one scan under several:
`"guard": {"policy": "adult", "variants": ["adult-granular", "adult-cautious"]}`
(CLI `--guard-variants`). `guard` stays the official verdict, and
`guard_variants` holds every policy's verdict by name, the official one
included. A missing name means that check failed. Each variant is one more
SingGuard pass per image. The filters read the official verdict unless
`guard_policies` lists others (CLI `--guard-policies`). An image is then
dropped if any listed verdict fails, or is missing, with the policy named in
the keep reason (`guard-unsafe:adult-granular`). The CSV report adds a
`guard_variants` column.

`scripts/guard_eval.py --url <singguard>` scores every policy on the text
probes in `eval/guard_probes.jsonl` and on an optional folder of images
sorted by expected verdict. `--scan-json <scan_id>.json --context age,scene`
sends each image with the context and crop of its row in a saved scan, since
context changes verdicts and a bare image may not reproduce what a scan saw.
`--candidate NAME=PARENT:FILE` tries a policy wording from a file, scored as
its parent policy. Rows and the table record the pass that decided each verdict.

## Results and filters

Each image records `guard.safe`, the rules that fired, the explanation, the
model and the policy: `default`, `adult`, or `custom-<sha256 prefix>` of the
file. The CSV report has `guard_safe` and `guard_rules`.

```bash
# Drop anything judged unsafe
argus-curator scan /data/inbox --guard-unsafe --copy-to /data/clean

# Drop only crime and unethical-behaviour verdicts, by letter or full name
argus-curator scan /data/inbox --guard-rules B,C --copy-to /data/clean
```

Over HTTP these are `guard_unsafe` and `guard_rules`. As with NudeNet, an
unchecked image is excluded whenever a SingGuard filter is set
(`guard-unchecked`); other reasons are `guard-unsafe` and `guard-rule-<rule>`
(e.g. `guard-rule-c`), or `guard-rule-unknown` for an unsafe verdict that names
no rule. `GET /detectors` reports `singguard`.
