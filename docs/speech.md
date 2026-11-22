# Speech scans

A voice training set needs the decisions an image set does (what is usable,
what repeats, what is unsafe to train on), measured on audio. A speech scan
scores a folder of WAV/FLAC clips and their transcripts, can cluster them into
speakers, and exports the keepers with a transcript beside each clip and a
speech manifest ([below](#speakers)). It never rewrites a clip.

```bash
pip install "argus-curator[speech]"           # soundfile (WAV/FLAC) + onnxruntime for DNSMOS
argus-curator scan ./LJSpeech-1.1 --modality speech -v
argus-curator dnsmos-fetch                    # once, for --mos
argus-curator speaker-fetch                   # once, for --speakers
argus-curator vad-fetch                       # once, for --vad silero
pip install "argus-curator[lid-convert]" && argus-curator lid-convert   # once, for --language-id
argus-curator scan ./corpus --modality speech --mos --speakers --speech-config thresholds.json
```

Over HTTP it is the same `POST /scan/folder` (or `/scan/folder/stream`), with a
speech profile:

```json
{"folder": "corpus", "target_profile": {"modality": "speech", "target_category": "voice"},
 "speech": {"max_duration_s": 15, "min_snr_db": 25, "mos": true, "min_mos": 3.0}}
```

The profile is argus-cortex's `TargetProfile` with `modality: "speech"` and a
speech category (`voice`, `accent`, `language`, `speaking_style`). The request
is refused (422) if it also turns on an image check (`faces`, `scenes`,
`garments`, `embeddings`, `nsfw`, `guard`, `crops`, `prefilter`). The summary
keeps its rows in `clips` (`ClipResult`), with `results` empty, and the stream's
`chunk` events carry `clips` with `modality: "speech"`. `GET /scan/{id}`
paginates the clips. An image scan, the image scanner and `ManifestRow` still
refuse a speech profile.

## Transcripts

Read from inside the scanned folder only (a symlink out of it is ignored):

| `speech.transcripts` | Where | Notes |
|---|---|---|
| `sidecar` | `<clip>.txt` beside the clip | Up to 64 KB. |
| `ljspeech` | `metadata.csv` at the top | `id\|text\|normalized`; the normalized text wins. `id` is the clip's stem. |
| `common_voice` | `validated.tsv`, `train.tsv`, `dev.tsv`, `test.tsv` at the top | `path` matched by stem (convert Common Voice's MP3 to WAV/FLAC first); `client_id`, `accents`, `locale`, `age` and `gender` are carried as `speaker`, `accent`, `language`, `age`, `gender`. |
| `auto` (default) | a sidecar, else LJSpeech, else Common Voice | |

Common Voice's labels are read whatever the layout, so a sidecar can't hide
them. A clip whose corpus age label is under 18 (Common Voice's `teens` covers
13-19 and counts as a whole) is rejected as `corpus_under_18` before it is
decoded, and no setting turns that off. Nothing estimates age from a voice.

The Blizzard 2027 layout isn't read yet: its data isn't published
([#202](https://github.com/smk762/argus-curator/issues/202)).

## Checks

Each clip is read through `argus_cortex.limits.open_audio` (WAV/FLAC only,
size, channels, rate and duration checked before any sample is decoded). Every
threshold in `SpeechConfig` is a prefilter. A clip past one is rejected,
`reject_reason` names the first check it failed, `failed_checks` lists every
one, and `reject_detail` says what was measured. `null` turns a threshold off.

| Check | Measured | Default threshold |
|---|---|---|
| `unreadable` | not a WAV/FLAC clip the scan can decode (rates 8-96 kHz, at most 2 channels) | always |
| `duration` | length | `min_duration_s` 1, `max_duration_s` 20 (at most 120) |
| `sample_rate` | rate | `min_sample_rate` 16000 |
| `channels` | channels | `max_channels` 1 |
| `silent` | peak under -80 dBFS | always |
| `clipping` | share of samples at full scale | `max_clipping_ratio` 0.001 |
| `leading_silence`, `trailing_silence` | time before the first and after the last 30 ms frame within `silence_db` (-40) of the loudest | 1 s each |
| `transcript` | no transcript found | `require_transcript` true |
| `speaking_rate` | letters and digits per second of the clip less its edge silence | 4-30 chars/s |
| `snr` | speech frames' power over the rest (the `vad`'s mask, [below](#silero-vad)) | off (`min_snr_db`) |
| `speech_ratio` | share of 10 ms frames the `vad` marks as speech: the energy VAD (default) or Silero | off (`min_speech_ratio`) |
| `mos` | DNSMOS OVRL, 1-5 (with `mos: true`) | off (`min_mos`); with it set, a clip DNSMOS couldn't score is rejected |
| `cer` | CER of the transcript against an STT pass | off (`max_cer`); needs an STT engine, see below |
| `language` | the language-ID model disputes the corpus language label ([below](#language-labels)) | off: a dispute is flagged, and rejects only with `reject_language_mismatch` |

The score (0-1) weights the measured terms (`weight_mos` 0.5 on (OVRL-1)/4,
`weight_snr` 0.3 on SNR/`snr_ref_db`, `weight_speech_ratio` 0.2) and
renormalises over the ones measured.

**Near-duplicates.** Each clip gets a 64-bit spectrogram fingerprint: log
energies in 32 bands (100-3800 Hz) of the clip at 8 kHz, cut to its loud part,
resampled to 32 steps, with the 8x8 lowest DCT coefficients against their median
(pHash for a spectrogram). Clips within `duplicate_distance` bits (6; -1
disables) whose lengths are within 10% group together, and the best-scoring
one is kept as the representative. A re-encode, resample or gain change of a
clip lands within a few bits. A different take of the same line doesn't.

## Silero VAD

The default VAD is the energy VAD: a 30 ms frame is speech when it sits 10 dB
over the clip's 10th-percentile frame and within `silence_db` of its loudest,
with gaps under ~200 ms closed. It needs no model, but loudness is all it
reads, so noise hides speech from it and loud non-speech counts as speech.

With `vad: "silero"` (CLI `--vad silero`), the speech ratio and the SNR read
[Silero VAD](https://github.com/snakers4/silero-vad) instead (Silero Team,
MIT). `argus-curator vad-fetch` downloads its `silero_vad.onnx` (2.3 MB) from
release `v6.2.3`, pinned by commit `5cd79456` and sha256 `1a153a22...88e3`; a
file that doesn't hash to that is refused at load. Set
`ARGUS_CURATOR_VAD_MODEL` to keep it elsewhere. Every row records the VAD that
produced its numbers in `ClipResult.vad` (`energy` or `silero`). If Silero
can't read a clip, `vad_error` says why and `speech_ratio` and `snr_db` stay
null: the energy VAD is never swapped in, and with `min_snr_db` or
`min_speech_ratio` set the clip is rejected.

- **Input:** the clip is resampled to 16 kHz and read in 32 ms windows (512
  samples, with the previous window's last 64 before them). Each pass is one
  window, batch 1, carrying the model's state to the next, so every pass has
  the same shape.
- **Mask:** Silero's defaults. A window starts speech at probability 0.5 and
  speech ends below 0.35. Gaps under 100 ms are closed, and speech runs under
  250 ms are dropped. Each 10 ms scan frame takes the window its centre falls
  in.
- **SNR split:** the speech side is the mask. The noise side leaves out every
  window Silero gives 0.35 or more (a word too short to count as speech still
  isn't noise) and 50 ms either side of them. Without that guard band, the
  decay at the end of each word counted as noise, and clean clips read about
  20 dB low.
- **CPU only, no GPU lease.** Silero runs on onnxruntime's CPU provider alone,
  with one intra-op thread per session (the scan's `max_workers` clip threads
  are the parallelism). It never touches the GPU, so it takes no lease, holds
  no CUDA arena and has no `gpu-tenant.yaml` kind. One pass is launch-bound
  on CUDA: on a 3090 (onnxruntime-gpu 1.32) a pass took 1.6-2.0 ms, against
  0.12-0.24 ms on one CPU thread. Each pass needs the last one's state, so
  nothing can be batched to win that back. In a scan it costs about 5 ms per
  second of audio (~0.1 s for a 20 s clip), against ~0.05 ms for the energy
  VAD.

**Calibration.** 49 real takes (studio-style recordings, 24 kHz mono, 2-23 s,
770 s in all, read up to 20 s each) were measured clean. Each was then mixed
with white noise, pink noise and a non-speech "music" bed (a sustained chord
with tremolo and percussive clicks) at a known SNR, set over the clean
clip's speech frames. The music bed and pink noise were also measured alone,
at speech level. The medians, with the 10th-90th percentile for SNR:

| Condition | Speech ratio, energy | Speech ratio, Silero | SNR dB, energy | SNR dB, Silero |
|---|---|---|---|---|
| clean | 0.80 | 0.80 | 44.2 [39.9, 52.6] | 42.4 [33.6, 60.1] |
| white noise 30 dB | 0.77 | 0.80 | 28.8 [28.3, 29.5] | 29.7 [28.7, 30.0] |
| white noise 20 dB | 0.73 | 0.80 | 18.8 [18.2, 19.2] | 19.9 [19.8, 20.0] |
| white noise 10 dB | 0.48 | 0.80 | 7.6 [6.9, 8.5] | 9.9 [9.8, 10.1] |
| white noise 5 dB | 0.13 | 0.80 | 4.5 [3.6, 5.8] | 4.9 [4.8, 5.1] |
| white noise 0 dB | 0.00 | 0.81 | null on 42 of 49 | -0.1 [-0.4, 0.1] |
| pink noise 20 dB | 0.74 | 0.80 | 19.4 [18.9, 20.0] | 19.9 [19.3, 20.6] |
| pink noise 10 dB | 0.61 | 0.80 | 8.6 [7.8, 9.5] | 10.1 [9.2, 10.8] |
| pink noise 0 dB | 0.03 | 0.81 | 4.0 [2.9, 5.0] | 0.1 [-1.4, 1.9] |
| music bed 20 dB | 0.75 | 0.81 | 19.2 [18.2, 21.3] | 19.8 [18.8, 21.6] |
| music bed 10 dB | 0.67 | 0.79 | 9.3 [8.3, 11.9] | 9.8 [8.6, 11.7] |
| music bed 0 dB | 0.34 | 0.78 | 2.8 [2.0, 4.8] | 0.1 [-2.9, 2.6] |
| music bed alone | 0.46 | 0.00 | 5.5 [5.2, 5.6] | null on all 49 |
| pink noise alone | 0.00 | 0.00 | 6.2 on 32 of 49 | null on all 49 |

- On clean clips the two agree on the speech ratio: the median per-clip
  difference is 0.01, and the largest is 0.07.
- Under noise, Silero's speech ratio holds within 0.02 of the clean clip's
  down to 0 dB. The energy VAD's falls (0.48 at 10 dB white noise, 0.13 at
  5 dB), so `min_speech_ratio` on the energy VAD rejects noisy clips for
  noise, not for lacking speech.
- Silero's median SNR is within 0.5 dB of the mixed SNR from 30 dB down to
  0 dB. The energy VAD's reads 1-2.5 dB low at 10-20 dB, and at 0 dB it reads
  3-7 dB high, or nothing.
- Loud non-speech alone: the energy VAD calls 46% of the music bed speech, and
  gives pink noise alone an SNR. Silero finds no speech in either, so neither
  gets an SNR.
- The two VADs' SNRs aren't interchangeable at the top end. On takes padded
  with digital silence, Silero's noise side is all zero samples and its SNR
  reads the 100 dB cap, while the energy VAD's noise side takes in a word's
  onset and reads 49-58 dB. Set `min_snr_db` for the VAD you run. On 2 of the
  49 clean takes Silero read under 30 dB (16 and 27 dB) where the energy VAD
  read 52 and 31. On the first, a 60 ms click at the start of the take falls
  outside Silero's speech and counts as noise.

Those mixed SNRs were set over Silero's speech frames on the clean clip,
but the two VADs mark the same share of a clean clip (0.80 each), so the
reference barely favours either. The takes were copied to scratch for the
run and deleted afterwards. No audio is committed.

## DNSMOS

With `mos: true`, each clip that passed the checks above is scored by
Microsoft's DNSMOS P.835 (`sig_bak_ovr.onnx`, DNS-Challenge, code MIT and other
content CC BY 4.0): speech (`sig`), background (`bak`) and overall (`ovrl`).
`argus-curator dnsmos-fetch` downloads it (1.2 MB), pinned by commit
`82f1b17e` and sha256 `269fbebd...6edd`, and a file that doesn't hash to that is
refused at load. Set `ARGUS_CURATOR_DNSMOS_MODEL` to keep it elsewhere.

The model reads 9.01 s of 16 kHz audio. A clip is resampled, tiled up to
that length, and scored over at most 8 windows (1 s hops, thinned evenly), one
fixed-shape window per pass, so the capped CUDA arena never grows with a clip's
length. On CUDA it loads and runs under the host GPU lease as the kind `mos`
(about 270 MB loaded and 650 MB at peak on a 3090; `gpu-tenant.yaml`).

## CER and argus-echo

`speech.Transcriber` is the hook for a speech-to-text pass: mono samples and a
rate in, text out. Given one, `scan_speech_folder(..., transcriber=...)`
records `stt_transcript` and `cer` on every passing clip with a transcript, and
`max_cer` rejects clips past it. Curator ships no STT model. It will call
argus-echo's engine once that exists
([argus-echo#4](https://github.com/smk762/argus-echo/issues/4),
[#203](https://github.com/smk762/argus-curator/issues/203)). Until then a scan
that sets `max_cer` is refused.

## Speakers

With `speech.speakers: true` (CLI `--speakers`), each clip that passed gets a
speaker embedding, and the clips are clustered into speakers, the counterpart
of face clustering.

- **Model:** WeSpeaker's ECAPA-TDNN (`ECAPA_TDNN_GLOB_c512-ASTP-emb192`,
  large-margin fine-tuned, trained on VoxCeleb2 dev; WeSpeaker's code is
  Apache-2.0 and its VoxCeleb models CC BY 4.0). `argus-curator speaker-fetch`
  downloads WeSpeaker's own ONNX export (25 MB) from Hugging Face
  (`Wespeaker/wespeaker-ecapa-tdnn512-LM`), pinned by revision `a2f3dcb1` and
  sha256 `d71b85d9...10b0`; a file that doesn't hash to that is refused at
  load. Set `ARGUS_CURATOR_SPEAKER_MODEL` to keep it elsewhere.
- **Input:** 80-bin Kaldi filterbanks (25 ms frames every 10 ms, Hamming
  window, 16 kHz), mean-normalised over the clip, computed in numpy. A clip is
  cut into 4 windows of 3 s spread evenly across it (a short clip is tiled up
  to 3 s first), run as one fixed-shape batch, and its embedding is the mean of
  the windows', unit length. On CUDA it loads and runs under the host GPU
  lease as the kind `speakers` (298 MB loaded and 420 MB at peak on a 3090;
  `gpu-tenant.yaml`).
- **Clustering:** each clip joins the nearest speaker whose centroid is within
  `speaker_threshold` (cosine distance, 0.5), or starts one; then every clip
  is reassigned to its nearest centroid a few times. It is linear in clips
  times speakers, so a corpus of hundreds of thousands of clips clusters
  without a pairwise matrix. The threshold is loose on purpose: two speakers
  merged keeps the split speaker-disjoint, one speaker split in two could put
  them in both parts. `ClipResult.speaker_cluster` (`spk_1` is the largest),
  `speaker_distance` (to its centroid) and `speaker_error` record it;
  `ScanSummary.speaker_clusters` lists the speakers.
- **Corpus labels:** where the corpus names speakers (Common Voice's
  `client_id`), a cluster holding more than one label, or a label spread over
  clusters, is flagged: `SpeakerCluster.mismatch`, and `speaker_mismatch` on
  each clip whose label is the minority in its cluster or lives mostly in
  another. `ScanSummary.speaker_mismatches` counts them.
- **Memory only:** the embeddings stay in memory (`ScanSummary.embeddings`),
  held by the scan store per scan and dropped with it, as CLIP embeddings are.
  A centroid is a voiceprint, so none is stored.
- **Second-voice windows** ([#217](https://github.com/smk762/argus-curator/issues/217)):
  with `speech.second_voice: true` as well (it needs `speakers`), each clip's
  speech is also embedded in short windows, for the
  [second-voice check](#a-second-voice-in-a-clip) of a consented export. The
  scan's VAD (`speech.vad`, energy or Silero) marks the speech, the silence
  between it is dropped, and the speech is cut into windows of 1.5 s every
  0.75 s (speech under 1.5 s is tiled up to one window). They run 8 windows
  per pass, filled up by repeating the last pass's own windows, so every pass
  has the same shape and no more frames than a clip's 4 x 3 s pass: it runs
  under the same `speakers` lease and VRAM figure. A clip whose speech gives
  no windows records why in `ClipResult.second_voice_error`. The windows are
  voiceprints, so they are held exactly as the embeddings are
  (`ScanSummary.voice_windows`, float16, in memory only, dropped with the scan
  and with a moved clip). They cost about 0.5 KB per second of speech.

## Language labels

A speech scan carries the corpus's `language` label as given, and
spontaneous-speech corpora mislabel some clips. With `speech.language_id: true`
(CLI `--language-id`), each clip that passed is classified by a spoken
language-ID model, and a clip whose label the model rules out is flagged
([#208](https://github.com/smk762/argus-curator/issues/208)). The label is
never changed, and a flag alone rejects nothing.

- **Model:** SpeechBrain's VoxLingua107 ECAPA-TDNN
  (`speechbrain/lang-id-voxlingua107-ecapa`, Apache-2.0; trained on
  VoxLingua107, CC BY 4.0), 107 languages. SpeechBrain publishes it only as
  PyTorch checkpoints, so `argus-curator lid-convert` (`pip install
  "argus-curator[lid-convert]"`, which pulls torch and SpeechBrain for the
  conversion alone) fetches them pinned by revision `0253049a` and the sha256
  of each file (`embedding_model.ckpt` `ab750d5c...a19a`, `classifier.ckpt`
  `a50d9024...dbc`, `label_encoder.txt` `9f566d83...c17f`). It exports the
  encoder and classifier to ONNX (85 MB), and writes `<model>.json` beside the
  model naming those sources and the ONNX file's own sha256. At load, a model
  is refused if its record names other sources or doesn't match the file. The
  scan itself needs only onnxruntime (`[speech]`) and never imports torch. Set
  `ARGUS_CURATOR_LID_MODEL` (CLI `--lid-model`) to keep it elsewhere.
- **Input:** 60-bin log mel filterbanks of the clip at 16 kHz, as
  SpeechBrain's `Fbank` computes them (within 0.001 dB), mean-normalised over
  the clip. A clip is cut into 3 windows of 6 s spread evenly across it (a
  short clip is tiled up to 6 s first) and run as one fixed-shape batch. Its
  language probabilities are the mean of the windows'. On CUDA it loads and
  runs under the host GPU lease as the kind `language`. That is 352 MB loaded
  and 810 MB at peak on a 3090 with the 1 GB arena cap (`gpu-tenant.yaml`),
  and 7 ms a clip against 700 ms on CPU, so unlike Silero it leases a GPU.
- **What a clip records:** `language_predicted` (the model's top language,
  in its own codes: ISO 639-1 mostly, but Hebrew is `iw` and Javanese `jw`),
  `language_confidence` (that language's probability),
  `language_label_confidence` (the probability it gives the corpus label),
  `language_mismatch`, and `language_error` (why a passing clip got no
  reading). `ScanSummary.language_mismatches` counts the flagged clips. Only
  these numbers are kept: the 107-way probabilities are dropped once the clip
  is judged.
- **Reading the label:** a locale is read by its primary subtag (`pt-BR` is
  `pt`, `zh-CN` is `zh`). A few aliases are read too (`he`, `jv`, `nb`,
  `fil`, `cmn`), and so are the English language names the model lists. A
  label the model doesn't know (Common Voice's `fy-NL`, `yue`) gets no
  judgement. Languages the model can't tell apart by ear count as one:
  Bosnian, Croatian and Serbian; Norwegian and Nynorsk; Hindi and Urdu;
  Indonesian and Malay; Czech and Slovak.
- **The dispute:** a clip is `language_mismatch` when the model gives its
  label (with the label's group) a probability under
  `speech.language_dispute_below` (default **0.001**). The model has to rule
  the label out; disagreeing about its top guess isn't enough. On noisy or
  telephone audio the model is often unsure what a clip is but still sure what
  it is not (below).
- **Using it:** read `language_mismatch` and the two probabilities in the
  scan, or in `curation_report.csv` (`language_predicted`,
  `language_confidence`, `language_mismatch`). `speech.exclude_language_mismatch`
  (CLI `--exclude-language-mismatch`) leaves the flagged clips out at select
  and export (`language-mismatch`). `speech.reject_language_mismatch` makes a
  dispute a prefilter instead: the clip is rejected with the check
  `language`. Both are opt-in. The speech manifest row is unchanged.

**Calibration.** Measured with the pinned model on 272 real clips in 14
languages (fetched to a scratch folder and deleted afterwards). The read
speech is 105 clips of Multilingual LibriSpeech's test split (pt, es, fr, de,
it, nl, pl; 16 kHz audiobooks, median 15 s). The spontaneous speech is 167
clips of MInDS-14 (en-US, en-GB, en-AU, pt-PT, fr, de, es, it, nl, pl, ru, cs,
ko, zh; 8 kHz phone calls, median 7 s). Both are CC BY 4.0. A wrong label was
simulated by giving every clip each of the set's other languages in turn
(1,155 and 1,837 cases). The "hardest wrong label" is the other language the
model itself rated most likely for that clip, so it is the mislabel least
likely to be caught.

| `language_dispute_below` | Read: correct labels disputed | Read: wrong labels caught (hardest) | Phone: correct labels disputed | Phone: wrong labels caught (hardest) |
|---|---|---|---|---|
| 0.001 (default) | 0/105 | 96.7% (85/105) | 2/167 (1.2%) | 83.3% (57/167) |
| 0.005 | 0/105 | 97.7% (91/105) | 7/167 (4.2%) | 92.8% (86/167) |
| 0.01 | 0/105 | 98.4% (94/105) | 12/167 (7.2%) | 95.6% (108/167) |
| 0.05 | 0/105 | 99.1% (96/105) | 25/167 (15%) | 98.7% (146/167) |

The model's top guess is right on 95.2% of the read clips and only 62.3% of
the phone clips. Its usual misses are a neighbouring language: Spanish heard
as Galician or Catalan, English as Manx or Welsh, Czech as Slovak, Russian as
Ukrainian. That is why the dispute reads the label's probability. A rule
"flag when the top guess is another language at p >= 0.9" disputed 4 correct
phone labels but caught only 28% of wrong ones. At 0.001 the 2 correct phone
labels disputed were a French clip heard as Breton and a Russian one heard as
Lithuanian. End to end, a scan of 57 of these clips with 9 labels swapped
(pt/es, de/nl, fr/it, ru/pl) flagged 7 of the 9 and nothing else. It missed
German labelled Dutch (p(nl) 0.15) and a Russian phone clip labelled Polish
(p(pl) 0.009). On clean read speech the default could be raised to 0.01 with
no correct label disputed here. On phone-quality audio, read the
probabilities before excluding at a higher one. 14 languages are evidence of
the scale, not a calibration for your corpus. The model has only one `pt`,
so it can't split Brazilian from European Portuguese.

**Accent labels aren't checked.** No credible accent-ID model with a clear
permissive licence covers the corpora curated here. The English CommonAccent
ECAPA is English only. The one Portuguese pt-BR/pt-PT classifier was trained
partly on CC BY-NC-ND data, and was tested on studio TTS audio alone.
[#223](https://github.com/smk762/argus-curator/issues/223) records what was
checked and what would unblock it. Until then `accent` is the corpus label as
given.

## Select and export

`POST /scan/{id}/select` and `POST /export` take a speech scan. They read
`min_score`, `keep_similar`, `max_keep`, `split` / `split_part`, the transfer
fields and `speech` (`SpeechSelection`); any image filter set on a speech scan
is refused (400), and `speech` on an image scan too.

```json
{"scan_id": "...", "dest": "blizzard-train", "mode": "symlink",
 "split": {"held_out_fraction": 0.1, "group_by": "speaker"}, "split_part": "train",
 "speech": {"corpus": "Common Voice 17 pt", "license": "CC0-1.0",
            "max_hours_per_speaker": 2, "balance_accents": true}}
```

The CLI exports a speech scan the same way
([#207](https://github.com/smk762/argus-curator/issues/207)): `scan
--modality speech` with `--copy-to`, `--move-to` or `--symlink-to` builds the
same `ExportRequest` and runs it through `export_selection` once the scan is
saved, so it writes what `POST /export` writes. The request above is:

```bash
argus-curator scan corpus/ --modality speech --symlink-to blizzard-train \
  --corpus "Common Voice 17 pt" --license CC0-1.0 \
  --max-hours-per-speaker 2 --balance-accents \
  --held-out-fraction 0.1 --split-part train
```

| CLI flag | `ExportRequest` field |
|---|---|
| `--speaker-id`, `--accent`, `--language` (comma-separated) | `speech.speakers`, `speech.accents`, `speech.languages` |
| `--exclude-speaker-mismatch` | `speech.exclude_speaker_mismatch` (needs `--speakers`) |
| `--exclude-language-mismatch` | `speech.exclude_language_mismatch` (needs `--language-id`) |
| `--min-hours-per-speaker`, `--max-hours-per-speaker`, `--max-hours-per-accent`, `--balance-accents` | the `speech` hours caps |
| `--min-score`, `--keep-similar`, `--max-keep`, `--run-stats` | as for images |
| `--held-out` or `--held-out-fraction`, `--split-method`, `--split-seed`, `--split-part` | `split` (`group_by` is always `speaker`), `split_part` |
| `--corpus` and `--license` | `speech.corpus`, `speech.license` |
| `--voice-consent` and `--consent-delivery` | `speech.consent_record` (relative to the scanned folder, or absolute; needs `--speakers`, and turns `speech.second_voice` on for the scan), `speech.consent_delivery` |
| `--voice-consent-max-distance` | `speech.consent_max_distance` ([consented voice binding](#consented-voice-binding); needs `--voice-consent`) |
| `--second-voice-max-distance` | `speech.second_voice_max_distance` ([a second voice in a clip](#a-second-voice-in-a-clip); needs `--voice-consent`) |

The flags are checked before a clip is read: no provenance, half of one, both
kinds, an image filter (`--pose`, `--garment`, `--nsfw-threshold`,
`--carry-sidecars`, ...) or a speech export flag without a destination is
refused (exit 2). What can only be known from the scan (a split with too few
speakers, a consent record that doesn't cover the set) is refused after it,
before anything is transferred, and the scan stays saved. As with images, the
CLI has no move gate: it is your own shell.

**Selection.** A clip is kept when it passed, has a transcript, scores at
least `min_score` and isn't a near-duplicate (unless `keep_similar`). Then:

| `speech` field | Effect | Keep reason |
|---|---|---|
| `speakers`, `accents`, `languages` | keep only these (`speaker_id`, the corpus labels) | `speaker-excluded`, `accent-excluded`, `language-excluded` |
| `exclude_speaker_mismatch` | drop clips whose corpus label the clustering disputes (needs `speakers` on the scan) | `speaker-mismatch` |
| `exclude_language_mismatch` | drop clips whose corpus language label the language-ID model disputes (needs `language_id` on the scan) | `language-mismatch` |
| `min_hours_per_speaker` | drop a speaker with fewer hours | `speaker-under-min-hours` |
| `max_hours_per_speaker` | cap each speaker, best clips first | `speaker-cap` |
| `max_hours_per_accent`, `balance_accents` | cap each accent (at the smallest labelled accent's hours with `balance_accents`: clips with no accent label are capped too but never set it), taking clips from its speakers in turn | `accent-cap` |

`max_keep` then takes speakers in turn, each giving its best remaining clip
(`max-keep`), so a cap doesn't hand the set to a few voices. A clip whose
corpus age label reads under 18 is never kept, whatever the request; it is
already rejected at scan (`corpus_under_18`). Select returns each row's
`speaker_id` and `SelectResult.speech`: hours kept, by speaker, by accent, and
with a split hours and speakers per part.

**The split is speaker-disjoint.** Zero-shot TTS is judged on speakers the
model never heard, so `split.group_by` must be `"speaker"`. Clips sharing a
corpus speaker label *or* a speaker cluster form one group, transitively, so
neither reading of a speaker spans both parts even where they disagree. A
clip with neither (no corpus label, and a scan without `speakers`) refuses
the split: it could be anyone. Groups are stratified by their most common
accent and gender label: held-out speakers are shared across those strata in
proportion to how many speakers each has, and a stratum's share never takes
its last speaker. The split is curator's one `split_held_out` (#183), so
it is deterministic and `held_out` / `held_out_fraction` count clips. The
held-out part becomes proof's reference set: export each part to its own
`dest`.

**Provenance.** An export names where its clips came from, or is refused (400):

- a licensed corpus: `speech.corpus` and `speech.license`, written on every
  row, with `provenance` `{kind: "stock", source: corpus, license}`;
- a real person's voice outside one: `speech.consent_record` (an argus-cortex
  `ConsentRecord`, a path under the scanned folder) and
  `speech.consent_delivery`. It is read and checked at select and export for
  `training` of the subject's `voice` at that delivery (revoked, expired, or a
  use or delivery it doesn't cover is refused with cortex's reason). Rows
  carry `consent_release_sha256`, `consent_record_sha256` and a `consented`
  provenance that names no one: its `subject` is argus-cortex's subject ref
  `consent:<release_sha256>`, never the record's name, and its `source`
  doesn't name who collected the release. `speaker_label` and the transcript
  are your corpus's own text and are carried as they are. Which clips the
  record covers is [below](#consented-voice-binding).

Age is never estimated from a voice. A clip's corpus age label is carried as
`age`, and a corpus that gives none leaves it null.

**What is written.** Under `dest`, each clip by the same rules as images
(`copy`, `symlink`, `move` with the server's move gate; a symlink export into
the source never deletes the source), named by the same de-collision plan,
by stem. Beside each clip, `<stem>.txt` holds its transcript: written fresh
(never a link to the source's sidecar, so editing it never reaches the
source), before the clip, and removed if the clip's transfer fails. Then
`speech_manifest.jsonl` (one `SpeechManifestRow` per exported clip, below),
`curation_report.csv` (every clip and its keep reason) and `run_stats.json`
when asked. A move takes the moved clips out of the cached scan. Under
`move` a source sidecar transcript stays in the source folder.

## Consented voice binding

A voice consent record covers one voice: the voice of its
`reference_recordings`. At select and at export, curator reads the record
again and binds every clip to that voice
([#209](https://github.com/smk762/argus-curator/issues/209)), as a consented
face set binds images to the record's reference images (#133):

1. **The reference voice.** The record's reference recordings are found in
   the scan by sha256 (at least one must be a passing clip), and the speaker
   embeddings of those found are averaged into a unit centroid. The scan must have run with
   `speech.speakers` on, and the embeddings must still be held: they live in
   memory only, so a server restart drops them. With no usable embedding for
   any reference recording, the export is refused (400, "rescan with
   speech.speakers on"). It never falls back to the speaker clusters.
2. **Each clip.** The cosine distance from its speaker embedding to the
   centroid. A clip is kept only when that distance is at most
   `speech.consent_max_distance` (default **0.25**), whatever cluster it
   joined: a clip in the subject's cluster but past the distance is left out
   as `consent:other-speaker`, and a clip with no embedding as
   `consent:no-embedding`.
3. **A second voice.** A clip the distance keeps is then checked for a second
   voice inside it, and left out as `consent:second-voice` when it holds one
   ([below](#a-second-voice-in-a-clip)). The scan must have run with
   `speech.second_voice` on, and its windows must still be held; otherwise
   the export is refused (400). It never skips the check.
4. **For review.** The distances are reported as plain numbers:
   `consent_distance` and `second_voice_distance` in `curation_report.csv`
   and on each select row (`SelectRow`). The centroid is a voiceprint, so it's built
   for the one check and never stored, served or written. The binding is
   remade at every select and export, so nothing about it outlives the scan.
   The speech manifest row is unchanged.

**Why 0.25.** Measured with the pinned ECAPA model on 3 synthetic XTTS voices
(each voice's centroid from two of its clips, every other clip scored against
every centroid):

| Clips | Same voice, distance to its centroid | Another voice |
|---|---|---|
| 6 s (n=78) | median 0.120, p95 0.198, max 0.262 | min 0.336, median 0.49 |
| 3 s (n=153) | median 0.204, p95 0.292, max 0.402 | min 0.299, median 0.49 |

On 3 s clips the two overlap, so the ceiling sits below the closest other
voice measured at either length. Letting another voice in is the harmful
error, and leaving one of the subject's clips out is the safe one, so the
default leans strict: it keeps nearly all of the subject's 6 s clips, but
drops some of their 3 s clips, because longer clips give steadier
embeddings. Real recordings across sessions and microphones spread wider than
one synthetic voice does, so expect a real set to lose more of the subject's
clips. Read `consent_distance` in the report before you raise the ceiling, and
keep it well under the clustering's `speaker_threshold` (0.5), which is loose
on purpose. Three voices are evidence of the scale, not a calibration.

## A second voice in a clip

A clip's embedding is the mean of 4 windows of 3 s. When the subject speaks
most of a clip and someone else speaks a few seconds of it (an interjection, a
voice in the background), the mean stays near the subject's voice, and the
binding above keeps the clip. A consented voice export therefore also checks
every clip for a second voice
([#217](https://github.com/smk762/argus-curator/issues/217)), and leaves out
one that holds another voice as `consent:second-voice`.

**How.** The scan embeds each clip's speech in windows of 1.5 s every 0.75 s
([Speakers](#speakers), `speech.second_voice`). At select and at export, the
record's reference recordings' own windows are averaged into a unit centroid,
and each window of a clip is measured against it by cosine distance. The
clip's `second_voice_distance` is its farthest window's. A clip past
`speech.second_voice_max_distance` (default **0.40**) is left out. It compares
each window to the **consent reference**, not to the clip's own mean, because
in a consented set it's known whose voice the subject's is. The windows'
centroid is used rather than the whole-clip centroid the binding uses: a
1.5 s window reads farther from a whole clip's centroid than from other
1.5 s windows, and matching them halved the false flags on clean clips (26% to
13% at 0.40). The centroid is a voiceprint, so it's built for the one check
and never held, stored or served. Only the distance is kept, as a plain number.

**It never skips the check.** If the check can't run, the consented export is
refused (400), never passed: a scan without `speech.second_voice` (or without
`speakers`), a restart since the scan (the windows are held in memory only),
or reference recordings with no windows of their own (the VAD found no speech
in them). A clip with no windows (`second_voice_error`) is left out as
`consent:no-voice-windows`. Outside a consented export there's no reference
voice to measure against, so nothing is measured and the selection is
unchanged.

### Calibration (synthetic audio only)

> **Synthetic.** Everything below was measured on synthetic voices: XTTS
> takes, spliced and mixed in numpy. It hasn't yet been checked on real
> recordings of two people. The A/B against real mixed recordings is
> [#228](https://github.com/smk762/argus-curator/issues/228). Until it reports,
> read the numbers as the scale of what the check does, not as a guarantee.

Measured with the pinned ECAPA model on CPU and Silero VAD. There were five
distinct synthetic voices, from 12 XTTS takes and 32 more synthetic
recordings (clustered first to confirm they are five voices: same-voice files
0.01-0.27 apart, different voices 0.39-0.97 apart). Each subject's reference
was two 8 s cuts of one take. Every test clip came from **other takes**:

- **Clean controls:** 60 subject-only clips per subject, 4, 6, 8, 10 or 15 s.
- **Spliced:** a 6 or 10 s subject clip with each other voice cut in for 0.5,
  1, 2 or 4 s, at 10%, 50% or 90% of the clip, loudness-matched.
- **Overlapped:** an 8 s subject clip with each other voice mixed under it for
  1, 2, 4 or 8 s, at 0, 6 or 12 dB below the subject.

Two of the five voices vary so much between their takes that the binding
already leaves out 85-100% of their clean clips (0.25 ceiling), so they can't
serve as subjects for this check. The table is the other three voices as
subjects, with all four others as the second voice (1020 clips built, 612 of
them for these subjects). Each rate is over the clips the binding keeps, the ones only
this check can catch:

| Clips | n kept | flagged at 0.35 | **at 0.40** | at 0.45 |
|---|---|---|---|---|
| Clean subject clips (false flags) | 179 | 50% | **13%** | 6% |
| Second voice spliced in, 0.5 s | 71 | 92% | **75%** | 54% |
| spliced, 1 s | 71 | 100% | **92%** | 85% |
| spliced, 2 s | 71 | 100% | **100%** | 97% |
| spliced, 4 s | 61 | 100% | **100%** | 100% |
| Overlapped 1 s, at 0 / 6 / 12 dB under | 12 each | 83 / 83 / 50% | **83 / 50 / 33%** | 67 / 25 / 17% |
| Overlapped 2 s, at 0 / 6 / 12 dB under | 12 each | 100 / 100 / 83% | **92 / 83 / 67%** | 83 / 75 / 42% |
| Overlapped 4 s, at 0 / 6 / 12 dB under | 8 / 11 / 11 | 100 / 100 / 100% | **100 / 100 / 91%** | 100 / 82 / 73% |
| Overlapped 8 s, at 0 / 6 / 12 dB under | 1 / 5 / 12 | 100 / 100 / 100% | **100 / 100 / 100%** | 100 / 80 / 100% |

(The clean max was 0.532, p95 0.459, median 0.350. A 4 s or 8 s overlap is
often loud enough to move the whole clip's embedding, and the binding drops it
first, so n kept falls there. Counting both, the binding and this check
exclude 92-100% of the 4 s and 8 s mixtures at 0.40.)

**The default, 0.40, errs toward excluding.** Leaving out a clip of the
subject's is the safe error, and keeping a bystander's voice the harmful one.
So the default flags about one clean clip in eight (13%), where 0.45 would
halve that but miss more 0.5-1 s voices. 0.35 would catch nearly every 1 s
splice, but drops half the subject's clean clips. False flags grow with a
clip's length, because more windows give more chances for one to read far:
0-11% at 6 s, 0-30% at 10 s, and up to 57% for one voice at 15 s. Long
clips of the subject are the ones most often lost. Read
`second_voice_distance` in the report before raising the ceiling.

**What it reliably catches: a second voice speaking 2 s or more.** That
caught every one of the 2 s and 4 s splices, with all four other voices. At
1 s it caught 92% overall, but only 67% when the second voice was the one
closest to the subject (0.45 apart). At 0.5 s it caught 75%, down to 42% for
that closest voice. A voice mixed *under* the subject's is harder than one
taking a turn: a 1 s voice 12 dB below the subject is caught a third of the
time. The windowing matters as much as the threshold. Without the VAD, the
false flags rose to 58% at 0.40, since silence makes windows read far. With
1 s windows, 88% of clean clips were flagged. 2 s windows at their own
matching threshold (0.35: 8% false flags) caught 55% at 0.5 s and 85% at 1 s,
about what 1.5 s windows give at 0.45, so 1.5 s at 0.40 is the stricter of
the two. Requiring two
consecutive far windows (`SECOND_VOICE_RUN = 2`) cut false flags to 4% but
halved the 0.5 s catch rate, so the default is one window. The energy VAD
measured within a point or two of Silero.

Until #228 reports on real recordings, keep the scanned folder to clips of
the subject alone. The check is a backstop, not a licence to scan mixed
recordings.

## The speech manifest

`speech_manifest.jsonl` is its own file with its own row shape and version,
never mixed with image rows: `SpeechManifestRow`, version
`SPEECH_MANIFEST_VERSION` **1.0** (`GET /health`'s
`speech_manifest_version`, and every `ExportResult`). It moves independently
of the image `MANIFEST_VERSION` under the same rule (the major for a breaking
change, the minor for an additive one), checked against
`schema/speech-manifest-row.snapshot.json` by `make manifest-snapshot`. Every
key is always present; a field with no value is `null`.

| Field | Type | Meaning |
|---|---|---|
| `speech_manifest_version` | string | `"1.0"`. Refuse a major you don't know. |
| `modality` | string | always `speech` |
| `rel_path` | string | the clip's path in the scanned folder |
| `abs_path` | string | where the clip can be read (under `move`, the destination) |
| `exported_path`, `exported_abs_path` | string | the clip as written under the export root (de-collided) |
| `transcript_path`, `transcript_abs_path` | string | its `<stem>.txt` beside it |
| `target_profile` | object | the scan's `SpeechTargetProfile` (`modality: speech`, a speech category, a TTS `target_backend`) |
| `transcript` | string | the clip's transcript (what the `.txt` holds) |
| `duration_s`, `sample_rate`, `channels` | number | as decoded |
| `speaker_id` | string or null | the corpus speaker label, else the speaker cluster |
| `speaker_label`, `speaker_cluster` | string or null | each on its own |
| `speaker_mismatch` | boolean | the clustering disputes the corpus label |
| `accent`, `language`, `gender`, `age` | string or null | the corpus labels, verbatim; `age` null when unknown, never estimated |
| `score`, `similar_group` | number | as scanned |
| `split` | string or null | `train` or `held_out` with a split, else null |
| `corpus`, `license` | string | where every clip came from |
| `provenance` | object | argus-cortex's `Provenance` (`stock` for a corpus, `consented` under a record, whose `subject` is the subject ref `consent:<release_sha256>`, never a name) |
| `consent_release_sha256`, `consent_record_sha256` | string or null | a consented voice only |

argus-forge reading this file is
[argus-forge#58](https://github.com/smk762/argus-forge/issues/58) (for its
XTTS-v2 emitter, [argus-forge#56](https://github.com/smk762/argus-forge/issues/56)).
