# Hugging Face audio and the local explorer

The engine now supports natural-language sound retrieval, candidate filtering, and an interactive
local report alongside its existing history-based preference models. No audio, prompts, or personal
history are sent to Hugging Face during inference. The download command only fetches fixed public
model files; analysis never downloads models implicitly.

## Model choice

The existing Discogs-EffNet model remains the default music embedding backend and the input to
the six learned classifier heads. CLAP adds a distinct capability: audio and acoustic descriptions
share an embedding space, allowing a query such as “instrumental jazz with piano and bass.”
The selected [LAION CLAP checkpoint](https://huggingface.co/laion/clap-htsat-unfused) is listed as
Apache-2.0 and supported directly by [Transformers](https://huggingface.co/docs/transformers/model_doc/clap).
It complements the existing model; it has not been shown to outperform it on personal preference.

[MERT](https://huggingface.co/m-a-p/MERT-v1-95M) and
[MuQ](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter) are music representation alternatives,
both listed with noncommercial model licenses. They remain possible future ablations. Adding
more encoders cannot solve the missing waveform coverage for commercial tracks in the history.
This implementation prioritizes a working, evaluated audio-and-text path on the available CPU.

## Exact CLAP representation

- Repository: `laion/clap-htsat-unfused`.
- Revision: `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`.
- All config and tokenizer files are checked against upstream Git blob identities; the weights
  are checked against their pinned SHA-256 and byte length. Unexpected alternate weight files
  are rejected. Interrupted downloads retain already verified files.
- Built-in Transformers model code, local files only, restricted `weights_only=True` loading.
  No remote model code or unrestricted pickle loading is used.
- Mono 48 kHz audio, deterministic 10-second windows, and a final aligned window covering the
  tail. Short recordings use the processor's repeat padding; padding does not increase coverage.
- Each 512-dimensional window vector is normalized. Track pooling takes the arithmetic mean
  followed by L2 normalization. Tail overlap gives some final samples extra weight; this policy
  is explicit in the version and retained window starts.
- Feature: `clap_audio_embedding_v1`, source `huggingface_clap`, profile `local_clap`.
- Confidence is the existing duration-coverage heuristic, **not a calibrated accuracy estimate**.
  The profile requires confidence >= 0.5, so very short audio can be extracted but excluded from
  ranking. These exclusions are intentional.
- Descriptions exceeding 77 tokenizer tokens are rejected instead of silently truncated.
- The optional extra pins Transformers to the tested 5.8 minor series. Core commands and tests
  remain usable without installing PyTorch or Transformers.

## Extract and rank your permitted audio

```powershell
python -m pip install -e ".[clap]"
python -m myusic_engine download-clap-model
python -m myusic_engine analyze-audio data/private/audio_manifest.jsonl `
  --embedding-backend clap `
  --cache-dir data/interim/clap-checkpoints `
  --window-output-dir data/interim/clap-windows `
  --output data/processed/audio/clap-features.jsonl

python -m myusic_engine rank-candidates data/private/candidates.csv `
  --features data/processed/audio/clap-features.jsonl --profile local_clap `
  --text-query "Heavy bass, warm synthesizers and mellow vocals" `
  --output-dir data/processed/recommendations/clap
```

Use `--seed TRACK_ID=WEIGHT` repeatedly to combine song seeds. `--text-weight` sets the
description's contribution relative to those seeds. Each query vector is normalized before its
weighted sum. The query must match the selected embedding's exact name/source/version and
dimensions; Discogs vectors cannot accidentally enter the CLAP index. The report and run hash
retain the prompt, its vector and weight, filters, and requested playlist length.

Add `--model` and `--behavior-snapshots` to combine the query with an existing behavior model.
Train a new CLAP ablation using the existing `train-taste-model --profile local_clap --features ...`
workflow when these audio IDs overlap the history's temporal samples. Do not interpret a
description-match score as a probability that you will like the song.

`--cache-dir` checkpoints each completed recording. The key includes the audio byte hash, stable
track ID, rights basis, extractor settings, embedding version, classifier models, and window
output location. Changed audio/configuration gets a new checkpoint. Missing or changed window
files force recomputation. A later failure preserves earlier checkpoints and the previous complete
feature output. Checkpoints and all derived private files belong under ignored directories.

## Filters

Pass `--filters configs/filters.example.json` to require the example's tempo and bass ranges.
Copy and adjust that document for your own thresholds. Each rule identifies one exact feature
source and version; an AcousticBrainz tempo cannot silently replace a local DSP tempo. Required
missing or low-confidence measurements exclude the candidate with `missing_filter_feature`.
Out-of-range values use `feature_filter`; excluded artists use `excluded_artist`.

Numeric rules take `minimum` and/or `maximum`. Categorical rules take `allowed_values` plus an
optional `case_sensitive` flag, allowing key/mode or other text-feature filters. The document also
accepts `excluded_artists`. Learned mood/instrumental filters require actual learned observations;
CLAP text similarity does not fabricate those measurements.

## Browser report

```powershell
python -m myusic_engine build-report data/processed/recommendations/clap `
  --output data/processed/recommendations/clap/explorer.html
```

Optionally pass `--taste-map-assignments` from the same model used during ranking. Reports reject
a map from another model/profile and recommendation rows from another run. Open the HTML file
directly: no web server, account, telemetry, CDN, or upload is required. Track text is escaped and
inserted as text; a restrictive Content Security Policy permits only the bundled viewer scripts.
Search and filter tracks, inspect excluded candidates, select cluster points, and download the
displayed selected Spotify URIs. This is a saved snapshot; rerun ranking to change model inputs.
The HTML contains private recommendation data and must remain local unless deliberately shared.

## Reproducible real-audio check

```powershell
python scripts/run_demo.py --download
python -m myusic_engine validate-audio data/private/open-demo/audio_manifest.jsonl `
  --embedding-backend clap --output-dir data/processed/validation/clap
python -m myusic_engine validate-audio data/private/open-demo/audio_manifest.jsonl `
  --embedding-backend discogs --output-dir data/processed/validation/discogs
```

The demo uses short, attributed music excerpts from
[librosa's example recordings](https://librosa.org/doc/latest/recordings.html), verified against
its published registry. Source audio and license notices remain in ignored storage. The benchmark
compares original embeddings with a -6 dB version and a middle excerpt, then asks whether each
transformed recording uniquely retrieves its own original. Ties are not credited as successes.
Input hashes, per-recording measurements, and the embedding version are retained in its report.

The initial three-recording check on September 11, 2026 passed all six transformed-identity queries
for both backends. CLAP mean same-recording cosine was 0.9870 for gain reduction and 0.9873 for
excerpting; Discogs was 0.9933 and 0.9891. The demonstration subsequently gained a fourth recording
to support the clustering stage. These small checks establish working inference and elementary
robustness, not human semantic similarity or a lift in personal preference prediction.

The remaining quality gate is a representative permitted corpus matched to your history, plus
human similarity judgments and chronological embedding-vs-behavior evaluation. Live Spotify
publication also still requires a user-authorized OAuth token and explicit execution.
