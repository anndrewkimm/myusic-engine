# Spotify history to an audio taste model

The product goal is to learn personal taste from the supplied Spotify listening history,
attach real audio representations to those recordings, and use that model to rank release
candidates. Collecting or uploading an MP3 library is not a required user workflow.

## Published audio embeddings

[Music4All-Onion v2](https://zenodo.org/records/15394646), published May 13, 2025 under
CC BY 4.0, includes MAEST audio-transformer representations for 109,269 recordings.
The [authors' accompanying paper](https://doi.org/10.5334/tismir.235) describes features
from transformer block 7, initialized from PaSST and pretrained on Discogs20. These are
768-dimensional audio representations of 30-second research excerpts. They are distinct
from CLAP; their coordinates must never be mixed with locally computed CLAP vectors.

The importer downloads three public tables (approximately 358 MB total) and performs all
history matching locally. It requires an exact Spotify ID with one dataset match, plus
matching title, artist, and album after Unicode, case, and whitespace normalization.
Missing metadata, conflicting releases, and duplicate Spotify mappings are excluded.

The original Music4All metadata is mirrored on
[Hugging Face](https://huggingface.co/datasets/Leon299/music4all/tree/a391160e3e17f351d5ab2d05439a7d3d7f0440eb).
The mirror is pinned to that revision and all three input files are SHA-256 checked.
MAEST's published archive MD5 is `9d110b00b3b894aca907ea16e488a6e9`.
The metadata table supplies identity only: legacy Spotify feature values, popularity,
and release-year fields never enter the model.

```powershell
python scripts/import_history_embeddings.py --download

python -m myusic_engine train-taste-model `
  data/processed/modeling/temporal_taste_samples.jsonl `
  --features data/processed/audio/music4all-history/features.jsonl `
  --profile music4all_maest `
  --output-dir data/processed/models/music4all-maest-history `
  --modeling-config configs/modeling.yaml
```

Subsequent imports omit `--download`. No local audio files, Spotify OAuth, model-weight
download, or disclosure of listening history is needed for this route. The command
writes private `features.jsonl`, `identity_matches.jsonl`, and `coverage_report.json`.
Original tables, matches, feature vectors, training examples, models, and evaluation
results remain in Git-ignored paths.

The training command compares embedding-only, behavior-plus-embedding, and behavior
models, including a behavior baseline on exactly the same audio-covered examples.
Labels derive from later listening behavior; whole periods separate training,
validation, and testing. Selection uses validation results. This trains a personal
preference model on fixed audio embeddings; it does not fine-tune the MAEST encoder.

## Current evidence and remaining work

The September 13 private run successfully attached these embeddings to real history
and trained both audio variants. On the matched cohort, the initial combined model
performed worse than the behavior baseline on validation, so selection retained
behavior. The initial audio model is an experiment, not a demonstrated improvement.
Model regularization, limited coverage, and the older catalog need further evaluation.

Matching metadata does not independently verify the dataset authors' original audio
associations. Reviewable mapping records are retained. The confidence value of 0.95 is
the existing 30-second coverage convention, not an audited identity-confidence score.
The pretrained encoder's training chronology is not audited; the experiment is a
retrospective evaluation rather than proof of historical deployment performance.

This historical research catalog does not provide this week's new-release audio
coverage. A current candidate snapshot and an accessible source of comparable audio
features are still required before this becomes an audio-based Release Radar ranker.
Do not relabel an artist-only ranking as an audio recommendation or fill missing
embeddings with track-name embeddings.

Spotify's [API access changes](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api)
restrict audio features, audio analysis, and algorithmic playlists for new integrations.
A Release Radar URL alone therefore does not establish API access to its contents or
waveforms. The existing importer can use a saved playlist present in an account export;
the source and date of a future release snapshot must remain explicit.

## Attribution

Marta Moscati, Emilia Parada-Cabaleiro, Yashar Deldjoo, Eva Zangerle, and Markus Schedl.
Music4All-Onion: A Large-Scale Multi-faceted Content-Centric Music Recommendation
Dataset. CIKM 2022. DOI: 10.1145/3511808.3557656. Dataset v2:
10.5281/zenodo.15394646, CC BY 4.0. This project joins a subset by identity and
L2-normalizes the published MAEST vectors.

Also credit Igor Andre Pegoraro Santana and coauthors, Music4All: A New Music Database
and Its Applications (2020), for the original track metadata, and Andreas Peintner and
coauthors, Nuanced Music Emotion Recognition via a Semi-Supervised Multi-Relational
Graph Neural Network (2025), DOI: 10.5334/tismir.235, for the released neural features.
