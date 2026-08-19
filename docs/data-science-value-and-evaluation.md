# Data-science value and evaluation strategy

## Why the project is valuable

Data science is not only fitting a sophisticated model. The difficult, transferable work is often
problem definition, collection, privacy, schema design, data quality, entity resolution, label
construction, leakage prevention, baseline selection, evaluation, communication, and monitoring.
This project exercises all of those on a personally meaningful domain.

The strongest portfolio claim is not “I recreated Spotify.” It is:

> I built a reproducible local recommendation system from noisy lifetime implicit feedback and
> independently computed audio representations, evaluated it chronologically against transparent
> baselines, and documented where it succeeds and fails.

That statement is credible if the repository eventually contains a data-quality report, identity
match audit, exploratory analysis, model/feature cards, temporal evaluation, ablations, and a small
interactive or command-line demonstration without exposing private data.

## How this differs from Spotify

Spotify can use signals this project does not have: enormous cross-user collaborative behavior,
playlist and library graphs, impressions and ignored recommendations, searches, follows, explicit
feedback, session context, device/context information, editorial knowledge, release freshness, and
licensed access to its whole catalog. Its production objectives also include discovery, diversity,
retention, safety, marketplace constraints, and exploration—not simply “songs that sound alike.”

This project has different advantages: it is transparent, private, controllable, reproducible, and
can optimize a deliberately narrow personal objective. It can expose why a result matched, accept
weighted multi-song acoustic queries, enforce custom constraints, and run ablations that a consumer
Spotify interface cannot show. It is complementary to Spotify rather than a claim of parity.

## Research questions

Keep each phase tied to a testable question:

1. Can lifetime stream behavior produce a stable, explainable affinity label?
2. Do music embeddings retrieve human-judged similar tracks better than handcrafted descriptors?
3. Do embeddings plus behavior predict future high-affinity listening better than behavior-only,
   artist-only, and popularity/frequency baselines?
4. Which representation works best for rhythm, timbre, bass, harmony, and general similarity?
5. Do cluster assignments remain stable across seeds, samples, and representation choices?
6. Does explicit feedback improve ranking on later sessions without collapsing novelty?

## Minimum honest evaluation ladder

### 1. Data and identity quality

- Report rejected, duplicated, track, episode, unknown, and missing-URI rates.
- Report exact, fuzzy, ambiguous, unmatched, and audio-covered identity rates.
- Manually review a stratified sample and estimate precision by match tier.
- Never let fuzzy title/artist matches silently become ground truth.

### 2. Representation quality

- Build human-reviewed positive and negative song pairs.
- Include “same rhythm/different timbre,” “same timbre/different rhythm,” and hard negatives.
- Report Recall@K, mean reciprocal rank, and nDCG@K for embeddings, descriptors, and combinations.
- Run transformation tests and inspect failures rather than relying only on a projection plot.

### 3. Preference quality

- Split by time: train on earlier listening, validate on later listening, and test on the latest
  untouched period.
- Fit preprocessing only on training data.
- Treat unplayed songs as unknown, not automatic dislikes.
- Compare against repeat-frequency, recency, artist-affinity, and global/simple popularity baselines.
- Report ranking metrics such as Recall@K and nDCG@K plus calibration where probabilities are used.

### 4. Product usefulness

- Review blind A/B lists or collect lightweight accept/reject feedback.
- Track relevance, novelty, artist diversity, catalog coverage, and repeated failure modes.
- Preserve model/version metadata so a later run can reproduce every recommendation.

## Ablation table to target

| System | Audio | Behavior | Metadata | Purpose |
|---|---:|---:|---:|---|
| Frequency/recency baseline | No | Yes | No | Minimum personalized baseline |
| Artist baseline | No | Yes | Artist | Tests artist memorization |
| Handcrafted audio | DSP | No | No | Interpretable acoustic similarity |
| Embedding retrieval | EffNet | No | No | Learned acoustic/style similarity |
| Behavior ranker | No | Yes | Optional | Tests behavioral signal alone |
| Embedding + behavior | EffNet | Yes | Optional | Main candidate system |
| Full system | EffNet + DSP | Yes | Optional | Tests incremental value of descriptors/filters |

This table matters more for a student portfolio than using the newest algorithm without a baseline.
If a logistic model with clean temporal evaluation beats a complex model, that is a useful result.
If embeddings add no lift, diagnosing why is also a valid data-science result.

## What a strong final portfolio package contains

- a synthetic public demo path and a private real-data path;
- architecture and data-lineage diagrams;
- aggregate EDA with no personal raw records committed;
- a match-quality report and reviewed error taxonomy;
- a feature/model card and transformation tests;
- baseline and ablation results with confidence intervals where practical;
- one concise case study of a successful recommendation and one failure;
- reproducible commands, pinned configurations, and a limitations/ethics section.
