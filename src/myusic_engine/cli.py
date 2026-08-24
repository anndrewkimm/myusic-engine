"""Command-line entry points for local pipeline stages."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from myusic_engine.ingest import HistoryIngestionError, prepare_history
from myusic_engine.matching import (
    IdentityPolicy,
    IdentityResolutionError,
    load_account_catalog,
    load_identity_policy,
    read_track_queries,
    resolve_identities,
    write_identity_resolution,
)
from myusic_engine.ranking import (
    AffinityConfig,
    BehaviorAggregationError,
    aggregate_track_behavior,
    load_affinity_config,
    load_duration_map,
    write_track_affinities,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="myusic-engine",
        description="Build privacy-safe inputs for the personal music intelligence pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    history_parser = subparsers.add_parser(
        "prepare-history",
        help="Normalize a Spotify Extended Streaming History JSON, directory, or ZIP.",
    )
    history_parser.add_argument("source", type=Path, help="Private export JSON, directory, or ZIP")
    history_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Ignored local directory for cleaned output",
    )
    history_parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop at the first invalid event instead of reporting and skipping it",
    )
    history_parser.add_argument(
        "--recommendation-config",
        type=Path,
        help="Optional recommendation YAML; built-in versioned defaults are used when omitted",
    )
    history_parser.add_argument(
        "--duration-map",
        type=Path,
        help="Optional JSON object mapping track URIs to duration milliseconds",
    )

    identity_parser = subparsers.add_parser(
        "resolve-identities",
        help="Resolve history tracks against URI-bearing local Spotify account metadata.",
    )
    identity_parser.add_argument(
        "affinities",
        type=Path,
        help="Private user_track_affinity.jsonl produced by prepare-history",
    )
    identity_parser.add_argument(
        "account_data",
        type=Path,
        help="Private Spotify ZIP or directory containing YourLibrary and Playlist JSON",
    )
    identity_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Ignored local directory for matches, coverage report, and review sample",
    )
    identity_parser.add_argument(
        "--matching-config",
        type=Path,
        help="Optional versioned identity-resolution YAML; safe defaults are used when omitted",
    )

    audio_parser = subparsers.add_parser(
        "analyze-audio",
        help="Extract objective descriptors and music embeddings from permitted local audio.",
    )
    audio_parser.add_argument(
        "manifest",
        type=Path,
        help="Private JSON Lines manifest mapping stable track IDs to permitted audio files",
    )
    audio_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Ignored local JSON Lines destination for feature observations",
    )
    audio_parser.add_argument(
        "--feature-config",
        type=Path,
        help="Optional versioned feature YAML; built-in defaults are used when omitted",
    )
    audio_parser.add_argument(
        "--embedding-model",
        type=Path,
        default=Path("artifacts/models/discogs-effnet-bsdynamic-1.onnx"),
        help="Pinned Discogs-EffNet ONNX model",
    )
    audio_parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Extract objective descriptors without running the embedding model",
    )
    audio_parser.add_argument(
        "--feature-head-model-dir",
        type=Path,
        help=(
            "Optional pinned Essentia classifier-head directory; enables learned audio scores"
        ),
    )
    audio_parser.add_argument(
        "--window-output-dir",
        type=Path,
        help="Optional ignored directory for private window-level embedding NPZ files",
    )

    model_parser = subparsers.add_parser(
        "download-embedding-model",
        help="Download and verify the pinned noncommercial Discogs-EffNet ONNX model.",
    )
    model_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/models/discogs-effnet-bsdynamic-1.onnx"),
        help="Ignored local model destination",
    )
    model_parser.add_argument(
        "--accept-noncommercial-license",
        action="store_true",
        help="Acknowledge the model's CC BY-NC-SA 4.0 license",
    )

    feature_head_parser = subparsers.add_parser(
        "download-feature-head-models",
        help="Download and verify the pinned noncommercial learned audio score models.",
    )
    feature_head_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/models/feature-heads"),
        help="Ignored local model-pack destination",
    )
    feature_head_parser.add_argument(
        "--accept-noncommercial-license",
        action="store_true",
        help="Acknowledge the models' CC BY-NC-SA 4.0 license",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected command and return a process exit code."""

    args = _parser().parse_args(argv)
    if args.command == "prepare-history":
        try:
            config = (
                load_affinity_config(args.recommendation_config)
                if args.recommendation_config
                else AffinityConfig()
            )
            durations = load_duration_map(args.duration_map) if args.duration_map else None
            result = prepare_history(args.source, args.output_dir, strict=args.strict)
            affinities = aggregate_track_behavior(
                result.events,
                durations_ms=durations,
                config=config,
            )
            write_track_affinities(affinities, args.output_dir / "user_track_affinity.jsonl")
        except (BehaviorAggregationError, HistoryIngestionError, OSError) as exc:
            print(f"History preparation failed: {exc}", file=sys.stderr)
            return 2
        counts = result.report.media_counts
        print(
            "Prepared "
            f"{result.report.events_written} events "
            f"({counts.get('track', 0)} tracks, "
            f"{counts.get('episode', 0)} episodes, "
            f"{counts.get('unknown', 0)} unknown)."
        )
        print(
            f"Removed {result.report.duplicate_events_removed} duplicates; "
            f"rejected {result.report.records_rejected} invalid records."
        )
        print(f"Aggregated {len(affinities)} track affinity records.")
        print(f"Wrote privacy-cleaned output to {args.output_dir}")
        return 0
    if args.command == "resolve-identities":
        try:
            policy = (
                load_identity_policy(args.matching_config)
                if args.matching_config
                else IdentityPolicy()
            )
            queries = read_track_queries(args.affinities)
            catalog = load_account_catalog(args.account_data)
            identity_result = resolve_identities(queries, catalog, policy=policy)
            write_identity_resolution(
                identity_result,
                args.output_dir,
                review_sample_per_status=policy.review_sample_per_status,
            )
        except (IdentityResolutionError, OSError) as exc:
            print(f"Identity resolution failed: {exc}", file=sys.stderr)
            return 2
        counts = identity_result.report.status_counts
        print(
            f"Resolved {identity_result.report.resolved_count} of "
            f"{identity_result.report.queries_seen} tracks; "
            f"{counts['fuzzy']} fuzzy, {counts['ambiguous']} ambiguous, "
            f"{counts['unmatched']} unmatched."
        )
        print(
            f"Exact IDs cover {identity_result.report.resolved_play_rate:.1%} of plays and "
            f"{identity_result.report.resolved_ms_played_rate:.1%} of listening time."
        )
        print(
            f"Catalog contained {identity_result.report.catalog_unique_tracks} unique Spotify "
            f"tracks from {len(identity_result.report.catalog_source_files)} local account files."
        )
        print(f"Wrote private identity outputs to {args.output_dir}")
        return 0
    if args.command == "download-embedding-model":
        from myusic_engine.embeddings import EmbeddingExtractionError, download_model

        try:
            destination = download_model(
                args.output,
                accept_noncommercial_license=args.accept_noncommercial_license,
            )
        except (EmbeddingExtractionError, OSError) as exc:
            print(f"Embedding model download failed: {exc}", file=sys.stderr)
            return 2
        print(f"Downloaded and SHA-256 verified the pinned model at {destination}")
        return 0
    if args.command == "download-feature-head-models":
        from myusic_engine.features.learned import (
            LearnedFeatureError,
            download_feature_head_models,
        )

        try:
            destinations = download_feature_head_models(
                args.output_dir,
                accept_noncommercial_license=args.accept_noncommercial_license,
            )
        except (LearnedFeatureError, OSError) as exc:
            print(f"Feature-head model download failed: {exc}", file=sys.stderr)
            return 2
        print(
            f"Downloaded and SHA-256 verified {len(destinations)} pinned models "
            f"under {args.output_dir}"
        )
        return 0
    if args.command == "analyze-audio":
        from myusic_engine.audio import AudioInputError, read_audio_manifest
        from myusic_engine.embeddings import (
            DiscogsEffnetOnnxBackend,
            EmbeddingExtractionError,
        )
        from myusic_engine.features.config import (
            FeatureConfigError,
            ObjectiveFeatureConfig,
            load_objective_feature_config,
        )
        from myusic_engine.features.learned import (
            DiscogsEffnetFeatureHeadBackend,
            LearnedFeatureError,
        )
        from myusic_engine.features.objective import AudioAnalysisError
        from myusic_engine.features.pipeline import analyze_audio_assets
        from myusic_engine.features.records import FeatureRecordError, write_feature_observations

        try:
            audio_config = (
                load_objective_feature_config(args.feature_config)
                if args.feature_config
                else ObjectiveFeatureConfig()
            )
            backend = (
                None if args.skip_embeddings else DiscogsEffnetOnnxBackend(args.embedding_model)
            )
            feature_head_backend = (
                DiscogsEffnetFeatureHeadBackend(args.feature_head_model_dir)
                if args.feature_head_model_dir is not None
                else None
            )
            assets = read_audio_manifest(args.manifest)
            audio_result = analyze_audio_assets(
                assets,
                config=audio_config,
                embedding_backend=backend,
                feature_head_backend=feature_head_backend,
                window_output_dir=args.window_output_dir,
            )
            write_feature_observations(audio_result.observations, args.output)
        except (
            AudioAnalysisError,
            AudioInputError,
            EmbeddingExtractionError,
            FeatureConfigError,
            FeatureRecordError,
            LearnedFeatureError,
            OSError,
        ) as exc:
            print(f"Audio analysis failed: {exc}", file=sys.stderr)
            return 2
        print(
            f"Analyzed {audio_result.tracks_analyzed} tracks into "
            f"{len(audio_result.observations)} feature observations."
        )
        if backend is not None:
            print(f"Aggregated {audio_result.embedding_windows} Discogs-EffNet windows.")
        if feature_head_backend is not None:
            print(f"Emitted {audio_result.learned_scores} learned audio scores.")
        print(f"Wrote feature observations to {args.output}")
        return 0
    return 2
