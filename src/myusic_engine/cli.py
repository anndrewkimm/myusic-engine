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

    playlist_import_parser = subparsers.add_parser(
        "import-account-playlist",
        help="Import one named playlist from a private Spotify account-data ZIP or directory.",
    )
    playlist_import_parser.add_argument(
        "account_data",
        type=Path,
        help="Private Spotify account-data ZIP or directory containing Playlist JSON",
    )
    playlist_import_parser.add_argument(
        "--playlist-name",
        required=True,
        help="Exact playlist name from the account export (case-insensitive)",
    )
    playlist_import_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Ignored directory for candidates and aggregate import provenance",
    )

    external_parser = subparsers.add_parser(
        "map-musicbrainz",
        help="Map history or candidate metadata to MusicBrainz remotely or from a local dump.",
    )
    external_parser.add_argument(
        "input_file",
        type=Path,
        help="Private affinity or candidate file selected by --input-kind",
    )
    external_parser.add_argument(
        "--input-kind",
        choices=("affinities", "candidates"),
        default="affinities",
        help="Read history affinities (default) or a candidate JSONL/CSV/text file",
    )
    external_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Ignored local directory for matches, coverage, and review samples",
    )
    external_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/provider-cache"),
        help="Ignored resumable provider-response cache",
    )
    external_parser.add_argument(
        "--limit",
        type=int,
        help="Optionally process only the highest-weight inputs",
    )
    external_parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached responses only and fail on a cache miss",
    )
    external_parser.add_argument(
        "--canonical-dump",
        type=Path,
        help="Official canonical MusicBrainz CSV or tar.zst; keeps query metadata local",
    )

    acousticbrainz_parser = subparsers.add_parser(
        "fetch-acousticbrainz",
        help="Fetch frozen CC0 audio descriptors for exact MusicBrainz matches.",
    )
    acousticbrainz_parser.add_argument(
        "external_matches",
        type=Path,
        help="Private external_identity_matches.jsonl from map-musicbrainz",
    )
    acousticbrainz_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Ignored local directory for feature observations and coverage",
    )
    acousticbrainz_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/provider-cache"),
        help="Ignored resumable provider-response cache",
    )
    acousticbrainz_parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Official bulk request size, from 1 through 25",
    )
    acousticbrainz_parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached responses only and fail on a cache miss",
    )

    temporal_parser = subparsers.add_parser(
        "build-taste-dataset",
        help="Build leakage-safe chronological labels and point-in-time behavior features.",
    )
    temporal_parser.add_argument(
        "history",
        type=Path,
        help="Private listening_events.jsonl produced by prepare-history",
    )
    temporal_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Ignored local directory for temporal samples, snapshots, and aggregate report",
    )
    temporal_parser.add_argument(
        "--modeling-config",
        type=Path,
        default=Path("configs/modeling.yaml"),
        help="Versioned temporal split, label, model, and audio-profile YAML",
    )

    taste_model_parser = subparsers.add_parser(
        "train-taste-model",
        help="Train chronological behavior baselines and fair-cohort audio ablations.",
    )
    taste_model_parser.add_argument(
        "samples",
        type=Path,
        help="Private temporal_taste_samples.jsonl",
    )
    taste_model_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Ignored local model, held-out prediction, and evaluation directory",
    )
    taste_model_parser.add_argument(
        "--modeling-config",
        type=Path,
        default=Path("configs/modeling.yaml"),
        help="Versioned temporal split, model, and audio-profile YAML",
    )
    taste_model_parser.add_argument(
        "--profile",
        help="Audio profile name; omit to train behavior-only baselines now",
    )
    taste_model_parser.add_argument(
        "--features",
        type=Path,
        action="append",
        default=[],
        help="Source-tagged feature JSONL; repeat to combine lawful feature sources",
    )

    taste_map_parser = subparsers.add_parser(
        "build-taste-map",
        help="Compare K-Means/HDBSCAN and emit PCA taste-map coordinates.",
    )
    taste_map_parser.add_argument(
        "--features",
        type=Path,
        action="append",
        required=True,
        help="Source-tagged feature JSONL; repeat when a profile spans files",
    )
    taste_map_parser.add_argument("--profile", required=True, help="Audio profile name")
    taste_map_parser.add_argument(
        "--representation",
        choices=("descriptors", "embedding", "combined"),
        default="embedding",
    )
    taste_map_parser.add_argument("--minimum-k", type=int, default=2)
    taste_map_parser.add_argument("--maximum-k", type=int, default=12)
    taste_map_parser.add_argument(
        "--output-dir", type=Path, required=True, help="Ignored taste-map output directory"
    )
    taste_map_parser.add_argument(
        "--modeling-config", type=Path, default=Path("configs/modeling.yaml")
    )

    candidate_parser = subparsers.add_parser(
        "rank-candidates",
        help="Rank candidate CSV/JSONL/URI text and emit an ordered Spotify URI handoff.",
    )
    candidate_parser.add_argument("candidates", type=Path)
    candidate_parser.add_argument(
        "--features",
        type=Path,
        action="append",
        default=[],
        help="Optional source-tagged feature JSONL; repeat when an audio profile is used",
    )
    candidate_parser.add_argument(
        "--profile",
        help="Optional audio profile name; omit for behavior-only ranking",
    )
    candidate_parser.add_argument(
        "--seed",
        action="append",
        default=[],
        metavar="TRACK_ID=WEIGHT",
        help="Weighted acoustic seed; repeat for multi-seed retrieval",
    )
    candidate_parser.add_argument("--model", type=Path, help="Optional selected_model.json")
    candidate_parser.add_argument(
        "--behavior-snapshots",
        type=Path,
        help="Optional behavior_snapshots.jsonl for current preference and novelty",
    )
    candidate_parser.add_argument(
        "--taste-map-assignments",
        type=Path,
        help="Optional taste_map_assignments.jsonl for candidate and seed cluster context",
    )
    candidate_parser.add_argument("--top-k", type=int, default=50)
    candidate_parser.add_argument(
        "--output-dir", type=Path, required=True, help="Ignored recommendation output directory"
    )
    candidate_parser.add_argument(
        "--modeling-config", type=Path, default=Path("configs/modeling.yaml")
    )
    candidate_parser.add_argument(
        "--recommendation-config",
        type=Path,
        default=Path("configs/recommendation.yaml"),
    )

    feedback_parser = subparsers.add_parser(
        "record-feedback",
        help="Append an explicit local outcome for a recommendation run.",
    )
    feedback_parser.add_argument("feedback_log", type=Path)
    feedback_parser.add_argument("recommendation_run_id")
    feedback_parser.add_argument("track_id")
    feedback_parser.add_argument(
        "outcome",
        choices=("accepted", "rejected", "saved", "skipped", "listened"),
    )
    feedback_parser.add_argument(
        "--at",
        help="Optional timezone-aware ISO timestamp; current UTC time is used when omitted",
    )

    publish_parser = subparsers.add_parser(
        "publish-spotify-playlist",
        help="Plan or explicitly publish an ordered URI handoff to a private Spotify playlist.",
    )
    publish_parser.add_argument(
        "uri_file",
        type=Path,
        help="Ordered spotify_playlist_uris.txt emitted by rank-candidates",
    )
    publish_parser.add_argument("--name", required=True, help="Name for the private playlist")
    publish_parser.add_argument(
        "--description",
        default="Generated by Myusic Engine.",
        help="Optional private-playlist description",
    )
    publish_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Ignored directory for the deterministic plan and resumable receipt",
    )
    publish_parser.add_argument(
        "--execute",
        action="store_true",
        help="Create or resume the remote playlist; omission writes a dry-run plan only",
    )
    publish_parser.add_argument(
        "--access-token-env",
        default="SPOTIFY_ACCESS_TOKEN",
        help="Environment variable containing an authorized OAuth token",
    )
    publish_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTPS request timeout used only with --execute",
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
        help=("Optional pinned Essentia classifier-head directory; enables learned audio scores"),
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
    if args.command == "import-account-playlist":
        from myusic_engine.matching import (
            load_account_playlist,
            write_account_playlist_report,
        )
        from myusic_engine.ranking import CandidateInputError, CandidateTrack, write_candidates

        try:
            account_playlist = load_account_playlist(args.account_data, args.playlist_name)
            candidates = tuple(
                CandidateTrack(
                    track_id=track.track_uri,
                    spotify_uri=track.track_uri,
                    track_name=track.track_name,
                    artist_name=track.artist_name,
                    album_name=track.album_name,
                )
                for track in account_playlist.tracks
            )
            write_candidates(candidates, args.output_dir / "candidates.jsonl")
            write_account_playlist_report(
                account_playlist, args.output_dir / "account_playlist_import_report.json"
            )
        except (CandidateInputError, IdentityResolutionError, OSError) as exc:
            print(f"Account playlist import failed: {exc}", file=sys.stderr)
            return 2
        print(
            f"Imported {len(candidates)} unique tracks from playlist "
            f"{account_playlist.playlist_name!r}."
        )
        if account_playlist.last_modified_date is not None:
            print(f"Account-export playlist snapshot: {account_playlist.last_modified_date}")
        print(f"Wrote private candidates to {args.output_dir}")
        return 0
    if args.command == "map-musicbrainz":
        from myusic_engine.matching import (
            CanonicalDumpError,
            ExternalIdentityPolicy,
            TrackQuery,
            build_canonical_dump_mapper,
            resolve_external_identities,
            write_external_identity_resolution,
        )
        from myusic_engine.providers import (
            JsonCacheTransport,
            ListenBrainzMappingClient,
            MusicBrainzMapper,
            ProviderError,
        )
        from myusic_engine.ranking import CandidateInputError, read_candidates

        try:
            if args.input_kind == "candidates":
                candidates = read_candidates(args.input_file)
                queries = tuple(
                    TrackQuery(
                        source_track_id=candidate.track_id,
                        source_identity_source=(
                            "spotify_uri" if candidate.spotify_uri is not None else "metadata_hash"
                        ),
                        track_uri=candidate.spotify_uri,
                        track_name=candidate.track_name,
                        artist_name=candidate.artist_name,
                        album_name=candidate.album_name,
                    )
                    for candidate in candidates
                )
                progress_label = "playlist tracks"
            else:
                queries = read_track_queries(args.input_file)
                progress_label = "history tracks"

            mapper: MusicBrainzMapper
            if args.canonical_dump is not None:
                external_policy = ExternalIdentityPolicy(
                    policy_version="musicbrainz_canonical_dump_exact_metadata_v1"
                )

                def scan_progress(rows_scanned: int) -> None:
                    print(
                        f"Scanned {rows_scanned:,} canonical MusicBrainz rows locally...",
                        flush=True,
                    )

                mapper, scan_report = build_canonical_dump_mapper(
                    queries,
                    args.canonical_dump,
                    maximum_tracks=args.limit,
                    progress=scan_progress,
                )
                provider_name = scan_report.provider_name
                print(
                    f"Local canonical scan retained {scan_report.exact_query_keys} unique exact "
                    f"metadata keys and rejected {scan_report.ambiguous_query_keys} ambiguous "
                    "keys."
                )
            else:
                external_policy = ExternalIdentityPolicy()
                transport = JsonCacheTransport(
                    args.cache_dir,
                    minimum_interval_seconds=0.1,
                    offline=args.offline,
                )
                mapper = ListenBrainzMappingClient(transport)
                provider_name = "listenbrainz_labs_musicbrainz_mapper"

            def mapping_progress(processed: int, total: int) -> None:
                if processed == total or processed % 25 == 0:
                    print(f"Mapped {processed}/{total} {progress_label}...", flush=True)

            external_result = resolve_external_identities(
                queries,
                mapper,
                policy=external_policy,
                maximum_tracks=args.limit,
                progress=mapping_progress,
                provider_name=provider_name,
            )
            write_external_identity_resolution(
                external_result,
                args.output_dir,
                review_sample_per_status=external_policy.review_sample_per_status,
            )
        except (
            CandidateInputError,
            CanonicalDumpError,
            IdentityResolutionError,
            ProviderError,
            OSError,
        ) as exc:
            print(f"MusicBrainz mapping failed: {exc}", file=sys.stderr)
            return 2
        counts = external_result.report.status_counts
        print(
            f"Accepted {counts['exact']} exact MusicBrainz recordings; "
            f"retained {counts['fuzzy']} fuzzy and {counts['unmatched']} unmatched for review."
        )
        if args.input_kind == "candidates":
            print(
                f"Exact MBIDs cover {external_result.report.exact_play_rate:.1%} "
                "of processed candidate tracks."
            )
        else:
            print(
                f"Exact MBIDs cover {external_result.report.exact_play_rate:.1%} of processed "
                f"plays and {external_result.report.exact_ms_played_rate:.1%} of listening time."
            )
        print(f"Wrote private external identity outputs to {args.output_dir}")
        return 0
    if args.command == "fetch-acousticbrainz":
        from myusic_engine.features import (
            FeatureRecordError,
            fetch_acousticbrainz_features,
            write_acousticbrainz_result,
        )
        from myusic_engine.matching import read_external_identity_matches
        from myusic_engine.providers import (
            AcousticBrainzClient,
            JsonCacheTransport,
            ProviderError,
        )

        try:
            transport = JsonCacheTransport(
                args.cache_dir,
                minimum_interval_seconds=1.0,
                offline=args.offline,
            )
            provider = AcousticBrainzClient(transport)
            external_matches = read_external_identity_matches(args.external_matches)

            def feature_progress(processed: int, total: int) -> None:
                if processed == total or processed % 10 == 0:
                    print(f"Fetched {processed}/{total} AcousticBrainz batches...", flush=True)

            acoustic_result = fetch_acousticbrainz_features(
                external_matches,
                provider,
                batch_size=args.batch_size,
                progress=feature_progress,
            )
            write_acousticbrainz_result(acoustic_result, args.output_dir)
        except (FeatureRecordError, IdentityResolutionError, ProviderError, OSError) as exc:
            print(f"AcousticBrainz feature fetch failed: {exc}", file=sys.stderr)
            return 2
        acoustic_report = acoustic_result.report
        print(
            f"AcousticBrainz covered {acoustic_report.low_level_tracks_covered}/"
            f"{acoustic_report.exact_tracks_considered} exact tracks at low level and "
            f"{acoustic_report.high_level_tracks_covered}/"
            f"{acoustic_report.exact_tracks_considered} at high level."
        )
        print(
            f"Wrote {acoustic_report.observations_written} source-tagged observations to "
            f"{args.output_dir}"
        )
        return 0
    if args.command == "build-taste-dataset":
        from myusic_engine.ingest import ProcessedHistoryError, iter_normalized_events
        from myusic_engine.modeling import (
            ModelingConfigError,
            TemporalDatasetError,
            build_temporal_dataset,
            load_modeling_config,
            write_temporal_dataset,
        )

        try:
            modeling_config = load_modeling_config(args.modeling_config)
            temporal_result = build_temporal_dataset(
                iter_normalized_events(args.history),
                config=modeling_config.temporal,
            )
            write_temporal_dataset(temporal_result, args.output_dir)
        except (
            ModelingConfigError,
            ProcessedHistoryError,
            TemporalDatasetError,
            OSError,
        ) as exc:
            print(f"Temporal dataset build failed: {exc}", file=sys.stderr)
            return 2
        temporal_report = temporal_result.report
        print(
            f"Built {len(temporal_result.samples)} labeled track-period examples across "
            f"{temporal_report.periods_with_samples} whole chronological periods."
        )
        print(
            "Split periods: "
            f"{temporal_report.split_period_counts['train']} train, "
            f"{temporal_report.split_period_counts['validation']} validation, "
            f"{temporal_report.split_period_counts['test']} test."
        )
        print(
            f"Retained {temporal_report.known_positive_events} positive and "
            f"{temporal_report.known_negative_events} early-skip event signals; "
            "unknowns were not dislikes."
        )
        print(f"Wrote private temporal outputs to {args.output_dir}")
        return 0
    if args.command == "train-taste-model":
        from myusic_engine.features import FeatureRecordError, read_feature_observations
        from myusic_engine.modeling import (
            ModelingConfigError,
            RepresentationError,
            TasteTrainingError,
            TemporalDatasetError,
            load_modeling_config,
            read_temporal_samples,
            train_taste_models,
            write_taste_training_result,
        )

        try:
            modeling_config = load_modeling_config(args.modeling_config)
            if args.profile is not None and args.profile not in modeling_config.profiles:
                raise ModelingConfigError(f"Unknown audio profile: {args.profile}")
            if args.profile is not None and not args.features:
                raise ModelingConfigError("An audio profile requires at least one --features file")
            if args.profile is None and args.features:
                raise ModelingConfigError("--features requires an explicit --profile")
            feature_observations = tuple(
                observation
                for path in args.features
                for observation in read_feature_observations(path)
            )
            taste_result = train_taste_models(
                read_temporal_samples(args.samples),
                config=modeling_config.model,
                feature_observations=feature_observations,
                profile=(
                    modeling_config.profiles[args.profile] if args.profile is not None else None
                ),
                profile_name=args.profile,
            )
            write_taste_training_result(taste_result, args.output_dir)
        except (
            FeatureRecordError,
            ModelingConfigError,
            RepresentationError,
            TasteTrainingError,
            TemporalDatasetError,
            OSError,
        ) as exc:
            print(f"Taste model training failed: {exc}", file=sys.stderr)
            return 2
        trained = sum(variant.status == "trained" for variant in taste_result.report.variants)
        print(
            f"Trained and chronologically evaluated {trained} model variants; selected "
            f"{taste_result.report.selected_model_name} using validation data only."
        )
        selected_variant = next(
            variant
            for variant in taste_result.report.variants
            if variant.model_id == taste_result.report.selected_model_id
        )
        if selected_variant.test_metrics is not None:
            print(
                "Untouched test: "
                f"NDCG@{selected_variant.test_metrics.ranking_k}="
                f"{selected_variant.test_metrics.ndcg_at_k}, "
                f"average precision={selected_variant.test_metrics.average_precision}."
            )
        print(f"Wrote private model artifacts and predictions to {args.output_dir}")
        return 0
    if args.command == "build-taste-map":
        from myusic_engine.clustering import (
            TasteMapConfig,
            TasteMapError,
            build_taste_map,
            write_taste_map,
        )
        from myusic_engine.features import FeatureRecordError, read_feature_observations
        from myusic_engine.modeling import (
            ModelingConfigError,
            RepresentationError,
            load_modeling_config,
        )

        try:
            modeling_config = load_modeling_config(args.modeling_config)
            if args.profile not in modeling_config.profiles:
                raise ModelingConfigError(f"Unknown audio profile: {args.profile}")
            feature_observations = tuple(
                observation
                for path in args.features
                for observation in read_feature_observations(path)
            )
            taste_map_result = build_taste_map(
                feature_observations,
                profile=modeling_config.profiles[args.profile],
                profile_name=args.profile,
                config=TasteMapConfig(
                    representation=args.representation,
                    minimum_k=args.minimum_k,
                    maximum_k=args.maximum_k,
                ),
            )
            write_taste_map(taste_map_result, args.output_dir)
        except (
            FeatureRecordError,
            ModelingConfigError,
            RepresentationError,
            TasteMapError,
            OSError,
        ) as exc:
            print(f"Taste-map build failed: {exc}", file=sys.stderr)
            return 2
        taste_map_report = taste_map_result.report
        print(
            f"Clustered {taste_map_report.tracks_clustered} tracks with "
            f"{taste_map_report.selected_algorithm} into "
            f"{taste_map_report.selected_cluster_count} clusters "
            f"({taste_map_report.selected_noise_rate:.1%} noise)."
        )
        print(f"Wrote private taste-map artifacts to {args.output_dir}")
        return 0
    if args.command == "rank-candidates":
        from myusic_engine.clustering import TasteMapError, read_taste_map_assignments
        from myusic_engine.features import FeatureRecordError, read_feature_observations
        from myusic_engine.modeling import (
            ModelingConfigError,
            RepresentationError,
            TasteTrainingError,
            TemporalDatasetError,
            load_modeling_config,
            read_behavior_snapshots,
            read_taste_model,
        )
        from myusic_engine.ranking import (
            CandidateInputError,
            RecommendationError,
            load_recommendation_config,
            rank_candidates,
            read_candidates,
            write_recommendations,
        )

        try:
            if args.profile is None and args.features:
                raise ModelingConfigError("--features requires an explicit --profile")
            if args.profile is not None and not args.features:
                raise ModelingConfigError("An audio profile requires at least one --features file")
            selected_profile = None
            if args.profile is not None:
                modeling_config = load_modeling_config(args.modeling_config)
                if args.profile not in modeling_config.profiles:
                    raise ModelingConfigError(f"Unknown audio profile: {args.profile}")
                selected_profile = modeling_config.profiles[args.profile]
            seeds: dict[str, float] = {}
            for raw_seed in args.seed:
                if "=" not in raw_seed:
                    raise RecommendationError("Each --seed must use TRACK_ID=WEIGHT")
                track_id, raw_weight = raw_seed.rsplit("=", 1)
                if not track_id.strip() or track_id in seeds:
                    raise RecommendationError("Seed track IDs must be non-empty and unique")
                try:
                    weight = float(raw_weight)
                except ValueError as exc:
                    raise RecommendationError("Seed weights must be numeric") from exc
                if not weight > 0:
                    raise RecommendationError("Seed weights must be positive")
                seeds[track_id] = weight
            feature_observations = tuple(
                observation
                for path in args.features
                for observation in read_feature_observations(path)
            )
            recommendation_result = rank_candidates(
                read_candidates(args.candidates),
                feature_observations,
                profile=selected_profile,
                profile_name=args.profile,
                seed_weights=seeds,
                model=read_taste_model(args.model) if args.model is not None else None,
                behavior_snapshots=(
                    read_behavior_snapshots(args.behavior_snapshots)
                    if args.behavior_snapshots is not None
                    else ()
                ),
                cluster_assignments=(
                    read_taste_map_assignments(args.taste_map_assignments)
                    if args.taste_map_assignments is not None
                    else ()
                ),
                config=load_recommendation_config(args.recommendation_config),
                top_k=args.top_k,
            )
            write_recommendations(recommendation_result, args.output_dir)
        except (
            CandidateInputError,
            FeatureRecordError,
            ModelingConfigError,
            RecommendationError,
            RepresentationError,
            TasteTrainingError,
            TasteMapError,
            TemporalDatasetError,
            OSError,
        ) as exc:
            print(f"Candidate ranking failed: {exc}", file=sys.stderr)
            return 2
        print(
            f"Ranked {recommendation_result.report.ranked_count}/"
            f"{recommendation_result.report.candidates_seen} candidates; wrote "
            f"{recommendation_result.report.output_count} ordered results."
        )
        print(f"Recommendation run ID: {recommendation_result.report.run_id}")
        print(f"Wrote private recommendations to {args.output_dir}")
        return 0
    if args.command == "publish-spotify-playlist":
        import os

        from myusic_engine.spotify_output import (
            SpotifyPlaylistError,
            SpotifyWebApiClient,
            create_publication_plan,
            publish_playlist,
            read_spotify_uri_file,
            write_publication_plan,
        )

        plan_path = args.output_dir / "spotify_playlist_plan.json"
        receipt_path = args.output_dir / "spotify_playlist_receipt.json"
        try:
            plan = create_publication_plan(
                read_spotify_uri_file(args.uri_file),
                playlist_name=args.name,
                description=args.description,
            )
            write_publication_plan(plan, plan_path)
            if not args.execute:
                print(
                    f"Planned {plan.item_count} tracks for a private Spotify playlist; "
                    "no network request was made."
                )
                print(f"Review the deterministic publication plan at {plan_path}")
                return 0
            access_token = os.environ.get(args.access_token_env)
            if access_token is None:
                raise SpotifyPlaylistError(
                    f"Environment variable {args.access_token_env} is not set"
                )
            receipt = publish_playlist(
                plan,
                SpotifyWebApiClient(access_token, timeout_seconds=args.timeout_seconds),
                receipt_path,
            )
        except (OSError, SpotifyPlaylistError) as exc:
            print(f"Spotify playlist publication failed: {exc}", file=sys.stderr)
            return 2
        print(
            f"Published {receipt.confirmed_item_count} tracks to private playlist "
            f"{receipt.playlist_uri}."
        )
        print(f"Wrote the secret-free resumable receipt to {receipt_path}")
        return 0
    if args.command == "record-feedback":
        from datetime import datetime

        from myusic_engine.ranking import (
            FeedbackError,
            append_feedback,
            create_feedback,
        )

        try:
            recorded_at = datetime.fromisoformat(args.at) if args.at is not None else None
            feedback = create_feedback(
                args.recommendation_run_id,
                args.track_id,
                args.outcome,
                recorded_at=recorded_at,
            )
            append_feedback(args.feedback_log, feedback)
        except (FeedbackError, OSError, ValueError) as exc:
            print(f"Feedback recording failed: {exc}", file=sys.stderr)
            return 2
        print(f"Recorded {feedback.outcome} feedback event {feedback.feedback_id}.")
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
