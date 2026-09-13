"""Reproduce the permitted-audio-to-recommendation demonstration locally.

python scripts/run_demo.py --download     # first run: fetch verified public inputs
python scripts/run_demo.py                # subsequent runs: completely offline
"""

import argparse
from pathlib import Path

from prepare_demo import prepare

from myusic_engine.cli import main
from myusic_engine.embeddings.clap import download_clap_model


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if not Path("configs/modeling.yaml").is_file():
        parser.error("Run this script from the myusic-engine repository root")
    if args.download:
        download_clap_model("artifacts/models/clap", progress=print)
        prepare(Path("data/private/open-demo"))
    commands = [
        [
            "analyze-audio",
            "data/private/open-demo/audio_manifest.jsonl",
            "--embedding-backend",
            "clap",
            "--output",
            "data/processed/demo/features.jsonl",
            "--cache-dir",
            "data/interim/demo/checkpoints",
            "--window-output-dir",
            "data/interim/demo/clap-windows",
        ],
        [
            "build-taste-map",
            "--features",
            "data/processed/demo/features.jsonl",
            "--profile",
            "local_clap",
            "--output-dir",
            "data/processed/demo/taste-map",
        ],
        [
            "rank-candidates",
            "data/private/open-demo/candidates.jsonl",
            "--features",
            "data/processed/demo/features.jsonl",
            "--profile",
            "local_clap",
            "--text-query",
            "An instrumental jazz recording with piano, vibraphone and bass.",
            "--taste-map-assignments",
            "data/processed/demo/taste-map/taste_map_assignments.jsonl",
            "--output-dir",
            "data/processed/demo/recommendations",
        ],
        [
            "build-report",
            "data/processed/demo/recommendations",
            "--output",
            "data/processed/demo/explorer.html",
            "--taste-map-assignments",
            "data/processed/demo/taste-map/taste_map_assignments.jsonl",
        ],
    ]
    for command in commands:
        if status := main(command):
            raise SystemExit(status)


if __name__ == "__main__":
    run()
