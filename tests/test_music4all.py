import bz2
import csv
import hashlib
import math
from dataclasses import replace

import pytest

from myusic_engine.features import music4all
from myusic_engine.ranking.candidates import CandidateTrack

URI = "spotify:track:" + "A" * 22
TRACK = CandidateTrack(URI, URI, "Synthetic Song", "Synthetic Artist", "Synthetic Album")


def source(tmp_path, monkeypatch, *, duplicate_id=False, vector=None):
    metadata = [{"id": "synthetic", "spotify_id": "A" * 22}]
    if duplicate_id:
        metadata.append({"id": "another", "spotify_id": "A" * 22})
    with (tmp_path / "id_metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "spotify_id"], delimiter="\t")
        writer.writeheader()
        writer.writerows(metadata)
    with (tmp_path / "id_information.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["id", "artist", "song", "album_name"])
        writer.writerow(["synthetic", "Synthetic Artist", "Synthetic Song", "Synthetic Album"])
    with bz2.open(tmp_path / "id_maest.tsv.bz2", "wt", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["id", *map(str, range(music4all.DIMENSIONS))])
        writer.writerow(["synthetic", *(vector if vector is not None else [1] * 768)])
    pins = {}
    for path in tmp_path.iterdir():
        content = path.read_bytes()
        pins[path.name] = (
            "https://example.invalid",
            len(content),
            hashlib.sha256(content).hexdigest(),
        )
    monkeypatch.setattr(music4all, "FILES", pins)
    return tmp_path


def test_exact_id_and_metadata_join_preserves_audio_provenance(tmp_path, monkeypatch):
    directory = source(tmp_path, monkeypatch)
    result = music4all.import_maest_features(
        [replace(TRACK, track_name="  SYNTHETIC song ")], directory
    )
    (observation,) = result.observations
    assert observation.track_id == URI
    assert observation.feature_name == music4all.FEATURE_NAME
    assert observation.source_version == music4all.SOURCE_VERSION
    assert len(observation.value) == 768
    assert math.sqrt(sum(value**2 for value in observation.value)) == pytest.approx(1)
    assert observation.coverage_seconds == 30
    assert result.matches[0]["music4all_id"] == "synthetic"


@pytest.mark.parametrize(
    "changed", [{"album_name": "Live"}, {"artist_name": None}, {"track_name": "Other"}]
)
def test_identity_conflicts_never_receive_an_embedding(tmp_path, monkeypatch, changed):
    result = music4all.import_maest_features(
        [replace(TRACK, **changed)], source(tmp_path, monkeypatch)
    )
    assert not result.observations
    assert result.counts["metadata_unconfirmed"] == 1


def test_duplicate_spotify_mapping_abstains(tmp_path, monkeypatch):
    result = music4all.import_maest_features(
        [TRACK], source(tmp_path, monkeypatch, duplicate_id=True)
    )
    assert not result.observations
    assert result.counts["ambiguous_spotify_id"] == 1


@pytest.mark.parametrize("vector", [[0] * 768, [float("nan")] * 768, [1, 2]])
def test_invalid_embeddings_are_rejected(tmp_path, monkeypatch, vector):
    with pytest.raises(ValueError, match="MAEST embedding"):
        music4all.import_maest_features([TRACK], source(tmp_path, monkeypatch, vector=vector))


def test_modified_source_is_rejected_before_import(tmp_path, monkeypatch):
    directory = source(tmp_path, monkeypatch)
    path = directory / "id_information.csv"
    path.write_bytes(path.read_bytes().replace(b"Synthetic Song", b"Different Song"))
    with pytest.raises(ValueError, match="checksum"):
        music4all.import_maest_features([TRACK], directory)
