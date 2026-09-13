"""Download attributed Creative Commons music examples into ignored local storage.

Run from the repository root: python scripts/prepare_demo.py
Audio is pinned to librosa's published SHA-256 registry. Keep the attribution files.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import soundfile as sf

from myusic_engine.io import atomic_write_bytes, atomic_write_text
from myusic_engine.ranking.candidates import CandidateTrack

EXAMPLES = (
    (
        "nutcracker",
        "Kevin_MacLeod_-_P_I_Tchaikovsky_Dance_of_the_Sugar_Plum_Fairy",
        "Dance of the Sugar Plum Fairy",
        "Kevin MacLeod",
        "CC-BY-3.0",
        "https://creativecommons.org/licenses/by/3.0/",
        "b5c1a3e26310e6618d3c124f458654cd235650fcb9db7d711302644566600484",
        "059acb340170385d2bfa4c7ab7c2a06b1d8f8af3e0f11cb4f46ff4049e950915",
    ),
    (
        "vibeace",
        "Kevin_MacLeod_-_Vibe_Ace",
        "Vibe Ace",
        "Kevin MacLeod",
        "CC-BY-3.0",
        "https://creativecommons.org/licenses/by/3.0/",
        "6c23aed3dd5aa57f2b1652ecab68d15d9b82ad257f54e639eb2880ca09bc118a",
        "6c71e0525cb0452ea74c6d6f5fde6fa1e221223db7b2aa35b9914b98367ee7b9",
    ),
    (
        "fishin",
        "Karissa_Hobbs_-_Lets_Go_Fishin",
        "Let's Go Fishin'",
        "Karissa Hobbs",
        "CC-BY-3.0",
        "https://creativecommons.org/licenses/by/3.0/",
        "27b3667c396c1831511aa3c415fcf582b6e8be560cafb844c5b67b76b72c1cb3",
        "199bf3408b98916cd9d28a22b2b43c1935ab70072b46fa05c0b9f40e7882802e",
    ),
    (
        "pistachio",
        "442789__lena-orsa__happy-music-pistachio-ice-cream-ragtime",
        "Pistachio Ice Cream Ragtime",
        "The Piano Lady (Lena Orsa)",
        "CC-BY-NC-3.0",
        "https://creativecommons.org/licenses/by-nc/3.0/",
        "9617c9be55c128177b13c20fbc52178ed482e3545094517efb30a7db2798991e",
        "99f9c44368918572ac154ae4a2fec4020d1d4bb418ae7bf241864916d68f9d04",
    ),
)


def prepare(destination: Path) -> None:
    manifest, candidates, attributions = [], [], []
    for key, stem, title, artist, license_name, license_url, audio_hash, text_hash in EXAMPLES:
        for extension, digest in (("ogg", audio_hash), ("txt", text_hash)):
            path = destination / f"{stem}.{extension}"
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
                continue
            with urllib.request.urlopen(
                f"https://librosa.org/data/audio/{stem}.{extension}", timeout=60
            ) as response:
                content = response.read(16 * 1024 * 1024 + 1)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Public sample checksum mismatch: {key}.{extension}")
            atomic_write_bytes(path, content)
        with sf.SoundFile(destination / f"{stem}.ogg") as recording:
            # Short, explicitly documented excerpts keep CPU validation practical.
            samples = recording.read(frames=35 * recording.samplerate, dtype="float32")
            sf.write(destination / f"{key}.wav", samples, recording.samplerate, subtype="PCM_16")
        track_id = f"open-demo:{key}"
        manifest.append(
            {"track_id": track_id, "audio_path": f"{key}.wav", "rights_basis": "creative_commons"}
        )
        candidates.append(CandidateTrack(track_id, track_name=title, artist_name=artist).to_dict())
        attributions.append(
            {
                "track_id": track_id,
                "title": title,
                "artist": artist,
                "license": license_name,
                "license_url": license_url,
                "source": f"https://librosa.org/data/audio/{stem}.ogg",
                "original_sha256": audio_hash,
                "modification": (
                    "First 35 seconds, decoded to PCM16 WAV for local noncommercial validation."
                ),
                "original_attribution": f"{stem}.txt",
            }
        )
    for name, rows in (("audio_manifest.jsonl", manifest), ("candidates.jsonl", candidates)):
        atomic_write_text(destination / name, "".join(json.dumps(row) + "\n" for row in rows))
    atomic_write_text(destination / "attribution.json", json.dumps(attributions, indent=2) + "\n")
    print(f"Prepared {len(manifest)} attributed music excerpts at {destination}")


if __name__ == "__main__":
    prepare(Path("data/private/open-demo"))
