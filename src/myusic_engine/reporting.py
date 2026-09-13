"""Self-contained, private browser reports without a server or third-party requests."""

from __future__ import annotations

import base64
import hashlib
import json
from importlib.resources import files
from pathlib import Path

from myusic_engine.clustering import read_taste_map_assignments
from myusic_engine.io import atomic_write_text
from myusic_engine.privacy import assert_privacy_safe


def build_explorer_report(
    recommendation_dir: str | Path,
    output: str | Path,
    *,
    taste_map_path: str | Path | None = None,
) -> Path:
    """Export one run and an optional matching taste map to a portable local HTML report."""
    if Path(output).suffix.casefold() not in {".html", ".htm"}:
        raise ValueError("Explorer output must be an .html or .htm file")
    source = Path(recommendation_dir)
    report = json.loads((source / "recommendation_run.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("Unsupported recommendation report")
    rows = []
    with (source / "recommendations.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("run_id") != report.get("run_id"):
                raise ValueError("Recommendation rows do not belong to the report run")
            assert_privacy_safe(row)
            rows.append(row)
    assignments = read_taste_map_assignments(taste_map_path) if taste_map_path else ()
    for assignment in assignments:
        if (
            assignment.profile_name != report.get("profile_name")
            or assignment.profile_version != report.get("profile_version")
            or assignment.model_id != report.get("taste_map_model_id")
        ):
            raise ValueError("Taste map does not belong to the recommendation run")
    assert_privacy_safe(report)
    payload = json.dumps(
        {"report": report, "tracks": rows, "map": [row.to_dict() for row in assignments]},
        ensure_ascii=False,
        allow_nan=False,
    )
    # Escape HTML parser delimiters, including malicious </script> in imported track names.
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    payload = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = files("myusic_engine").joinpath("assets/explorer.html").read_text(encoding="utf-8")
    script = template.split('<script id="app">', 1)[1].split("</script>", 1)[0]
    style = template.split("<style>", 1)[1].split("</style>", 1)[0]

    def digest(value: str) -> str:
        return base64.b64encode(hashlib.sha256(value.encode()).digest()).decode()

    # Script hashes allow this exact viewer but block injected scripts and all network requests.
    policy = (
        "default-src 'none'; "
        f"script-src 'sha256-{digest(script)}'; style-src 'sha256-{digest(style)}'; "
        "img-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'"
    )
    html = template.replace("__CSP__", policy).replace("__PAYLOAD__", payload)
    return atomic_write_text(output, html)
