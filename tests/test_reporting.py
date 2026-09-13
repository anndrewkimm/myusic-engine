import json

import pytest

from myusic_engine.cli import main
from myusic_engine.ranking import CandidateTrack, rank_candidates, write_recommendations
from myusic_engine.reporting import build_explorer_report


def test_browser_report_escapes_imported_html_and_works_through_cli(tmp_path):
    attack = '</script><script>alert("private")</script>'
    result = rank_candidates((CandidateTrack("synthetic", track_name=attack),), ())
    write_recommendations(result, tmp_path)
    output = tmp_path / "explorer.html"
    assert main(["build-report", str(tmp_path), "--output", str(output)]) == 0
    html = output.read_text(encoding="utf-8")
    assert attack not in html
    assert "\\u003c/script\\u003e" in html
    assert "connect-src 'none'" in html
    assert "script-src 'sha256-" in html
    payload = html.split('<script type="application/json" id="data">')[1].split("</script>")[0]
    assert json.loads(payload)["tracks"][0]["track_name"] == attack
    assert "__PAYLOAD__" not in html and "__CSP__" not in html


def test_report_rejects_rows_from_a_different_run(tmp_path):
    result = rank_candidates((CandidateTrack("synthetic"),), ())
    write_recommendations(result, tmp_path)
    path = tmp_path / "recommendations.jsonl"
    row = json.loads(path.read_text())
    row["run_id"] = "wrong"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="run"):
        build_explorer_report(tmp_path, tmp_path / "report.html")
