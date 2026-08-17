import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_json_contracts_are_parseable_and_versioned() -> None:
    contract_paths = sorted((PROJECT_ROOT / "data_contracts").glob("*.schema.json"))

    assert contract_paths
    for contract_path in contract_paths:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        assert contract["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert contract["$id"].startswith("https://myusic-engine.local/schemas/")


def test_synthetic_fixture_uses_reserved_non_personal_values() -> None:
    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "spotify_history_synthetic.json"
    records = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert len(records) == 3
    assert {record["username"] for record in records} == {"synthetic_user"}
    assert {record["ip_addr"].split(".")[0] for record in records} == {
        "192",
        "198",
        "203",
    }
