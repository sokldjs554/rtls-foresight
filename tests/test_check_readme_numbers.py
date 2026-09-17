from __future__ import annotations

import json
from pathlib import Path

import tools.check_readme_numbers as crn


def test_marker_sync_and_check(tmp_path: Path, monkeypatch) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "demo.json").write_text(
        json.dumps({"a": {"b": 0.4567, "n": 12345, "list": [1, 2.5]}}), encoding="utf-8"
    )
    doc = tmp_path / "README.md"
    doc.write_text(
        "x <!-- num:demo.a.b -->0.00<!-- /num --> y <!-- num:demo.a.n:,d -->0<!-- /num --> z <!-- num:demo.a.list.1:.1f -->9<!-- /num -->",
        encoding="utf-8",
    )
    monkeypatch.setattr(crn, "RESULTS", results)
    monkeypatch.setattr(crn, "TARGETS", [doc])
    crn._cache.clear()
    assert crn.process(check=True) == 1  # stale
    assert crn.process(check=False) == 0
    assert (
        doc.read_text(encoding="utf-8")
        == "x <!-- num:demo.a.b -->0.46<!-- /num --> y <!-- num:demo.a.n:,d -->12,345<!-- /num --> z <!-- num:demo.a.list.1:.1f -->2.5<!-- /num -->"
    )
    assert crn.process(check=True) == 0


def test_missing_key_fails_check(tmp_path: Path, monkeypatch) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "demo.json").write_text("{}", encoding="utf-8")
    doc = tmp_path / "README.md"
    doc.write_text("<!-- num:demo.nope -->1<!-- /num -->", encoding="utf-8")
    monkeypatch.setattr(crn, "RESULTS", results)
    monkeypatch.setattr(crn, "TARGETS", [doc])
    crn._cache.clear()
    assert crn.process(check=True) == 1
