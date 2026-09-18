"""공식 로더와의 장면 집합 동일성 — eth/test 골든(장면 수 70, 장면별 float32 위치 해시)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from foresight.data.ethucy import build_scenes

ROOT = Path(__file__).resolve().parents[1]


def test_eth_test_scenes_match_official_loader() -> None:
    raw = ROOT / "data" / "raw" / "ethucy" / "eth" / "test"
    if not raw.exists():
        pytest.skip("run `foresight download` first")
    golden = json.loads((ROOT / "tests" / "fixtures" / "golden_eth_test_scenes.json").read_text())
    ss = build_scenes(raw, 8, 12, 1, 1)
    assert len(ss) == golden["n_scenes"] and len(ss.pos) == golden["n_agents"]
    hashes = sorted(
        hashlib.md5(np.ascontiguousarray(ss.scene(i).astype(np.float32)).tobytes()).hexdigest()
        for i in range(len(ss))
    )
    assert hashes == golden["scene_md5_float32"]
