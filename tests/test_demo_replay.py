"""데모 재생 데이터 생성: 창 선택 규칙, 프레임·예측 구조, JS 파일 쓰기."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from foresight.data.ethucy import SceneSet
from foresight.demo.replay import (
    ReplayWindow,
    build_replay,
    select_window,
    write_js,
    zone_layout,
)
from foresight.eval.evaluate import load_model
from foresight.utils import project_root

CKPT = project_root() / "results" / "checkpoints" / "rtls-scratch-fast" / "best.pth"


def _scene(
    worker: np.ndarray, vehicle: np.ndarray, zone: int, start: float
) -> tuple[np.ndarray, tuple[str, float]]:
    return np.stack([worker, vehicle]), (f"zone={zone}", start)


def test_select_window_prefers_zone_with_positives() -> None:
    t = np.arange(20, dtype=np.float64)
    walk = np.stack([t * 0.5, np.zeros(20)], -1)
    head_on = np.stack([10.0 - t * 0.5, np.zeros(20)], -1)  # t=10 에 만남 → 양성
    far = np.stack([t * 0.5, np.full(20, 30.0)], -1)
    scenes_pos, meta = [], []
    for start in (100, 101, 102):  # 구역 3: 양성 3 장면
        p, m = _scene(walk, head_on, 3, start)
        scenes_pos.append(p)
        meta.append(m)
    for start in (100, 101, 102, 103):  # 구역 5: 음성만 (더 많은 장면)
        p, m = _scene(walk, far, 5, start)
        scenes_pos.append(p)
        meta.append(m)
    pos = np.concatenate(scenes_pos)
    idx = np.array([[2 * i, 2 * i + 2] for i in range(len(scenes_pos))])
    ss = SceneSet(
        pos=pos,
        scene_index=idx,
        meta=meta,
        obs_len=8,
        pred_len=12,
        agent_type=np.tile(np.array([0, 1], dtype=np.int8), len(scenes_pos)),
    )
    w = select_window(ss, d_safe=1.0, window_bins=150)
    assert w.zone == 3 and w.n_positive == 3
    assert w.bin_lo == 100 and w.bin_hi == 100 + 7 + 150 + 12
    w5 = select_window(ss, d_safe=1.0, window_bins=150, zone=5)  # 양성이 없으면 첫 창
    assert w5.zone == 5 and w5.n_positive == 0


def test_build_replay_and_write_js(tmp_path: Path) -> None:
    if not CKPT.exists():
        pytest.skip("RTLS 체크포인트가 없다")
    import polars as pl

    model = load_model(CKPT)
    bins = list(range(1000, 1030))
    rows = []
    for b in bins:
        k = b - 1000
        rows.append((7, b, 1.0 + 0.4 * k, 5.0, 0))  # 작업자: x 방향 이동
        if b >= 1005:  # 차량은 5 프레임 뒤에 나타난다 (이력 규칙 확인)
            rows.append((9, b, 12.0 - 0.6 * (k - 5), 5.2, 1))
    df = pl.DataFrame(rows, schema=["tag_id", "bin", "x", "y", "agent_type"], orient="row")
    win = ReplayWindow(zone=2, bin_lo=1000, bin_hi=1030, n_positive=0)
    rep = build_replay(df, model, win, k=5, seed=1, check_every=3, layout=zone_layout(2))
    assert rep["meta"]["n_frames"] == 30 and len(rep["frames"]) == 30 and len(rep["pred"]) == 30
    assert [a["id"] for a in rep["agents"]] == ["W7", "V9"]
    assert rep["pred"][6] is None and rep["pred"][7]["ids"] == [
        0
    ]  # 8 프레임 이력이 생기는 첫 프레임
    assert rep["pred"][12]["ids"] == [0, 1]  # 차량은 1005 부터 8 프레임 → 1012
    assert np.array(rep["pred"][12]["params"]).shape == (2, 12, 5)
    assert rep["check"] and {"f", "mean", "risk_det", "risk"} <= set(rep["check"][0])
    assert rep["meta"]["layout"] == {
        "x0": 40.0,
        "y0": 0.0,
        "size": 20.0,
        "aisle_m": 10.0,
        "n_cols": 6,
        "col": 2,
        "row": 0,
    }
    out = tmp_path / "replay.js"
    n = write_js(out, "FORESIGHT_REPLAY", rep)
    text = out.read_text(encoding="utf-8")
    assert n == len(text.encode("utf-8")) and text.startswith("window.FORESIGHT_REPLAY = ")
    back = json.loads(text[len("window.FORESIGHT_REPLAY = ") :].rstrip().rstrip(";"))
    assert back["meta"]["zone"] == 2 and len(back["frames"]) == 30
