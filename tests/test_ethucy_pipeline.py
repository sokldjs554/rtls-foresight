"""ETH/UCY 윈도우 규칙과 SceneSet 직렬화."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from foresight.data.dataset import SceneGraphDataset
from foresight.data.ethucy import SceneSet, build_scenes


def _write_raw(path: Path, rows: list[tuple[float, float, float, float]]) -> None:
    pl.DataFrame(rows, schema=["frame", "ped", "x", "y"], orient="row").write_csv(
        path, separator="\t", include_header=False
    )


def test_windowing_keeps_only_agents_present_in_all_frames(tmp_path: Path) -> None:
    rows = []
    for f in range(25):
        rows.append((10.0 * f, 1.0, float(f), 0.0))
        rows.append((10.0 * f, 2.0, float(f), 1.0))
        if 3 <= f < 15:  # 보행자 3 은 일부 프레임에만
            rows.append((10.0 * f, 3.0, float(f), 2.0))
    _write_raw(tmp_path / "a.txt", rows)
    ss = build_scenes(tmp_path, obs_len=8, pred_len=12, skip=1, min_ped=1)
    assert len(ss) == 25 - 20 + 1
    assert set(ss.num_agents.tolist()) == {2}
    ds = SceneGraphDataset(ss)
    b = ds[0]
    assert (
        b.v_obs.shape == (1, 2, 8, 2)
        and b.a_obs.shape == (1, 8, 2, 2)
        and b.v_pred.shape == (1, 12, 2, 2)
    )


def test_min_ped_drops_single_agent_scenes(tmp_path: Path) -> None:
    rows = [(10.0 * f, 1.0, float(f), 0.0) for f in range(20)]
    _write_raw(tmp_path / "a.txt", rows)
    assert len(build_scenes(tmp_path, min_ped=1)) == 0
    assert len(build_scenes(tmp_path, min_ped=0)) == 1


def test_sceneset_roundtrip_with_agent_type(tmp_path: Path) -> None:
    pos = np.random.default_rng(0).normal(size=(5, 20, 2))
    ss = SceneSet(
        pos=pos,
        scene_index=np.array([[0, 2], [2, 5]]),
        meta=[("a", 0.0), ("a", 10.0)],
        obs_len=8,
        pred_len=12,
        agent_type=np.array([0, 1, 0, 0, 1], dtype=np.int8),
    )
    ss.save(tmp_path / "s.npz")
    back = SceneSet.load(tmp_path / "s.npz")
    np.testing.assert_array_equal(back.pos, pos)
    assert back.types(1).tolist() == [0, 0, 1]
    assert len(back.to_long_frame()) == 5 * 20


def test_bucket_iteration_covers_every_scene_once(eth_test_npz: Path) -> None:
    ds = SceneGraphDataset.from_npz(eth_test_npz)
    seen = []
    for b in ds.iter_buckets(4, np.random.default_rng(0)):
        assert len({int(n) for n in [b.v_obs.shape[-1]]}) == 1
        seen += b.indices
    assert sorted(seen) == list(range(len(ds)))
