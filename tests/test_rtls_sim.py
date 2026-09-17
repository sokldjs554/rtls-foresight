"""합성 RTLS 생성기 테스트 — 결정성, 스키마/파티션, near-miss 라벨이 참값과 맞는지, 측정 모델 통계."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from foresight.data.rtls_sim import (
    MANIFEST_NAME,
    PROFILES,
    RAW_SCHEMA,
    TRUTH_DIR,
    PlantLayout,
    main,
    resolve_config,
    simulate,
)

# 테스트용 소형 공장: 밀도를 높여(40 × 30 m, 12 태그) 4분 안에 near-miss 가 확실히 생기게 하고,
# 청크를 2분으로 잡아 여러 파일이 만들어지는 경로도 지나간다.
SIM_KW = dict(
    hours=4 / 60,
    tags=12,
    seed=7,
    plant_width_m=40.0,
    plant_height_m=30.0,
    zone_cell_m=10.0,
    aisle_spacing_m=10.0,
    distracted_per_hour=60.0,
    chunk_minutes=2,
)


@pytest.fixture(scope="module")
def sim_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("rtls_raw")
    simulate(out, **SIM_KW)
    return out


def _raw(sim_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(str(sim_dir / "date=*" / "hour=*" / "*.parquet"))


def test_determinism_same_seed_identical_bytes(tmp_path: Path) -> None:
    a = simulate(tmp_path / "a", hours=1 / 60, tags=6, seed=3, chunk_minutes=1)
    b = simulate(tmp_path / "b", hours=1 / 60, tags=6, seed=3, chunk_minutes=1)
    assert a.rows == b.rows and a.n_events == b.n_events
    fa = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*.parquet"))
    fb = sorted(p.relative_to(tmp_path / "b") for p in (tmp_path / "b").rglob("*.parquet"))
    assert fa == fb and len(fa) >= 1
    for rel in fa:
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes(), rel
    # 다른 시드는 달라야 한다 (결정성 테스트가 "항상 같은 상수" 를 통과시키지 않도록)
    c = simulate(tmp_path / "c", hours=1 / 60, tags=6, seed=4, chunk_minutes=1)
    assert not _raw(tmp_path / "c").equals(_raw(tmp_path / "a"))
    assert c.rows > 0


def test_schema_partitions_and_manifest(sim_dir: Path) -> None:
    files = sorted(sim_dir.rglob("part-*.parquet"))
    files = [f for f in files if TRUTH_DIR not in f.parts]
    assert len(files) == 2  # 4분 / 2분 청크
    for f in files:
        assert f.parent.name.startswith("hour=") and f.parent.parent.name.startswith("date=")
        assert pq.read_schema(f).equals(RAW_SCHEMA), f
        t = pq.read_table(f).to_pandas()
        # 파일 안은 (ts, tag) 정렬, 시간 파티션과 일치
        assert (np.diff(t["ts_ms"].to_numpy()) >= 0).all()
        hour = int(f.parent.name.split("=")[1])
        assert ((t["ts_ms"] // 3_600_000) % 24 == hour).all()
    man = json.loads((sim_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    raw = _raw(sim_dir)
    assert man["total_rows"] == len(raw) == sum(man["rows_per_hour"].values())
    # 드롭아웃 ~1% + 중복 ~0.2% 이므로 기대 행 수의 ±3% 안
    assert abs(len(raw) / man["expected_rows"] - 1) < 0.03
    assert {r["tag_id"] for r in man["roster"]} == set(raw["tag_id"].unique().to_list())
    assert man["n_workers"] + man["n_vehicles"] == SIM_KW["tags"]
    assert raw["quality"].min() >= 0 and raw["quality"].max() <= 100
    lay = PlantLayout(40.0, 30.0, 10.0)
    assert raw["zone_id"].max() < lay.n_zones
    # 중복 패킷이 실제로 들어 있어야 파이프라인의 dedupe 규칙이 의미가 있다
    assert raw.select(pl.struct("tag_id", "ts_ms").is_duplicated().sum()).item() > 0


def test_near_miss_events_match_noise_free_truth(sim_dir: Path) -> None:
    man = json.loads((sim_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    events = man["near_miss"]["events"]
    d_safe = man["near_miss"]["d_safe"]
    assert len(events) >= 1, "테스트 설정(고밀도·고 distracted)에서는 near-miss 가 있어야 한다"
    truth = pl.read_parquet(str(sim_dir / TRUTH_DIR / "date=*" / "hour=*" / "*.parquet"))
    roster = {r["tag_id"]: r["agent_type"] for r in man["roster"]}

    def dist_at(ts: int, a: int, b: int) -> float:
        rows = truth.filter(pl.col("ts_ms") == ts).filter(pl.col("tag_id").is_in([a, b]))
        assert len(rows) == 2
        xy = rows.sort("tag_id").select("x", "y").to_numpy().astype(np.float64)
        return float(np.hypot(*(xy[0] - xy[1])))

    for e in events:
        assert roster[e["worker_tag"]] == 0 and roster[e["vehicle_tag"]] == 1
        assert e["min_dist"] < d_safe
        # 최소 거리 시점: 참값 거리 == min_dist (float32 저장 오차 안)
        assert dist_at(e["ts_min_ms"], e["worker_tag"], e["vehicle_tag"]) == pytest.approx(
            e["min_dist"], abs=2e-3
        )
        # 진입 시점: 거리 < d_safe, 그 직전 스텝: 거리 >= d_safe (이벤트는 "임계 아래로 들어간 순간")
        assert dist_at(e["ts_ms"], e["worker_tag"], e["vehicle_tag"]) < d_safe
        if e["ts_ms"] - 100 >= truth["ts_ms"].min():
            assert dist_at(e["ts_ms"] - 100, e["worker_tag"], e["vehicle_tag"]) >= d_safe
        assert e["worker_state"] in ("aware", "idle", "distracted")
    assert man["near_miss"]["count"] == len(events)


def test_measurement_noise_and_dropout(sim_dir: Path) -> None:
    raw = _raw(sim_dir)
    truth = pl.read_parquet(str(sim_dir / TRUTH_DIR / "date=*" / "hour=*" / "*.parquet"))
    j = raw.unique(["tag_id", "ts_ms"]).join(truth, on=["tag_id", "ts_ms"], suffix="_true")
    err = (j["x"] - j["x_true"]).to_numpy()
    # σ 0.15 m 가우시안 + 드문 이상치 → 강건 표준편차(MAD·1.4826) 는 0.15 근처, 이상치 꼬리(>1 m)는 소수
    robust_sd = 1.4826 * np.median(np.abs(err - np.median(err)))
    assert 0.11 < robust_sd < 0.20
    assert (np.abs(err) > 1.0).mean() < 0.01
    dropout = 1 - len(j) / len(truth)
    assert 0.003 < dropout < 0.05


def test_profiles_and_config_resolution() -> None:
    assert set(PROFILES) == {"smoke", "small", "full"}
    cfg = resolve_config(profile="full")
    assert cfg.hours == 12.0 and cfg.tags == 200
    assert round(cfg.hours * 3600 * cfg.hz * cfg.tags) == 86_400_000
    cfg = resolve_config(hours=0.5, profile="small", seed=9)
    assert cfg.hours == 0.5 and cfg.tags == 60 and cfg.seed == 9  # 명시 인자가 profile 보다 우선
    with pytest.raises(ValueError):
        resolve_config(profile="huge")
    with pytest.raises(ValueError):
        resolve_config(chunk_minutes=7)


def test_main_entrypoint(tmp_path: Path) -> None:
    r = main(
        [
            "--out-dir",
            str(tmp_path / "o"),
            "--hours",
            "0.01",
            "--tags",
            "4",
            "--seed",
            "1",
            "--no-truth",
        ]
    )
    assert r.rows > 0 and (tmp_path / "o" / MANIFEST_NAME).exists()
    assert not (tmp_path / "o" / TRUTH_DIR).exists()


def test_zone_of_row_major() -> None:
    lay = PlantLayout(120.0, 80.0, 20.0)
    assert lay.n_cols == 6 and lay.n_rows == 4 and lay.n_zones == 24
    z = lay.zone_of(
        np.array([0.0, 25.0, 119.9, -5.0, 200.0]), np.array([0.0, 25.0, 79.9, -5.0, 200.0])
    )
    assert z.dtype == np.int16
    assert z.tolist() == [0, 7, 23, 0, 23]  # 평면 밖은 가장 가까운 셀로 클립
