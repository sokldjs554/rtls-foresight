"""시간 파티션 단위 처리 == 전체 단일 쿼리 처리 (정확성 근거: 중복 키와 400 ms 빈은 시간 경계를 넘지 않는다)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from foresight.data.rtls_pipeline import (
    PipelineConfig,
    clean,
    list_hour_dirs,
    resample,
    scan_frames,
    write_frames,
)
from foresight.data.rtls_sim import PlantLayout


def _write_partitioned(root: Path, df: pl.DataFrame) -> None:
    """ts_ms 로 date=/hour= 파티션에 나눠 쓴다 (생성기와 같은 레이아웃)."""
    df = df.with_columns(
        date=pl.from_epoch(pl.col("ts_ms"), time_unit="ms").dt.strftime("%Y-%m-%d"),
        hour=pl.from_epoch(pl.col("ts_ms"), time_unit="ms").dt.strftime("%H"),
    )
    for (d, h), part in df.group_by(["date", "hour"]):
        out = root / f"date={d}" / f"hour={h}"
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(part.drop("date", "hour").to_pandas()), out / "part-00.parquet"
        )


def test_per_hour_sink_equals_global_query(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    layout = PlantLayout()
    # 두 시간 경계(정확히 3,600,000 ms)를 걸치는 3 시간 분량, 태그 5개, 10 Hz, 중복 패킷과 저품질 행 섞기
    t0 = 1_700_000_000_000 - (1_700_000_000_000 % 3_600_000) + 1_800_000  # 시간의 한가운데에서 시작
    n = 3 * 3600 * 10
    ts = t0 + np.repeat(np.arange(n) * 100, 5)
    tag = np.tile(np.arange(5, dtype=np.int32), n)
    x = rng.uniform(0, layout.width_m, size=ts.size).astype(np.float32)
    y = rng.uniform(0, layout.height_m, size=ts.size).astype(np.float32)
    q = rng.integers(0, 101, size=ts.size).astype(np.uint8)
    df = pl.DataFrame(
        {
            "ts_ms": ts,
            "tag_id": tag,
            "agent_type": np.zeros(ts.size, dtype=np.int8),
            "zone_id": np.zeros(ts.size, dtype=np.int16),
            "x": x,
            "y": y,
            "quality": q,
        }
    )
    dup = df.sample(n=2000, seed=1)  # 중복 키 주입
    df = pl.concat([df, dup]).sort("ts_ms", "tag_id")
    raw = tmp_path / "raw"
    _write_partitioned(raw, df)
    assert len(list_hour_dirs(raw)) >= 3

    cfg = PipelineConfig()
    frames_dir = tmp_path / "frames"
    write_frames(raw, frames_dir, cfg, layout)
    per_hour = scan_frames(frames_dir).collect().sort("tag_id", "bin")
    global_q = (
        resample(
            clean(pl.scan_parquet(str(raw / "date=*" / "hour=*" / "*.parquet")), cfg, layout),
            cfg,
            layout,
        )
        .collect()
        .sort("tag_id", "bin")
    )
    assert per_hour.height == global_q.height
    for col in ("tag_id", "bin", "n", "zone_id"):
        assert per_hour[col].to_list() == global_q[col].to_list()
    np.testing.assert_allclose(per_hour["x"].to_numpy(), global_q["x"].to_numpy(), atol=1e-5)
    np.testing.assert_allclose(per_hour["y"].to_numpy(), global_q["y"].to_numpy(), atol=1e-5)
