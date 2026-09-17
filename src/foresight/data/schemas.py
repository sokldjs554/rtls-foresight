"""RTLS 데이터 계약 — Pandera(Polars 백엔드) 스키마.

두 지점에서 스키마를 고정한다.

* ``RawRtlsFrame``  — 10 Hz 원시 스트림 한 행 (``docs/CLI_CONTRACT.md`` 의 Parquet 스키마). 좌표는 측정 잡음·이상치
  때문에 평면 밖으로 조금 나갈 수 있으므로 여유(``RAW_MARGIN_M``)를 둔다. 파이프라인이 클립하기 *전* 의 계약이다.
* ``ResampledFrame`` — 2.5 Hz 프레임 (``rtls_pipeline.resample`` 출력). 클립 뒤이므로 좌표는 평면 안, ``(tag_id, bin)``
  은 유일, 빈당 원시 샘플 수 ``n`` 은 최대 4 (10 Hz × 0.4 s, 중복 제거 뒤).

dtype 은 ``strict=True`` 로 정확히 일치해야 한다 — Int32 가 Int64 로 바뀌면 downstream 의 Arrow/ONNX 입력이 조용히
달라지므로 "넓은 타입도 허용" 하지 않는다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import pandera.polars as pa
import polars as pl

from foresight.data.rtls_sim import PlantLayout

_DEFAULT = PlantLayout()
PLANT_W = _DEFAULT.width_m
PLANT_H = _DEFAULT.height_m
N_ZONES = _DEFAULT.n_zones
RAW_MARGIN_M = 3.0  # 4σ(0.6 m) + 이상치 최대 2 m


class RawRtlsFrame(pa.DataFrameModel):
    """10 Hz 원시 RTLS 행. hive 파티션 열(date, hour)은 경로에서 오므로 계약에 포함하지 않는다."""

    ts_ms: pl.Int64 = pa.Field(ge=0)
    tag_id: pl.Int32 = pa.Field(ge=0)
    agent_type: pl.Int8 = pa.Field(isin=[0, 1])
    zone_id: pl.Int16 = pa.Field(ge=0, lt=N_ZONES)
    x: pl.Float32 = pa.Field(ge=-RAW_MARGIN_M, le=PLANT_W + RAW_MARGIN_M)
    y: pl.Float32 = pa.Field(ge=-RAW_MARGIN_M, le=PLANT_H + RAW_MARGIN_M)
    quality: pl.UInt8 = pa.Field(le=100)

    class Config:
        strict = True
        coerce = False


class ResampledFrame(pa.DataFrameModel):
    """2.5 Hz 프레임 (클립·중복 제거·리샘플 뒤)."""

    tag_id: pl.Int32 = pa.Field(ge=0)
    bin: pl.Int64 = pa.Field(ge=0)
    x: pl.Float32 = pa.Field(ge=0.0, le=PLANT_W)
    y: pl.Float32 = pa.Field(ge=0.0, le=PLANT_H)
    zone_id: pl.Int16 = pa.Field(ge=0, lt=N_ZONES)
    agent_type: pl.Int8 = pa.Field(isin=[0, 1])
    n: pl.UInt8 = pa.Field(ge=1, le=4)
    quality: pl.UInt8 = pa.Field(le=100)

    class Config:
        strict = True
        coerce = False
        unique: ClassVar[list[str]] = ["tag_id", "bin"]


def sample_rows(lf: pl.LazyFrame, n: int, keys: Sequence[str], seed: int = 0) -> pl.DataFrame:
    """LazyFrame 에서 약 ``n`` 행을 결정적으로 뽑는다 (키 해시 기반, 스트리밍 실행).

    ``head(n)`` 은 첫 파티션(첫 시간대)만 보게 되므로 편향된다. 키의 해시를 stride 로 나눠 전 구간에서 고르게
    뽑되 총 행 수는 Parquet 메타데이터로 미리 세어 stride 를 정한다.
    """
    total = int(lf.select(pl.len()).collect(engine="streaming").item())
    stride = max(1, total // max(n, 1))
    return (
        lf.filter(pl.struct(list(keys)).hash(seed) % stride == 0)
        .head(n)
        .collect(engine="streaming")
    )


def validate_sample(
    lf: pl.LazyFrame,
    model: type[pa.DataFrameModel] = RawRtlsFrame,
    n: int = 100_000,
    keys: Sequence[str] | None = None,
    seed: int = 0,
) -> pl.DataFrame:
    """스키마를 **표본**에 대해 검증하고 검증된 표본을 돌려준다 (실패 시 ``pandera.errors.SchemaErrors``).

    왜 전체가 아니라 표본인가: 86M 행 전체를 pandera 로 검증하면 메모리에 다 올려야 하고(수 GB) 검증 자체가
    파이프라인보다 오래 걸린다. 스키마 위반의 두 종류 중 **구조적 위반**(열 누락, dtype 변경, 범위 밖 상수)은
    어느 행을 봐도 드러나므로 표본으로 충분하고, **산발적 위반**(품질 낮은 행, 경계 밖 좌표)은 파이프라인의
    필터 규칙이 처리하며 그 비율은 DuckDB 통계(``compute_stats``)로 전수 집계한다. 즉 표본 검증 = 계약 확인,
    전수 통계 = 데이터 품질 모니터링으로 역할을 나눈다.
    """
    cols = list(model.to_schema().columns)
    keys = list(keys) if keys is not None else [c for c in ("tag_id", "ts_ms", "bin") if c in cols]
    sample = sample_rows(lf.select(cols), n, keys, seed)
    return model.validate(sample, lazy=True)
