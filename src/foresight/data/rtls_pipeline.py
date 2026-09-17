"""RTLS 원시 스트림(10 Hz, 파티션 Parquet) → 2.5 Hz 프레임 → ETH/UCY 형식 장면(``SceneSet``).

파이프라인
----------
1. **scan**      ``pl.scan_parquet(date=*/hour=*/*.parquet)`` — 메타데이터만 읽고 실행 계획을 만든다. 정리·리샘플·sink 는 시간 파티션 단위로 반복한다(메모리 상한 고정).
2. **clean**     ``quality`` < 임계 제거, ``(tag_id, ts_ms)`` 중복 제거(at-least-once 전달), 좌표를 평면 경계로 클립.
3. **resample**  0.4 s 빈 평균 (10 Hz → 2.5 Hz; ETH/UCY 프레임 레이트와 같다). ``zone_id`` 는 평균 위치에서 재계산.
                 → ``frames_2p5hz/part=<date>T<hour>/zone_id=K/*.parquet`` 로 sink (시간 파티션마다 스트리밍, 구역별 파티션). EDA 용 long 포맷.
4. **segments**  태그별로 연속 빈이 끊기면(결손 빈, 구역 이동, 분할 경계) 새 세그먼트. 선택적 평활(EMA / median3).
5. **windows**   구역마다 20 프레임 슬라이딩 윈도우(stride ``skip``). 20 프레임 모두 있는 에이전트만, ``min_agents``
                 이상인 장면만 — ``foresight.data.ethucy`` 와 같은 규칙.
6. **SceneSet**  시간 기준 train/val/test (기본 70/15/15). 윈도우는 시작 빈이 속한 분할에 들어가고, 분할 경계를
                 걸치는 윈도우는 세그먼트가 경계에서 끊기므로 자동으로 버려진다(누수 없음, 테스트로 확인).

메모리 설계 — 왜 이렇게 하나
----------------------------
* 1~3 단계는 **단일 lazy 쿼리를 스트리밍 엔진으로 sink** 한다. 86M 행을 한 번에 올리면 x/y/ts 만으로도
  1.4 GB, 중간 결과까지 수 GB 가 되지만, 스트리밍 엔진은 morsel 단위로 흘려보내므로 피크 RSS 가 총 행 수에
  비례하지 않는다. 유일하게 상태를 갖는 연산은 ``group_by(tag_id, bin)`` 인데 출력(2.5 Hz)이 입력의 1/4 이라
  해시 테이블도 그만큼 작다.
* 4~5 단계는 **구역 단위**로 numpy 에 올린다. 한 구역은 전체 프레임의 ~1/24 이고 열이 좁아(tag, bin, x, y, type)
  200 태그 × 12 h 도 구역당 수십 MB 다. 윈도우 인덱스(어느 행부터 20개)만 먼저 만들고, 장면 좌표 배열은 총 크기를
  안 뒤 **한 번만 할당**해 채운다 — ``np.concatenate`` 로 두 배 메모리를 쓰는 순간이 없다.
* 통계는 DuckDB 가 Parquet 을 직접 읽어 out-of-core 로 집계한다(``memory_limit`` 로 상한 고정, 초과분은 디스크 스필).
"""

from __future__ import annotations

import json
import math
import resource
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import duckdb
import numpy as np
import polars as pl
import psutil
import pyarrow as pa

from foresight.data.ethucy import SceneSet
from foresight.data.rtls_sim import (
    MANIFEST_NAME,
    PlantLayout,
    get_logger,
    layout_from_manifest,
    load_manifest,
)

log = get_logger(__name__)

FRAMES_DIR = "frames_2p5hz"
META_NAME = "pipeline_meta.json"
SPLITS = ("train", "val", "test")
Smoothing = Literal["none", "ema", "median3"]


@dataclass(frozen=True)
class PipelineConfig:
    """전처리 파라미터. 기본값은 ETH/UCY 장면 규칙(8+12 프레임, 2.5 Hz, 2명 이상)과 맞춰 두었다."""

    quality_min: int = 30
    bin_ms: int = 400  # 2.5 Hz
    obs_len: int = 8
    pred_len: int = 12
    skip_train: int = 4  # 학습 분할 stride — 인접 윈도우는 19/20 프레임이 겹쳐 중복이 크다
    skip_eval: int = 1  # val/test 는 ETH/UCY 평가 관례대로 stride 1
    min_agents: int = 2
    smoothing: Smoothing = "none"
    ema_alpha: float = 0.5
    split_fracs: tuple[float, float, float] = (0.7, 0.15, 0.15)
    min_bin_samples: int = 1  # 빈 안 원시 샘플 수가 이보다 적으면 결손 취급

    @property
    def seq_len(self) -> int:
        return self.obs_len + self.pred_len


@dataclass
class PipelineResult:
    out_dir: Path
    rows_raw: int
    rows_clean: int
    rows_frames: int
    scenes: dict[str, int]
    agents: dict[str, int]
    timings: dict[str, float]
    peak_rss_mb: float
    ru_maxrss_mb: float
    config: PipelineConfig
    split_bounds: tuple[int, int, int, int]
    frames_dir: Path
    rss_stage_mb: dict[str, float] | None = (
        None  # 단계별 피크 RSS — 어느 단계가 메모리를 결정하는지
    )
    stats: dict[str, Any] | None = None

    @property
    def seconds(self) -> float:
        return self.timings.get("total", 0.0)


class PeakRss:
    """블록 실행 중 피크 RSS(MB) 를 백그라운드 스레드로 샘플링한다.

    ``ru_maxrss`` 는 프로세스 수명 전체의 최대라 단계별 피크를 볼 수 없어 따로 잰다. Polars/DuckDB 스레드는
    같은 프로세스 안이므로 RSS 에 모두 포함된다.
    """

    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._proc = psutil.Process()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self._proc.memory_info().rss / 2**20)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> PeakRss:
        self.peak_mb = self._proc.memory_info().rss / 2**20
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, self._proc.memory_info().rss / 2**20)


def ru_maxrss_mb() -> float:
    """프로세스 수명 전체의 피크 RSS (Linux: KB 단위)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ---------------------------------------------------------------- 1~3 단계: lazy + streaming
def scan_raw(in_dir: Path | str) -> pl.LazyFrame:
    """파티션 Parquet 전체를 하나의 LazyFrame 으로. hive 열(date, hour)은 통계에만 쓰고 projection pushdown 으로 빠진다."""
    return pl.scan_parquet(
        str(Path(in_dir) / "date=*" / "hour=*" / "*.parquet"), hive_partitioning=True
    )


def count_rows(lf: pl.LazyFrame) -> int:
    """행 수 — Parquet 메타데이터만으로 계산되므로 데이터를 읽지 않는다."""
    return int(lf.select(pl.len()).collect(engine="streaming").item())


def clean(lf: pl.LazyFrame, cfg: PipelineConfig, layout: PlantLayout) -> pl.LazyFrame:
    """데이터 품질 규칙: 저품질 제거 → (tag_id, ts_ms) 중복 제거 → 평면 경계 클립."""
    return (
        lf.select("ts_ms", "tag_id", "agent_type", "x", "y", "quality")
        .filter(pl.col("quality") >= cfg.quality_min)
        # keep="any": 순서 유지가 필요 없으면 스트리밍 dedupe 가 훨씬 싸다 (중복 행은 값이 같다)
        .unique(subset=["tag_id", "ts_ms"], keep="any")
        .with_columns(
            x=pl.col("x").clip(0.0, layout.width_m).cast(pl.Float32),
            y=pl.col("y").clip(0.0, layout.height_m).cast(pl.Float32),
        )
    )


def zone_expr(layout: PlantLayout) -> pl.Expr:
    """``PlantLayout.zone_of`` 와 같은 식의 Polars 표현식 (평균 위치로 zone_id 재계산)."""
    cell = layout.zone_cell_m
    col = (pl.col("x").clip(0.0, layout.width_m - 1e-3) / cell).floor()
    row = (pl.col("y").clip(0.0, layout.height_m - 1e-3) / cell).floor()
    return (row * layout.n_cols + col).cast(pl.Int16)


def resample(lf: pl.LazyFrame, cfg: PipelineConfig, layout: PlantLayout) -> pl.LazyFrame:
    """태그별 ``bin_ms`` 빈 평균 위치 (10 Hz → 2.5 Hz). ``n`` = 빈 안 원시 샘플 수 (결손·품질 진단용)."""
    return (
        lf.with_columns(bin=pl.col("ts_ms") // cfg.bin_ms)
        .group_by("tag_id", "bin")
        .agg(
            x=pl.col("x").mean().cast(pl.Float32),
            y=pl.col("y").mean().cast(pl.Float32),
            agent_type=pl.col("agent_type").first(),
            n=pl.len().cast(pl.UInt8),
            quality=pl.col("quality").mean().round(0).cast(pl.UInt8),
        )
        .filter(pl.col("n") >= cfg.min_bin_samples)
        .with_columns(zone_id=zone_expr(layout))
        .select("tag_id", "bin", "x", "y", "zone_id", "agent_type", "n", "quality")
    )


def list_hour_dirs(in_dir: Path) -> list[Path]:
    """``date=*/hour=*`` 파티션 디렉터리를 시간 순으로."""
    return sorted(p for p in Path(in_dir).glob("date=*/hour=*") if p.is_dir())


def write_frames(in_dir: Path, frames_dir: Path, cfg: PipelineConfig, layout: PlantLayout) -> int:
    """원시 파티션 → 2.5 Hz 프레임을 **시간(hour) 파티션마다 따로** 정리·리샘플·sink 한다.

    처음에는 전체를 하나의 lazy 쿼리(``unique`` → ``group_by`` → 구역별 PartitionBy sink)로 흘렸는데, 85M 행에서
    스트리밍 엔진이 dedupe/group_by 상태를 10.9 GB 까지 키워 OOM 으로 죽었다(cgroup kill). 중복 키 ``(tag_id, ts_ms)``
    와 400 ms 빈은 시간 파티션 경계를 넘지 않으므로(3,600,000 % 400 == 0) 파티션마다 독립적으로 처리해도 결과는
    **정확히 같다**. 메모리는 이제 "한 시간 분량"(200 태그 기준 7.2M 원시 행)에만 비례하고 전체 행 수와 무관하다.
    반환값: 처리한 파티션 수.
    """
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    hours = list_hour_dirs(in_dir)
    if not hours:
        raise FileNotFoundError(f"no date=*/hour=* partitions under {in_dir}")
    for hd in hours:
        part = f"{hd.parent.name.split('=', 1)[1]}T{hd.name.split('=', 1)[1]}"  # 예: 2026-09-17T08
        lf = pl.scan_parquet(str(hd / "*.parquet"), hive_partitioning=False)
        resample(clean(lf, cfg, layout), cfg, layout).sink_parquet(
            pl.PartitionBy(str(frames_dir / f"part={part}"), key="zone_id"),
            compression="zstd",
            engine="streaming",
            mkdir=True,
        )
    return len(hours)


def scan_frames(frames_dir: Path | str) -> pl.LazyFrame:
    """모든 시간 파티션의 구역 파티션 (zone_id 는 파일 안에도 있으므로 hive 파싱 없이 읽는다)."""
    return pl.scan_parquet(
        str(Path(frames_dir) / "part=*" / "zone_id=*" / "*.parquet"), hive_partitioning=False
    )


def list_zones(frames_dir: Path) -> list[int]:
    return sorted(
        {int(p.name.split("=")[1]) for p in frames_dir.glob("part=*/zone_id=*") if p.is_dir()}
    )


# ---------------------------------------------------------------- 4~5 단계: 세그먼트·윈도우
def split_bounds(
    frames: pl.LazyFrame, fracs: tuple[float, float, float]
) -> tuple[int, int, int, int]:
    """시간(빈) 기준 분할 경계 ``(first_bin, val_start, test_start, last_bin)``.

    빈 < val_start → train, < test_start → val, 나머지 → test. 경계는 시간에만 의존하므로 같은 태그의 미래가
    학습에 섞이는 일이 없다.
    """
    lo, hi = (
        frames.select(pl.col("bin").min().alias("lo"), pl.col("bin").max().alias("hi"))
        .collect(engine="streaming")
        .row(0)
    )
    lo, hi = int(lo), int(hi)
    n = hi - lo + 1
    val_start = lo + math.floor(n * fracs[0])
    test_start = lo + math.floor(n * (fracs[0] + fracs[1]))
    return lo, val_start, test_start, hi


def load_zone(
    frames_dir: Path, zone: int, cfg: PipelineConfig, bounds: tuple[int, int, int, int]
) -> pl.DataFrame:
    """한 구역의 프레임을 (tag_id, bin) 순으로 정렬하고 split / 세그먼트 / 평활 열을 붙여 메모리에 올린다."""
    _lo, val_start, test_start, _hi = bounds
    q = (
        pl.scan_parquet(
            str(frames_dir / "part=*" / f"zone_id={zone}" / "*.parquet"), hive_partitioning=False
        )
        .sort("tag_id", "bin")
        .with_columns(
            split=pl.when(pl.col("bin") < val_start)
            .then(0)
            .when(pl.col("bin") < test_start)
            .then(1)
            .otherwise(2)
            .cast(pl.Int8)
        )
        .with_columns(
            brk=(
                (pl.col("tag_id") != pl.col("tag_id").shift(1))
                | (pl.col("bin") != pl.col("bin").shift(1) + 1)
                | (pl.col("split") != pl.col("split").shift(1))
            ).fill_null(True)
        )
        .with_columns(seg=pl.col("brk").cast(pl.Int64).cum_sum())
    )
    if cfg.smoothing == "ema":
        q = q.with_columns(
            x=pl.col("x").ewm_mean(alpha=cfg.ema_alpha).over("seg").cast(pl.Float32),
            y=pl.col("y").ewm_mean(alpha=cfg.ema_alpha).over("seg").cast(pl.Float32),
        )
    elif cfg.smoothing == "median3":
        q = q.with_columns(
            x=pl.col("x")
            .rolling_median(3, center=True, min_samples=1)
            .over("seg")
            .cast(pl.Float32),
            y=pl.col("y")
            .rolling_median(3, center=True, min_samples=1)
            .over("seg")
            .cast(pl.Float32),
        )
    elif cfg.smoothing != "none":
        raise ValueError(f"unknown smoothing {cfg.smoothing!r}")
    return q.select("tag_id", "bin", "x", "y", "agent_type", "split", "brk").collect(
        engine="streaming"
    )


@dataclass
class ZoneWindows:
    """한 구역의 윈도우 인덱스. 좌표는 아직 복사하지 않는다 — 총 크기를 안 뒤 한 번에 채우기 위해."""

    zone: int
    row0: np.ndarray  # (P,) 각 (장면, 에이전트) 쌍의 첫 프레임 행 번호 (구역 프레임 배열 기준)
    sizes: np.ndarray  # (S,) 장면별 에이전트 수
    scene_bin: np.ndarray  # (S,) 장면 시작 빈
    scene_split: np.ndarray  # (S,) 0 train / 1 val / 2 test
    xy: np.ndarray  # (R, 2) float32 구역 프레임 좌표
    agent_type: np.ndarray  # (R,) int8

    def pair_split(self) -> np.ndarray:
        return np.repeat(self.scene_split, self.sizes)


def window_index(
    bins: np.ndarray, brk: np.ndarray, split: np.ndarray, tag: np.ndarray, cfg: PipelineConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """세그먼트 → (장면, 에이전트) 쌍 인덱스. 행 루프 없이 ``repeat``/``cumsum`` 으로 전개한다.

    세그먼트 s 가 빈 ``b0`` 에서 길이 ``L`` 이면 시작 빈 ``w ∈ [b0, b0+L-seq_len]``, ``w % skip == 0`` 인
    윈도우마다 이 에이전트가 20 프레임 모두 존재한다. 같은 ``w`` 의 쌍을 모으면 장면이고, 에이전트 순서는
    ETH/UCY 처럼 id 오름차순이다.

    Returns:
        ``(row0, sizes, scene_bin, scene_split)`` — ``min_agents`` 미만 장면은 제거된 뒤.
    """
    seq_len = cfg.seq_len
    seg_row0 = np.flatnonzero(brk)
    seg_len = np.diff(np.append(seg_row0, len(bins)))
    seg_bin0 = bins[seg_row0]
    seg_split = split[seg_row0]
    skip = np.where(seg_split == 0, cfg.skip_train, cfg.skip_eval).astype(np.int64)
    first_w = -(-seg_bin0 // skip) * skip  # skip 의 배수로 올림
    last_w = seg_bin0 + seg_len - seq_len
    count = np.where(last_w >= first_w, (last_w - first_w) // skip + 1, 0)
    total = int(count.sum())
    empty = np.zeros(0, dtype=np.int64)
    if total == 0:
        return empty, empty, empty, empty.astype(np.int8)
    rep = np.repeat(np.arange(len(count)), count)
    offs = np.arange(total) - np.repeat(np.cumsum(count) - count, count)
    w = first_w[rep] + skip[rep] * offs
    row0 = seg_row0[rep] + (w - seg_bin0[rep])
    pair_split = seg_split[rep]
    order = np.lexsort((tag[row0], w))
    w, row0, pair_split = w[order], row0[order], pair_split[order]
    starts = np.flatnonzero(np.concatenate([[True], w[1:] != w[:-1]]))
    sizes = np.diff(np.append(starts, total))
    keep = sizes >= cfg.min_agents
    keep_rows = np.repeat(keep, sizes)
    return row0[keep_rows], sizes[keep], w[starts[keep]], pair_split[starts[keep]]


def zone_windows(
    frames_dir: Path, zone: int, cfg: PipelineConfig, bounds: tuple[int, int, int, int]
) -> ZoneWindows:
    df = load_zone(frames_dir, zone, cfg, bounds)
    bins = df["bin"].to_numpy()
    row0, sizes, scene_bin, scene_split = window_index(
        bins, df["brk"].to_numpy(), df["split"].to_numpy(), df["tag_id"].to_numpy(), cfg
    )
    xy = np.stack([df["x"].to_numpy(), df["y"].to_numpy()], axis=1)
    return ZoneWindows(zone, row0, sizes, scene_bin, scene_split, xy, df["agent_type"].to_numpy())


def build_sceneset(
    zws: list[ZoneWindows], split_idx: int, cfg: PipelineConfig, gather_chunk: int = 500_000
) -> SceneSet:
    """구역별 윈도우 인덱스 → 한 분할의 ``SceneSet``. 좌표 배열은 총 쌍 수를 센 뒤 한 번만 할당한다."""
    seq_len = cfg.seq_len
    masks = [zw.scene_split == split_idx for zw in zws]
    total_pairs = sum(int(zw.sizes[m].sum()) for zw, m in zip(zws, masks))
    pos = np.empty((total_pairs, seq_len, 2), dtype=np.float64)
    atype = np.empty(total_pairs, dtype=np.int8)
    sizes_all: list[np.ndarray] = []
    meta: list[tuple[str, float]] = []
    cursor = 0
    offsets = np.arange(seq_len)
    for zw, m in zip(zws, masks):
        if not m.any():
            continue
        row0 = zw.row0[np.repeat(m, zw.sizes)]
        for s in range(0, len(row0), gather_chunk):  # 임시 인덱스 배열 크기를 제한
            r = row0[s : s + gather_chunk]
            idx = r[:, None] + offsets[None, :]
            k = len(r)
            # ETH/UCY 와 같이 소수 4자리 (UWB 잡음 σ 0.15 m 에서 그 이하 자릿수는 의미가 없다)
            pos[cursor : cursor + k] = np.around(zw.xy[idx].astype(np.float64), 4)
            atype[cursor : cursor + k] = zw.agent_type[r]
            cursor += k
        sizes_all.append(zw.sizes[m])
        meta.extend((f"zone={zw.zone:02d}", float(b)) for b in zw.scene_bin[m])
    sizes = np.concatenate(sizes_all) if sizes_all else np.zeros(0, dtype=np.int64)
    ends = np.cumsum(sizes)
    scene_index = np.stack([ends - sizes, ends], axis=1).astype(np.int64)
    return SceneSet(
        pos=pos,
        scene_index=scene_index,
        meta=meta,
        obs_len=cfg.obs_len,
        pred_len=cfg.pred_len,
        agent_type=atype,
    )


# ---------------------------------------------------------------- 전체 실행
def run_pipeline(
    in_dir: Path | str,
    out_dir: Path | str,
    cfg: PipelineConfig | None = None,
    stats_path: Path | str | None = None,
    layout: PlantLayout | None = None,
) -> PipelineResult:
    """원시 파티션 Parquet → ``out_dir/{train,val,test}.npz`` + ``out_dir/frames_2p5hz/`` + ``pipeline_meta.json``.

    ``stats_path`` 를 주면 DuckDB 집계(``compute_stats``)도 실행해 JSON 으로 쓴다.
    """
    cfg = cfg or PipelineConfig()
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(in_dir) if (in_dir / MANIFEST_NAME).exists() else None
    layout = layout or layout_from_manifest(manifest)
    timings: dict[str, float] = {}
    rss_stage: dict[str, float] = {}
    t_all = time.perf_counter()
    with PeakRss() as rss:
        # 1~3: 단일 lazy 쿼리 → 스트리밍 sink
        t = time.perf_counter()
        raw = scan_raw(in_dir)
        rows_raw = count_rows(raw)
        frames_dir = out_dir / FRAMES_DIR
        with PeakRss() as r1:
            n_parts = write_frames(in_dir, frames_dir, cfg, layout)
        timings["resample_sink"] = time.perf_counter() - t
        log.info("resample: %d hour partitions sunk", n_parts)
        rss_stage["resample_sink"] = r1.peak_mb
        frames = scan_frames(frames_dir)
        agg = frames.select(
            pl.len().alias("rows"), pl.col("n").cast(pl.Int64).sum().alias("clean")
        ).collect(engine="streaming")
        rows_frames, rows_clean = int(agg["rows"][0]), int(agg["clean"][0])
        log.info(
            "resample: %d raw → %d clean → %d frames in %.1fs",
            rows_raw,
            rows_clean,
            rows_frames,
            timings["resample_sink"],
        )

        # 4~5: 구역 단위 윈도우 인덱스
        t = time.perf_counter()
        bounds = split_bounds(frames, cfg.split_fracs)
        with PeakRss() as r2:
            zws = [zone_windows(frames_dir, z, cfg, bounds) for z in list_zones(frames_dir)]
        timings["window_index"] = time.perf_counter() - t
        rss_stage["window_index"] = r2.peak_mb

        # 6: 분할별 SceneSet
        t = time.perf_counter()
        scenes: dict[str, int] = {}
        agents: dict[str, int] = {}
        with PeakRss() as r3:
            for i, name in enumerate(SPLITS):
                ss = build_sceneset(zws, i, cfg)
                ss.save(out_dir / f"{name}.npz")
                scenes[name], agents[name] = len(ss), len(ss.pos)
                log.info("%s: %d scenes, %d agent-windows", name, scenes[name], agents[name])
                del ss
        timings["sceneset"] = time.perf_counter() - t
        rss_stage["sceneset"] = r3.peak_mb
        del zws
    timings["total"] = time.perf_counter() - t_all
    result = PipelineResult(
        out_dir=out_dir,
        rows_raw=rows_raw,
        rows_clean=rows_clean,
        rows_frames=rows_frames,
        scenes=scenes,
        agents=agents,
        timings=timings,
        peak_rss_mb=rss.peak_mb,
        ru_maxrss_mb=ru_maxrss_mb(),
        config=cfg,
        split_bounds=bounds,
        frames_dir=frames_dir,
        rss_stage_mb={k: round(v, 1) for k, v in rss_stage.items()},
    )
    if stats_path is not None:
        t = time.perf_counter()
        result.stats = compute_stats(in_dir, out_dir, stats_path, cfg, manifest, layout)
        result.timings["stats"] = time.perf_counter() - t
    meta = {k: v for k, v in asdict(result).items() if k not in ("stats",)}
    meta["config"] = asdict(cfg)
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in meta.items()}
    (out_dir / META_NAME).write_text(
        json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    log.info("pipeline done in %.1fs, peak RSS %.0f MB", timings["total"], rss.peak_mb)
    return result


# ---------------------------------------------------------------- DuckDB 통계
def compute_stats(
    in_dir: Path | str,
    out_dir: Path | str,
    stats_path: Path | str,
    cfg: PipelineConfig | None = None,
    manifest: dict[str, Any] | None = None,
    layout: PlantLayout | None = None,
    memory_limit: str = "2GB",
    threads: int = 4,
) -> dict[str, Any]:
    """DuckDB 로 원시·프레임 Parquet 을 직접 읽어 집계 통계를 만든다.

    Polars 가 아닌 DuckDB 를 쓰는 이유: (1) 시간당/구역당 행 수, 분위수, self-join(측정 near-miss) 같은 SQL 은
    DuckDB 가 out-of-core 해시 집계·조인을 ``memory_limit`` 안에서 자동으로 스필하며 처리하고, (2) 같은 질의를
    분석가가 노트북/CLI 에서 그대로 재사용할 수 있다. 결과는 작은 JSON 이라 파이프라인 산출물과 함께 커밋한다.
    """
    cfg = cfg or PipelineConfig()
    in_dir, out_dir, stats_path = Path(in_dir), Path(out_dir), Path(stats_path)
    if manifest is None and (in_dir / MANIFEST_NAME).exists():
        manifest = load_manifest(in_dir)
    layout = layout or layout_from_manifest(manifest)
    d_safe = float(manifest["near_miss"]["d_safe"]) if manifest else 1.0
    raw = f"read_parquet('{in_dir}/date=*/hour=*/*.parquet', hive_partitioning=true)"
    frames = f"read_parquet('{out_dir / FRAMES_DIR}/part=*/zone_id=*/*.parquet', hive_partitioning=false)"
    out: dict[str, Any] = {
        "d_safe": d_safe,
        "quality_min": cfg.quality_min,
        "bin_ms": cfg.bin_ms,
        "queries": {},
    }
    tmp = tempfile.mkdtemp(prefix="duckdb_spill_")
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads TO {threads}")
    con.execute(f"SET temp_directory='{tmp}'")

    def q(name: str, sql: str) -> list[tuple]:
        t = time.perf_counter()
        rows = con.execute(sql).fetchall()
        out["queries"][name] = round(time.perf_counter() - t, 3)
        return rows

    try:
        out["raw"] = {
            "rows": q("raw_rows", f"SELECT count(*) FROM {raw}")[0][0],
            "rows_per_hour": [
                {"date": str(d), "hour": int(h), "rows": int(n)}
                for d, h, n in q(
                    "rows_per_hour",
                    f"SELECT date, hour, count(*) FROM {raw} GROUP BY 1, 2 ORDER BY 1, 2",
                )
            ],
            "rows_per_zone": {
                int(z): int(n)
                for z, n in q(
                    "rows_per_zone", f"SELECT zone_id, count(*) FROM {raw} GROUP BY 1 ORDER BY 1"
                )
            },
            "quality": dict(
                zip(
                    ("mean", "p05", "p50", "frac_below_min"),
                    [
                        float(v)
                        for v in q(
                            "quality",
                            f"SELECT avg(quality), quantile_cont(quality, 0.05), quantile_cont(quality, 0.5), "
                            f"avg(CASE WHEN quality < {cfg.quality_min} THEN 1 ELSE 0 END) FROM {raw}",
                        )[0]
                    ],
                )
            ),
            "duplicate_keys": int(
                q(
                    "duplicates",
                    f"SELECT count(*) FROM (SELECT tag_id, ts_ms FROM {raw} GROUP BY ALL HAVING count(*) > 1)",
                )[0][0]
            ),
            "out_of_bounds_rows": int(
                q(
                    "out_of_bounds",
                    f"SELECT count(*) FROM {raw} WHERE x < 0 OR x > {layout.width_m} OR y < 0 OR y > {layout.height_m}",
                )[0][0]
            ),
        }
        apf = q(
            "agents_per_frame",
            f"SELECT avg(n), quantile_cont(n, 0.5), quantile_cont(n, 0.9), quantile_cont(n, 0.99), max(n), count(*) "
            f"FROM (SELECT zone_id, bin, count(*) AS n FROM {frames} GROUP BY 1, 2)",
        )[0]
        out["frames"] = {
            "rows": int(q("frames_rows", f"SELECT count(*) FROM {frames}")[0][0]),
            "rows_per_zone": {
                int(z): int(n)
                for z, n in q(
                    "frames_per_zone",
                    f"SELECT zone_id, count(*) FROM {frames} GROUP BY 1 ORDER BY 1",
                )
            },
            "agents_per_zone_frame": {
                "mean": float(apf[0]),
                "p50": float(apf[1]),
                "p90": float(apf[2]),
                "p99": float(apf[3]),
                "max": int(apf[4]),
                "zone_frames": int(apf[5]),
            },
            "samples_per_bin": {
                int(n): int(c)
                for n, c in q(
                    "samples_per_bin", f"SELECT n, count(*) FROM {frames} GROUP BY 1 ORDER BY 1"
                )
            },
        }
        # 2.5 Hz 프레임에서 측정된 near-miss (같은 구역·빈의 작업자-차량 self-join). 참값과 정의가 다르므로
        # (잡음 + 0.4 s 평균) 두 수를 나란히 적는다 — "측정으로 얼마나 보이는가" 가 곧 라벨 품질이다.
        nm = q(
            "near_miss_measured",
            f"""
            WITH w AS (SELECT * FROM {frames} WHERE agent_type = 0),
                 v AS (SELECT * FROM {frames} WHERE agent_type = 1)
            SELECT count(*), count(DISTINCT (w.tag_id, v.tag_id))
            FROM w JOIN v ON w.zone_id = v.zone_id AND w.bin = v.bin
            WHERE sqrt((w.x - v.x) * (w.x - v.x) + (w.y - v.y) * (w.y - v.y)) < {d_safe}
            """,
        )[0]
        out["near_miss"] = {
            "measured_frame_pairs": int(nm[0]),
            "measured_distinct_pairs": int(nm[1]),
        }
        if manifest:
            events = manifest["near_miss"]["events"]
            out["near_miss"]["truth_count"] = int(manifest["near_miss"]["count"])
            out["near_miss"]["truth_per_vehicle_hour"] = float(
                manifest["near_miss"]["per_vehicle_hour"]
            )
            if events:
                # 매니페스트 이벤트를 Arrow 테이블로 등록 — DuckDB 는 Python 객체를 복사 없이 스캔한다
                cols = (
                    "ts_ms",
                    "worker_tag",
                    "vehicle_tag",
                    "min_dist",
                    "worker_state",
                    "vehicle_state",
                )
                con.register(
                    "events", pa.Table.from_pylist([{k: e[k] for k in cols} for e in events])
                )
                out["near_miss"]["truth_by_state"] = {
                    f"{ws}/{vs}": int(n)
                    for ws, vs, n in q(
                        "truth_by_state",
                        "SELECT worker_state, vehicle_state, count(*) FROM events GROUP BY 1, 2 ORDER BY 3 DESC",
                    )
                }
                md = q(
                    "truth_min_dist",
                    "SELECT quantile_cont(min_dist, 0.05), quantile_cont(min_dist, 0.5), quantile_cont(min_dist, 0.95) FROM events",
                )[0]
                out["near_miss"]["truth_min_dist"] = {
                    "p05": float(md[0]),
                    "p50": float(md[1]),
                    "p95": float(md[2]),
                }
                out["near_miss"]["truth_per_hour"] = [
                    {"hour": int(h), "events": int(n)}
                    for h, n in q(
                        "truth_per_hour",
                        "SELECT hour(to_timestamp(ts_ms / 1000)), count(*) FROM events GROUP BY 1 ORDER BY 1",
                    )
                ]
    finally:
        con.close()
        shutil.rmtree(tmp, ignore_errors=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> PipelineResult:
    """typer 없는 진입점 — CLI 담당자가 ``foresight prepare-rtls`` 에서 감싼다."""
    import argparse

    p = argparse.ArgumentParser(description="RTLS Parquet → 2.5 Hz 프레임 → SceneSet")
    p.add_argument("--in-dir", default="data/rtls/raw")
    p.add_argument("--out-dir", default="data/processed/rtls")
    p.add_argument(
        "--stats", default=None, help="DuckDB 통계 JSON 경로 (예: results/data_pipeline/stats.json)"
    )
    p.add_argument("--quality-min", type=int, default=PipelineConfig.quality_min)
    p.add_argument("--skip-train", type=int, default=PipelineConfig.skip_train)
    p.add_argument("--skip-eval", type=int, default=PipelineConfig.skip_eval)
    p.add_argument("--smoothing", choices=("none", "ema", "median3"), default="none")
    a = p.parse_args(argv)
    cfg = PipelineConfig(
        quality_min=a.quality_min,
        skip_train=a.skip_train,
        skip_eval=a.skip_eval,
        smoothing=a.smoothing,
    )
    return run_pipeline(a.in_dir, a.out_dir, cfg, a.stats)


if __name__ == "__main__":
    main()
