"""RTLS 파이프라인 테스트 — 품질 규칙, 리샘플 산술, 윈도우 규칙, 시간 분할 누수, SceneSet 왕복, 스키마."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandera.polars as pa
import polars as pl
import pytest

from foresight.data.ethucy import SceneSet
from foresight.data.rtls_pipeline import (
    FRAMES_DIR,
    META_NAME,
    SPLITS,
    PipelineConfig,
    PipelineResult,
    clean,
    list_zones,
    load_zone,
    resample,
    run_pipeline,
    scan_frames,
    scan_raw,
    window_index,
    zone_expr,
)
from foresight.data.rtls_sim import PlantLayout, load_manifest, simulate
from foresight.data.schemas import RawRtlsFrame, ResampledFrame, validate_sample

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
LAYOUT = PlantLayout(40.0, 30.0, 10.0)


@pytest.fixture(scope="module")
def sim_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("rtls_raw")
    simulate(out, **SIM_KW)
    return out


@pytest.fixture(scope="module")
def pipe(sim_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> PipelineResult:
    out = tmp_path_factory.mktemp("rtls_processed")
    return run_pipeline(sim_dir, out, PipelineConfig(), stats_path=out / "stats.json")


def test_clean_rules(sim_dir: Path, pipe: PipelineResult) -> None:
    cfg = PipelineConfig()
    raw = scan_raw(sim_dir).collect()
    cl = clean(scan_raw(sim_dir), cfg, LAYOUT).collect(engine="streaming")
    assert cl["quality"].min() >= cfg.quality_min
    assert cl.select(pl.struct("tag_id", "ts_ms").is_duplicated().any()).item() is False
    assert cl["x"].min() >= 0 and cl["x"].max() <= LAYOUT.width_m
    assert cl["y"].min() >= 0 and cl["y"].max() <= LAYOUT.height_m
    # 독립 계산: 저품질 제거 후 유일 키 수 == clean 행 수, 그리고 파이프라인이 보고한 수와 같다
    expected = raw.filter(pl.col("quality") >= cfg.quality_min).select("tag_id", "ts_ms").n_unique()
    assert len(cl) == expected == pipe.rows_clean
    assert pipe.rows_raw == len(raw)


def test_resample_bin_arithmetic(sim_dir: Path, pipe: PipelineResult) -> None:
    cfg = PipelineConfig()
    cl = clean(scan_raw(sim_dir), cfg, LAYOUT)
    fr = resample(cl, cfg, LAYOUT).collect(engine="streaming")
    assert fr.schema["n"] == pl.UInt8 and fr["n"].max() <= 4  # 10 Hz × 0.4 s, 중복 제거 뒤
    assert fr.select(pl.struct("tag_id", "bin").is_duplicated().any()).item() is False
    assert len(fr) == pipe.rows_frames
    # 임의의 (tag, bin) 하나를 원시 행으로 직접 재계산
    row = fr.sort("tag_id", "bin").row(len(fr) // 2, named=True)
    lo, hi = row["bin"] * cfg.bin_ms, (row["bin"] + 1) * cfg.bin_ms
    src = cl.filter(
        (pl.col("tag_id") == row["tag_id"]) & (pl.col("ts_ms") >= lo) & (pl.col("ts_ms") < hi)
    ).collect()
    assert len(src) == row["n"]
    assert row["x"] == pytest.approx(src["x"].mean(), abs=1e-5)
    assert row["y"] == pytest.approx(src["y"].mean(), abs=1e-5)
    # zone_id 는 평균 위치에서 재계산 — numpy 구현과 Polars 표현식이 같아야 한다
    z_np = LAYOUT.zone_of(fr["x"].to_numpy(), fr["y"].to_numpy())
    assert (fr.select(zone_expr(LAYOUT)).to_series().to_numpy() == z_np).all()
    assert (fr["zone_id"].to_numpy() == z_np).all()
    # 연속 빈 = 0.4 s 간격 (2.5 Hz)
    one = fr.filter(pl.col("tag_id") == row["tag_id"]).sort("bin")
    assert (np.diff(one["bin"].to_numpy()) >= 1).all()


def test_window_index_unit() -> None:
    cfg = PipelineConfig(
        obs_len=2, pred_len=3, skip_train=1, skip_eval=1, min_agents=2
    )  # seq_len 5
    # 태그 A: 빈 0..9 연속(10), 태그 B: 빈 0..6 연속 + 8..9 (7 에 결손 → 두 세그먼트)
    bins = np.array(list(range(10)) + list(range(7)) + [8, 9])
    tag = np.array([1] * 10 + [2] * 9)
    split = np.zeros(len(bins), dtype=np.int8)
    brk = np.zeros(len(bins), dtype=bool)
    brk[[0, 10, 17]] = True
    row0, sizes, scene_bin, scene_split = window_index(bins, brk, split, tag, cfg)
    # A 는 w=0..5 (6개), B 세그먼트1 은 w=0..2 (3개), 세그먼트2(길이 2) 는 0개 → 2명 장면 = w 0,1,2
    assert scene_bin.tolist() == [0, 1, 2] and sizes.tolist() == [2, 2, 2]
    assert row0.tolist() == [0, 10, 1, 11, 2, 12]  # (w, tag) 순 정렬
    assert scene_split.tolist() == [0, 0, 0]
    # stride 2 → w ∈ {0, 2}
    cfg2 = PipelineConfig(obs_len=2, pred_len=3, skip_train=2, skip_eval=1, min_agents=2)
    _, _, scene_bin2, _ = window_index(bins, brk, split, tag, cfg2)
    assert scene_bin2.tolist() == [0, 2]


def test_window_index_stride_alignment() -> None:
    cfg = PipelineConfig(obs_len=2, pred_len=3, skip_train=2, skip_eval=1, min_agents=2)
    bins = np.concatenate([np.arange(3, 8), np.arange(3, 8)])
    tag = np.array([1] * 5 + [2] * 5)
    brk = np.zeros(10, dtype=bool)
    brk[[0, 5]] = True
    row0, _, scene_bin, _ = window_index(bins, brk, np.zeros(10, dtype=np.int8), tag, cfg)
    assert len(scene_bin) == 0 and len(row0) == 0  # first_w=4 > last_w=3
    bins = np.concatenate([np.arange(3, 9), np.arange(3, 9)])
    tag = np.array([1] * 6 + [2] * 6)
    brk = np.zeros(12, dtype=bool)
    brk[[0, 6]] = True
    _, _, scene_bin, _ = window_index(bins, brk, np.zeros(12, dtype=np.int8), tag, cfg)
    assert scene_bin.tolist() == [4]


def test_scenes_agents_present_in_all_frames(pipe: PipelineResult) -> None:
    """장면의 모든 에이전트 궤적이 프레임 테이블의 한 태그와 20 프레임 전부 일치해야 한다."""
    frames = scan_frames(pipe.frames_dir).collect()
    seq_len = PipelineConfig().seq_len
    for name in SPLITS:
        ss = SceneSet.load(pipe.out_dir / f"{name}.npz")
        assert pipe.scenes[name] == len(ss) and pipe.agents[name] == len(ss.pos)
        assert ss.pos.shape[1:] == (seq_len, 2) and ss.pos.dtype == np.float64
        assert (ss.num_agents >= 2).all()
        assert len(ss.meta) == len(ss)
        if len(ss) == 0:
            continue
        long = ss.to_long_frame()
        zone = np.array([int(m[0].split("=")[1]) for m in ss.meta], dtype=np.int16)
        start = np.array([m[1] for m in ss.meta], dtype=np.int64)
        long = long.with_columns(
            zone_id=pl.Series(zone[long["scene"].to_numpy()]),
            bin=pl.Series(start[long["scene"].to_numpy()]) + pl.col("t"),
            x=pl.col("x").cast(pl.Float32),
            y=pl.col("y").cast(pl.Float32),
        )
        key = frames.select("zone_id", "bin", "tag_id", pl.col("x").round(4), pl.col("y").round(4))
        hit = long.join(key, on=["zone_id", "bin", "x", "y"], how="left")
        assert hit["tag_id"].null_count() == 0, "장면 좌표가 프레임에 없다"
        per_agent = hit.group_by("agent").agg(
            pl.col("tag_id").n_unique().alias("tags"), pl.len().alias("rows")
        )
        assert (per_agent["tags"] == 1).all() and (per_agent["rows"] == seq_len).all()
        # 같은 장면 안에 같은 태그가 두 번 있으면 안 되고, 에이전트는 태그 오름차순
        per_scene = (
            hit.group_by("scene", "agent").agg(pl.col("tag_id").first()).sort("scene", "agent")
        )
        assert (
            per_scene.group_by("scene")
            .agg((pl.col("tag_id").diff().drop_nulls() > 0).all().alias("inc"))["inc"]
            .all()
        )


def test_time_split_no_leakage(pipe: PipelineResult) -> None:
    lo, val_start, test_start, hi = pipe.split_bounds
    seq_len = PipelineConfig().seq_len
    meta = json.loads((pipe.out_dir / META_NAME).read_text(encoding="utf-8"))
    assert tuple(meta["split_bounds"]) == pipe.split_bounds
    ranges = {}
    for name in SPLITS:
        ss = SceneSet.load(pipe.out_dir / f"{name}.npz")
        starts = np.array([m[1] for m in ss.meta], dtype=np.int64)
        assert len(starts) > 0, name
        ranges[name] = (
            starts.min(),
            starts.max() + seq_len - 1,
        )  # 윈도우가 덮는 [첫 빈, 마지막 빈]
    assert ranges["train"][1] < val_start <= ranges["val"][0]
    assert ranges["val"][1] < test_start <= ranges["test"][0]
    assert ranges["test"][1] <= hi and ranges["train"][0] >= lo
    # 학습 stride 4 → 학습 시작 빈은 4 의 배수, 평가는 stride 1
    tr = np.array([m[1] for m in SceneSet.load(pipe.out_dir / "train.npz").meta], dtype=np.int64)
    assert (tr % 4 == 0).all()


def test_sceneset_roundtrip_with_agent_type(pipe: PipelineResult, tmp_path: Path) -> None:
    ss = SceneSet.load(pipe.out_dir / "val.npz")
    assert (
        ss.agent_type is not None
        and ss.agent_type.dtype == np.int8
        and len(ss.agent_type) == len(ss.pos)
    )
    assert set(np.unique(ss.agent_type)) <= {0, 1} and (ss.agent_type == 1).any()
    i = int(np.argmax(ss.num_agents))
    assert (
        ss.types(i).tolist() == ss.agent_type[ss.scene_index[i, 0] : ss.scene_index[i, 1]].tolist()
    )
    ss.save(tmp_path / "rt.npz")
    back = SceneSet.load(tmp_path / "rt.npz")
    assert np.array_equal(back.pos, ss.pos) and np.array_equal(back.scene_index, ss.scene_index)
    assert np.array_equal(back.agent_type, ss.agent_type) and back.meta == ss.meta
    assert back.obs_len == 8 and back.pred_len == 12


def test_smoothing_flag(pipe: PipelineResult) -> None:
    zone = list_zones(pipe.frames_dir)[0]
    base = load_zone(pipe.frames_dir, zone, PipelineConfig(), pipe.split_bounds)
    for mode in ("ema", "median3"):
        sm = load_zone(pipe.frames_dir, zone, PipelineConfig(smoothing=mode), pipe.split_bounds)
        assert len(sm) == len(base) and sm["x"].dtype == pl.Float32
        diff = np.abs(sm["x"].to_numpy() - base["x"].to_numpy())
        assert diff.max() < 1.0 and diff.mean() > 0  # 바뀌지만 잡음 크기 안에서만
        if mode == "ema":
            # EMA 의 세그먼트 첫 프레임은 자기 자신 — 평활이 세그먼트(결손·구역·분할 경계)를 넘지 않는다는 증거.
            # (median3 은 center=True 라 첫 프레임도 이웃과 섞이므로 이 검사가 성립하지 않는다.)
            first = sm.filter(pl.col("brk"))["x"].to_numpy()
            assert np.allclose(first, base.filter(pl.col("brk"))["x"].to_numpy(), atol=1e-6)
    with pytest.raises(ValueError):
        load_zone(pipe.frames_dir, zone, PipelineConfig(smoothing="lowpass"), pipe.split_bounds)  # type: ignore[arg-type]


def test_schemas_validate_samples(sim_dir: Path, pipe: PipelineResult) -> None:
    # 소형 공장(40 × 30) 좌표는 기본 평면(120 × 80) 스키마 안에 들어간다
    raw = validate_sample(scan_raw(sim_dir), RawRtlsFrame, n=1500)
    assert (
        1000 <= len(raw) <= 1500 and raw["ts_ms"].n_unique() > 100
    )  # 첫 파티션만이 아니라 전 구간 표본
    fr = validate_sample(scan_frames(pipe.frames_dir), ResampledFrame, n=1500)
    assert len(fr) > 500
    with pytest.raises(pa.errors.SchemaErrors):
        ResampledFrame.validate(pl.concat([fr, fr.head(1)]), lazy=True)  # (tag_id, bin) 중복
    with pytest.raises(pa.errors.SchemaError):
        RawRtlsFrame.validate(raw.with_columns(pl.col("tag_id").cast(pl.Int64)))  # dtype 드리프트
    with pytest.raises(pa.errors.SchemaError):
        RawRtlsFrame.validate(raw.drop("quality"))  # 열 누락


def test_stats_json(sim_dir: Path, pipe: PipelineResult) -> None:
    st = json.loads((pipe.out_dir / "stats.json").read_text(encoding="utf-8"))
    man = load_manifest(sim_dir)
    assert st["raw"]["rows"] == man["total_rows"] == pipe.rows_raw
    assert st["raw"]["duplicate_keys"] > 0
    assert st["frames"]["rows"] == pipe.rows_frames
    assert sum(st["frames"]["rows_per_zone"].values()) == pipe.rows_frames
    assert st["near_miss"]["truth_count"] == man["near_miss"]["count"] >= 1
    assert "truth_by_state" in st["near_miss"] and st["frames"]["agents_per_zone_frame"]["max"] >= 2
    assert (pipe.out_dir / FRAMES_DIR).is_dir() and pipe.peak_rss_mb > 0
