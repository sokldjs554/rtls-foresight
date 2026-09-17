"""스트리밍 소비자 — 프레임 조립기 단위 테스트 + zara1 테스트 분할 재생(50×, 10 s) 통합 테스트."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from foresight.serving.risk import AlertPolicy
from foresight.serving.stream import (
    BIN_MS,
    FrameAssembler,
    KafkaSource,
    Record,
    ReplaySource,
    StdoutSink,
    StreamPipeline,
    make_sink,
    run_stream,
)
from foresight.utils import project_root

ROOT = project_root()
ZARA1 = ROOT / "data" / "processed" / "ethucy" / "zara1" / "test.npz"


def _rec(tag: str, b: int, x: float, zone: int = 0, typ: int = 0) -> Record:
    return Record(b * BIN_MS, tag, typ, zone, x, 0.0)


def test_assembler_completes_after_eight_consecutive_bins() -> None:
    asm = FrameAssembler(obs_len=8, stale_bins=3)
    for b in range(7):
        cur = asm.push([_rec("t1", b, float(b))])
        assert asm.complete_zones(cur) == []
    cur = asm.push([_rec("t1", 7, 7.0), _rec("t2", 7, 0.0, zone=1, typ=1)])
    zones = asm.complete_zones(cur)
    assert len(zones) == 1 and zones[0].zone_id == 0 and zones[0].ids == ["t1"]
    assert zones[0].obs.shape == (1, 8, 2) and np.allclose(zones[0].obs[0, :, 0], np.arange(8))


def test_assembler_gap_breaks_continuity_and_duplicates_overwrite() -> None:
    asm = FrameAssembler(obs_len=8)
    for b in list(range(4)) + list(range(5, 10)):  # 빈 4 누락
        cur = asm.push([_rec("t", b, float(b))])
    assert asm.complete_zones(cur) == []  # 링 버퍼에 8 개가 있어도 연속이 아니다
    for b in range(10, 13):
        cur = asm.push([_rec("t", b, float(b)), _rec("t", b, 100.0)])  # 같은 빈 중복 → 마지막 값
    zones = asm.complete_zones(cur)
    assert len(zones) == 1 and zones[0].obs[0, -1, 0] == 100.0


def test_assembler_evicts_stale_tags() -> None:
    asm = FrameAssembler(obs_len=8, stale_bins=3)
    asm.push([_rec("old", 0, 0.0), _rec("new", 0, 0.0)])
    cur = asm.push([_rec("new", 5, 0.0)])
    assert asm.evict(cur) == 1 and set(asm.tags) == {"new"}


def test_replay_source_from_npz_yields_ordered_frames() -> None:
    if not ZARA1.exists():
        pytest.skip("zara1 test split missing")
    src = ReplaySource(ZARA1, speed=0, stride_frames=4, vehicle_every_other=True, max_scenes=3)
    frames = list(src.frames())
    assert len(frames) == 2 * 4 + 20  # 장면 3 개, stride 4 → 마지막 장면이 프레임 8~27
    bins = [f[0].bin for f in frames]
    assert bins == sorted(bins) and all(len({r.bin for r in f}) == 1 for f in frames)
    types = {r.tag_id: r.agent_type for r in frames[0]}
    assert set(types.values()) == {0, 1}


def test_kafka_source_degrades_gracefully_without_confluent_kafka() -> None:
    try:
        import confluent_kafka  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="confluent-kafka"):
            KafkaSource("localhost:9092", "rtls.positions")
        assert isinstance(make_sink("kafka"), StdoutSink)
    else:
        pytest.skip("confluent-kafka installed; graceful-degradation path not exercised")


class _FakePredictor:
    """직선 외삽 예측기 — 파이프라인 배선을 모델 없이 검증한다."""

    name = "fake"

    def predict(self, obs: np.ndarray, k: int = 0, seed: int | None = None):
        from foresight.inference.predictor import Prediction

        vel = obs[:, -1] - obs[:, -2]  # (N, 2)
        steps = np.arange(1, 13)[None, :, None]
        mean = obs[:, -1:, :] + vel[:, None, :] * steps
        samples = np.repeat(mean[None], max(k, 1), axis=0)
        return Prediction(
            params=np.zeros((12, obs.shape[0], 5)),
            mean_abs=mean,
            samples_abs=samples if k else None,
            timing_ms={},
        )


def test_pipeline_emits_alert_for_head_on_pair() -> None:
    buf = io.StringIO()
    pipe = StreamPipeline(
        _FakePredictor(),
        StdoutSink(buf),
        AlertPolicy(threshold=0.3, cooldown_s=5.0, min_consecutive=2),
        d_safe=1.0,
        k=4,
    )
    frames = []
    for b in range(12):
        frames.append(
            [
                Record(b * BIN_MS, "w", 0, 7, -6.0 + 0.4 * b, 0.0),
                Record(b * BIN_MS, "v", 1, 7, 6.0 - 0.6 * b, 0.0),
            ]
        )
    emitted = [pipe.process_frame(f) for f in frames]
    n = [len(e) for e in emitted]
    assert sum(n) >= 1 and n[7] == 0 and n[8] == 1  # 8 번째 빈에서 완성 → 2 연속 후 첫 경보
    lines = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert (
        lines[0]["worker_id"] == "w"
        and lines[0]["vehicle_id"] == "v"
        and lines[0]["zone_id"] == 7
        and lines[0]["risk"] == 1.0
    )
    s = pipe.stats.summary()
    assert s["frames"] == 12 and s["alerts"] == sum(n) and s["zones_evaluated"] == 5


@pytest.mark.skipif(not ZARA1.exists(), reason="zara1 test split missing")
def test_run_stream_replay_zara1_50x_10s(tmp_path: Path) -> None:
    out = tmp_path / "alerts.jsonl"
    stats = run_stream(
        source="replay",
        sink=str(out),
        backend="onnx",
        replay_file=ZARA1,
        speed=50.0,
        max_seconds=10.0,
        vehicle_every_other=True,
    )
    assert stats["frames"] > 100 and stats["zones_evaluated"] > 50 and stats["agents_predicted"] > 0
    assert stats["e2e_p95_ms"] < 400  # 실시간 예산(400 ms 빈) 안
    assert stats["alerts"] >= 1
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == stats["alerts"]
    assert {"worker_id", "vehicle_id", "risk", "ttc_s", "zone_id", "ts_ms", "backend"} <= set(
        lines[0]
    )
    assert all(line["risk"] >= 0.3 for line in lines)
