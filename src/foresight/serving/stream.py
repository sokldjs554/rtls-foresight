"""스트리밍 소비자 — RTLS 위치 스트림 → 2.5 Hz 프레임 조립 → 구역별 예측 → 충돌 위험 → 경보.

구성
    Source (ReplaySource | KafkaSource)  ──frames──▶  FrameAssembler  ──(zone, ids, types, obs)──▶
    Predictor(k=20)  ──▶  pairwise_risk  ──▶  AlertPolicy  ──▶  Sink (stdout | file | kafka)

프레임(bin) 단위 처리인 이유
    RTLS 태그는 10 Hz 로 비동기 도착하지만 모델은 0.4 s 간격 8 점을 본다. ``ts_ms // 400`` 으로 빈을 나누고
    빈마다 태그의 마지막 위치를 취해 2.5 Hz 로 다운샘플한다 (학습 데이터와 같은 주기). 한 빈이 닫힐 때
    한 번만 예측하므로 태그 수와 무관하게 구역당 예측은 초당 2.5 회다.

구역(zone) 단위 그래프
    Social-STGCNN 의 인접행렬은 N² 이고 멀리 떨어진 태그는 서로 영향이 없으므로, 공장 전체가 아니라
    구역별로 그래프를 만든다. 벤치마크의 "프레임당 Z 구역" 워크로드가 이 구조의 비용이다.

Kafka
    ``confluent_kafka`` 는 선택 의존성이다 (``pip install 'rtls-foresight[stream]'``). 없으면 KafkaSource 는
    설치 안내와 함께 실패하고, kafka 싱크는 stdout 으로 내려간다 — 파일 재생 경로는 항상 동작해야 한다.

백프레셔
    처리 시간이 프레임 간격(400 ms / speed)보다 길면 ReplaySource 는 잠들지 않고 바로 다음 프레임을 내
    (재생이 실시간보다 느려질 뿐 프레임을 버리지 않는다). KafkaSource 는 컨슈머 랙으로 나타난다 —
    ``stats()`` 의 ``e2e_p95_ms`` 가 빈 간격에 가까워지면 구역을 파티션으로 나눠 소비자를 늘리는 것이 답이다.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TextIO

import numpy as np

from foresight.inference.predictor import Predictor
from foresight.serving.risk import STEP_SECONDS, AlertPolicy, PairRisk, pairwise_risk
from foresight.utils import get_logger, project_root

log = get_logger("foresight.serving.stream")

BIN_MS = int(STEP_SECONDS * 1000)
DEFAULT_REPLAY = Path("data/processed/ethucy/zara1/test.npz")


@dataclass(slots=True)
class Record:
    ts_ms: int
    tag_id: str
    agent_type: int
    zone_id: int
    x: float
    y: float

    @property
    def bin(self) -> int:
        return self.ts_ms // BIN_MS


class Source(Protocol):
    def frames(self) -> Iterator[list[Record]]: ...


# ----------------------------------------------------------------------------- sources
class ReplaySource:
    """SceneSet npz 또는 RTLS long parquet 를 2.5 Hz 프레임으로 재생한다.

    npz: 장면 i 를 구역 i 로, 에이전트 j 를 태그 ``s{i}a{j}`` 로 보고, 장면 i 는 프레임 ``i*stride_frames`` 부터
    20 프레임 동안 흐른다 — 여러 장면(구역)이 동시에 살아 있어 다구역 처리를 흉내 낸다.
    ``vehicle_every_other=True`` 면 홀수 에이전트를 차량으로 표시한다 (ETH/UCY 에는 차량이 없으므로 테스트용).
    """

    def __init__(
        self,
        path: str | Path,
        speed: float = 1.0,
        stride_frames: int = 4,
        vehicle_every_other: bool = False,
        max_scenes: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.speed = float(speed)
        self.stride_frames = stride_frames
        self.vehicle_every_other = vehicle_every_other
        self.max_scenes = max_scenes

    def _from_npz(self) -> dict[int, list[Record]]:
        from foresight.data.ethucy import SceneSet

        ss = SceneSet.load(self.path)
        frames: dict[int, list[Record]] = defaultdict(list)
        n_scenes = len(ss) if self.max_scenes is None else min(len(ss), self.max_scenes)
        seq_len = ss.obs_len + ss.pred_len
        for i in range(n_scenes):
            pos = ss.scene(i)
            types = ss.types(i)
            offset = i * self.stride_frames
            for j in range(pos.shape[0]):
                a_type = int(types[j]) if not self.vehicle_every_other else int(j % 2)
                for t in range(seq_len):
                    f = offset + t
                    frames[f].append(
                        Record(
                            f * BIN_MS,
                            f"s{i}a{j}",
                            a_type,
                            i,
                            float(pos[j, t, 0]),
                            float(pos[j, t, 1]),
                        )
                    )
        return frames

    def _from_parquet(self) -> dict[int, list[Record]]:
        import polars as pl

        df = pl.read_parquet(
            self.path, columns=["ts_ms", "tag_id", "agent_type", "zone_id", "x", "y"]
        )
        # 10 Hz → 2.5 Hz: 빈마다 태그의 마지막 샘플
        df = (
            df.with_columns((pl.col("ts_ms") // BIN_MS).alias("bin"))
            .sort("ts_ms")
            .group_by(["bin", "tag_id"], maintain_order=True)
            .last()
        )
        frames: dict[int, list[Record]] = defaultdict(list)
        for row in df.iter_rows(named=True):
            b = int(row["bin"])
            frames[b].append(
                Record(
                    b * BIN_MS,
                    str(row["tag_id"]),
                    int(row["agent_type"]),
                    int(row["zone_id"]),
                    float(row["x"]),
                    float(row["y"]),
                )
            )
        return frames

    def frames(self) -> Iterator[list[Record]]:
        table = self._from_parquet() if self.path.suffix == ".parquet" else self._from_npz()
        keys = sorted(table)
        t_start = time.perf_counter()
        b0 = keys[0] if keys else 0
        for b in keys:
            if self.speed > 0:
                due = t_start + (b - b0) * (BIN_MS / 1000.0) / self.speed
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            yield table[b]


class KafkaSource:
    """Kafka/Redpanda 토픽의 JSON 메시지 ``{ts_ms, tag_id, agent_type, zone_id, x, y}`` 를 빈 단위 프레임으로 묶는다."""

    def __init__(
        self,
        bootstrap: str,
        topic: str,
        group_id: str = "foresight-stream",
        poll_timeout_s: float = 0.2,
        flush_after_s: float = 1.0,
    ) -> None:
        try:
            from confluent_kafka import Consumer
        except ImportError as e:  # 선택 의존성 — 안내와 함께 실패
            raise RuntimeError(
                "confluent-kafka 가 설치되어 있지 않습니다: pip install 'rtls-foresight[stream]' (또는 --source replay 사용)"
            ) from e
        self.consumer = Consumer(
            {
                "bootstrap.servers": bootstrap,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": True,
            }
        )
        self.consumer.subscribe([topic])
        self.poll_timeout_s = poll_timeout_s
        self.flush_after_s = flush_after_s
        self.topic = topic

    def frames(self) -> Iterator[list[Record]]:
        buf: dict[int, list[Record]] = defaultdict(list)
        current: int | None = None
        last_msg = time.perf_counter()
        while True:
            msg = self.consumer.poll(self.poll_timeout_s)
            if msg is None:
                # 잠시 조용하면 열려 있는 빈을 닫는다 (마지막 프레임이 영원히 안 나오는 것을 막기 위해)
                if current is not None and time.perf_counter() - last_msg > self.flush_after_s:
                    yield buf.pop(current)
                    current = None
                continue
            if msg.error():
                log.warning("kafka error: %s", msg.error())
                continue
            last_msg = time.perf_counter()
            try:
                d = json.loads(msg.value())
                rec = Record(
                    int(d["ts_ms"]),
                    str(d["tag_id"]),
                    int(d.get("agent_type", 0)),
                    int(d.get("zone_id", 0)),
                    float(d["x"]),
                    float(d["y"]),
                )
            except (KeyError, ValueError, TypeError) as e:
                log.warning("bad message skipped: %s", e)
                continue
            b = rec.bin
            if current is None:
                current = b
            if b > current:  # 새 빈 시작 → 이전 빈 방출
                for done in sorted(k for k in buf if k < b):
                    yield buf.pop(done)
                current = b
            buf[b].append(rec)

    def close(self) -> None:
        self.consumer.close()


# ----------------------------------------------------------------------------- assembler
@dataclass
class _TagState:
    zone_id: int
    agent_type: int
    bins: deque[int]
    xy: deque[tuple[float, float]]
    last_bin: int


@dataclass
class ZoneBatch:
    zone_id: int
    ids: list[str]
    types: np.ndarray  # (N,) int8
    obs: np.ndarray  # (N, obs_len, 2)


class FrameAssembler:
    """태그별 최근 ``obs_len`` 빈의 링 버퍼. 연속 8 빈이 채워진 태그만 예측 대상으로 내보낸다."""

    def __init__(self, obs_len: int = 8, stale_bins: int = 3) -> None:
        self.obs_len = obs_len
        self.stale_bins = stale_bins
        self.tags: dict[str, _TagState] = {}

    def push(self, records: list[Record]) -> int:
        """레코드 반영 후 현재 빈을 돌려준다. 같은 빈의 중복 샘플은 마지막 값으로 덮어쓴다."""
        cur = -1
        for r in records:
            b = r.bin
            cur = max(cur, b)
            st = self.tags.get(r.tag_id)
            if st is None:
                st = _TagState(
                    r.zone_id,
                    r.agent_type,
                    deque(maxlen=self.obs_len),
                    deque(maxlen=self.obs_len),
                    b,
                )
                self.tags[r.tag_id] = st
            if st.bins and st.bins[-1] == b:
                st.xy[-1] = (r.x, r.y)
            else:
                st.bins.append(b)
                st.xy.append((r.x, r.y))
            st.zone_id, st.agent_type, st.last_bin = r.zone_id, r.agent_type, b
        return cur

    def evict(self, current_bin: int) -> int:
        stale = [t for t, st in self.tags.items() if current_bin - st.last_bin > self.stale_bins]
        for t in stale:
            del self.tags[t]
        return len(stale)

    def complete_zones(self, current_bin: int) -> list[ZoneBatch]:
        by_zone: dict[int, list[tuple[str, _TagState]]] = defaultdict(list)
        for tid, st in self.tags.items():
            if (
                len(st.bins) == self.obs_len
                and st.bins[-1] == current_bin
                and st.bins[0] == current_bin - self.obs_len + 1
            ):
                by_zone[st.zone_id].append((tid, st))
        out: list[ZoneBatch] = []
        for zone, items in sorted(by_zone.items()):
            ids = [tid for tid, _ in items]
            types = np.array([st.agent_type for _, st in items], dtype=np.int8)
            obs = np.array([list(st.xy) for _, st in items], dtype=np.float64)
            out.append(ZoneBatch(zone, ids, types, obs))
        return out


# ----------------------------------------------------------------------------- sinks
class Sink(Protocol):
    def emit(self, alert: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class StdoutSink:
    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout

    def emit(self, alert: dict[str, Any]) -> None:
        self.stream.write(json.dumps(alert, ensure_ascii=False) + "\n")
        self.stream.flush()

    def close(self) -> None:
        pass


class FileSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def emit(self, alert: dict[str, Any]) -> None:
        self._f.write(json.dumps(alert, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._f.close()


class KafkaSink:
    def __init__(self, bootstrap: str, topic: str) -> None:
        from confluent_kafka import Producer  # ImportError 는 make_sink 에서 처리

        self.producer = Producer({"bootstrap.servers": bootstrap, "linger.ms": 5})
        self.topic = topic

    def emit(self, alert: dict[str, Any]) -> None:
        key = f"{alert.get('zone_id')}:{alert.get('worker_id')}:{alert.get('vehicle_id')}"
        self.producer.produce(self.topic, key=key, value=json.dumps(alert, ensure_ascii=False))
        self.producer.poll(0)

    def close(self) -> None:
        self.producer.flush(5)


def make_sink(
    name: str,
    bootstrap: str = "localhost:9092",
    alerts_topic: str = "rtls.alerts",
    path: str | Path | None = None,
) -> Sink:
    if name == "stdout":
        return StdoutSink()
    if name == "file":
        return FileSink(path or project_root() / "results" / "alerts.jsonl")
    if name == "kafka":
        try:
            return KafkaSink(bootstrap, alerts_topic)
        except ImportError:
            log.warning("confluent-kafka 미설치 — 경보 싱크를 stdout 으로 대체합니다")
            return StdoutSink()
    if name.endswith(".jsonl") or name.endswith(".json"):
        return FileSink(name)
    raise ValueError(f"unknown sink {name!r} (stdout | file | kafka | <path.jsonl>)")


# ----------------------------------------------------------------------------- pipeline
@dataclass
class StreamStats:
    frames: int = 0
    zones_evaluated: int = 0
    agents_predicted: int = 0
    alerts: int = 0
    evicted_tags: int = 0
    elapsed_s: float = 0.0
    e2e_ms: list[float] = field(default_factory=list, repr=False)

    def summary(self) -> dict[str, Any]:
        e2e = np.asarray(self.e2e_ms) if self.e2e_ms else np.zeros(1)
        return {
            "frames": self.frames,
            "zones_evaluated": self.zones_evaluated,
            "agents_predicted": self.agents_predicted,
            "alerts": self.alerts,
            "evicted_tags": self.evicted_tags,
            "elapsed_s": round(self.elapsed_s, 3),
            "frames_per_s": round(self.frames / self.elapsed_s, 2) if self.elapsed_s > 0 else 0.0,
            "e2e_p50_ms": round(float(np.percentile(e2e, 50)), 3),
            "e2e_p95_ms": round(float(np.percentile(e2e, 95)), 3),
            "e2e_max_ms": round(float(e2e.max()), 3),
        }


class StreamPipeline:
    def __init__(
        self,
        predictor: Predictor,
        sink: Sink,
        policy: AlertPolicy | None = None,
        d_safe: float = 1.0,
        k: int = 20,
        obs_len: int = 8,
        stale_bins: int = 3,
    ) -> None:
        self.predictor = predictor
        self.sink = sink
        self.policy = policy or AlertPolicy()
        self.d_safe, self.k = d_safe, k
        self.assembler = FrameAssembler(obs_len, stale_bins)
        self.stats = StreamStats()
        self._last_bin: dict[int, int] = {}

    def process_frame(self, records: list[Record]) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        cur = self.assembler.push(records)
        self.stats.evicted_tags += self.assembler.evict(cur)
        now_s = cur * STEP_SECONDS
        emitted: list[dict[str, Any]] = []
        for zb in self.assembler.complete_zones(cur):
            if self._last_bin.get(zb.zone_id) == cur:
                continue  # 같은 빈에 같은 구역이 두 번 오면(카프카 배치가 빈 경계를 걸칠 때) 한 번만 평가한다
            self._last_bin[zb.zone_id] = cur
            if not ((zb.types == 0).any() and (zb.types == 1).any()):
                continue  # 작업자–차량 쌍이 없으면 예측할 필요가 없다 (비용 절감)
            self.stats.zones_evaluated += 1
            self.stats.agents_predicted += len(zb.ids)
            pred = self.predictor.predict(zb.obs, k=self.k)
            samples = pred.samples_abs if pred.samples_abs is not None else pred.mean_abs[None]
            rm = pairwise_risk(samples, zb.types, self.d_safe)
            obs_pairs: list[tuple[str, str, PairRisk]] = [
                (zb.ids[p.worker], zb.ids[p.vehicle], p) for p in rm.pairs()
            ]
            for alert in self.policy.update(obs_pairs, now_s, zone_id=zb.zone_id):
                d = alert.to_dict()
                d.update({"ts_ms": cur * BIN_MS, "backend": self.predictor.name})
                self.sink.emit(d)
                emitted.append(d)
        if self.stats.frames % 250 == 0:
            self.policy.gc(now_s)
        self.stats.frames += 1
        self.stats.alerts += len(emitted)
        self.stats.e2e_ms.append((time.perf_counter() - t0) * 1e3)
        return emitted

    def run(self, source: Source, max_seconds: float | None = None) -> dict[str, Any]:
        t_start = time.perf_counter()
        try:
            for records in source.frames():
                self.process_frame(records)
                if max_seconds is not None and time.perf_counter() - t_start > max_seconds:
                    log.info("max_seconds=%.1f reached", max_seconds)
                    break
        except KeyboardInterrupt:
            log.info("interrupted")
        finally:
            self.stats.elapsed_s = time.perf_counter() - t_start
            self.sink.close()
        s = self.stats.summary()
        log.info("stream done: %s", s)
        return s


def run_stream(
    source: str = "replay",
    sink: str = "stdout",
    backend: str = "onnx",
    bootstrap: str = "localhost:9092",
    topic: str = "rtls.positions",
    alerts_topic: str = "rtls.alerts",
    replay_file: Path | None = None,
    speed: float = 10.0,
    max_seconds: float | None = None,
    d_safe: float = 1.0,
    k: int = 20,
    threshold: float = 0.3,
    cooldown_s: float = 5.0,
    min_consecutive: int = 2,
    vehicle_every_other: bool = False,
    threads: int = 1,
) -> dict[str, Any]:
    """CLI ``foresight stream`` 진입점. 통계 dict 를 돌려준다."""
    from foresight.serving import load_predictor

    if source == "replay":
        path = replay_file or project_root() / DEFAULT_REPLAY
        # ETH/UCY 재생은 차량이 없어 경보가 나올 수 없으므로 기본으로 홀수 에이전트를 차량으로 본다
        auto_vehicle = vehicle_every_other or (path.suffix == ".npz" and "ethucy" in str(path))
        src: Source = ReplaySource(path, speed=speed, vehicle_every_other=auto_vehicle)
    elif source == "kafka":
        src = KafkaSource(bootstrap, topic)
    else:
        raise ValueError("source must be 'replay' or 'kafka'")
    predictor = load_predictor(backend, threads=threads)
    policy = AlertPolicy(
        threshold=threshold, cooldown_s=cooldown_s, min_consecutive=min_consecutive
    )
    pipe = StreamPipeline(
        predictor, make_sink(sink, bootstrap, alerts_topic), policy, d_safe=d_safe, k=k
    )
    log.info(
        "stream start: source=%s sink=%s backend=%s speed=%.1fx",
        source,
        sink,
        predictor.name,
        speed,
    )
    stats = pipe.run(src, max_seconds=max_seconds)
    if hasattr(src, "close"):
        src.close()
    return stats


def produce_positions(
    replay_file: Path | str,
    bootstrap: str,
    topic: str,
    speed: float = 10.0,
    max_seconds: float | None = None,
    stride_frames: int = 4,
) -> dict[str, Any]:
    """재생 파일의 2.5 Hz 프레임을 Kafka 토픽에 위치 메시지로 발행한다 (compose 의 `replay` 서비스).

    현장에서는 UWB 엔진 → Kafka Connect 가 이 자리다. 메시지 스키마는 KafkaSource 가 읽는 것과 같다:
    {ts_ms, tag_id, agent_type, zone_id, x, y}. 키는 tag_id (같은 태그의 순서 보장).
    """
    try:
        from confluent_kafka import Producer
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "confluent-kafka 가 필요하다: pip install 'rtls-foresight[stream]'"
        ) from e
    src = ReplaySource(replay_file, speed=speed, stride_frames=stride_frames)
    producer = Producer({"bootstrap.servers": bootstrap, "linger.ms": 5})
    t_start = time.perf_counter()
    n = 0
    try:
        for records in src.frames():
            for r in records:
                payload = json.dumps(
                    {
                        "ts_ms": r.ts_ms,
                        "tag_id": r.tag_id,
                        "agent_type": r.agent_type,
                        "zone_id": r.zone_id,
                        "x": r.x,
                        "y": r.y,
                    }
                ).encode()
                producer.produce(topic, key=str(r.tag_id).encode(), value=payload)
                n += 1
            producer.poll(0)
            if max_seconds is not None and time.perf_counter() - t_start > max_seconds:
                break
    finally:
        producer.flush(10)
    stats = {"messages": n, "elapsed_s": time.perf_counter() - t_start, "topic": topic}
    log.info("produce done: %s", stats)
    return stats
