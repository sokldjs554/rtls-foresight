# 파일 재생 → Kafka/Redpanda 로 흘려 보내기 (how-to)

`foresight stream --source replay` 는 브로커 없이 파일을 직접 재생한다. 실제 Kafka 경로(`--source kafka`)를
로컬에서 검증하려면 Redpanda 컨테이너 하나와 생산자 스크립트 몇 줄이면 된다.

## 1. Redpanda 띄우기

```bash
docker run -d --name redpanda -p 9092:9092 \
  docker.redpanda.com/redpandadata/redpanda:latest \
  redpanda start --overprovisioned --smp 1 --memory 512M --node-id 0 \
  --kafka-addr PLAINTEXT://0.0.0.0:9092 --advertise-kafka-addr PLAINTEXT://127.0.0.1:9092
docker exec redpanda rpk topic create rtls.positions rtls.alerts -p 4
```

## 2. 클라이언트 설치

```bash
pip install 'rtls-foresight[stream]'   # confluent-kafka
```

## 3. 생산자: SceneSet npz 또는 RTLS parquet 을 JSON 메시지로

메시지 스키마는 `KafkaSource` 가 읽는 것과 같다: `{ts_ms, tag_id, agent_type, zone_id, x, y}`.
`ReplaySource` 가 이미 프레임 단위 `Record` 를 만들어 주므로 그대로 직렬화하면 된다.

```python
# scripts/replay_to_kafka.py (예시 — 필요할 때 만들어 쓴다)
import json, sys
from confluent_kafka import Producer
from foresight.serving.stream import ReplaySource

path, speed = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
p = Producer({"bootstrap.servers": "127.0.0.1:9092", "linger.ms": 5})
src = ReplaySource(path, speed=speed, vehicle_every_other=path.endswith(".npz"))
for frame in src.frames():  # speed 배속으로 2.5 Hz 페이싱
    for r in frame:
        p.produce(
            "rtls.positions",
            key=str(r.zone_id),  # 구역 = 파티션 키 → 같은 구역은 같은 소비자로
            value=json.dumps(
                {
                    "ts_ms": r.ts_ms,
                    "tag_id": r.tag_id,
                    "agent_type": r.agent_type,
                    "zone_id": r.zone_id,
                    "x": r.x,
                    "y": r.y,
                }
            ),
        )
    p.poll(0)
p.flush()
```

```bash
python scripts/replay_to_kafka.py data/processed/ethucy/zara1/test.npz 10
```

## 4. 소비자

```bash
foresight stream --source kafka --bootstrap 127.0.0.1:9092 --topic rtls.positions \
                 --sink kafka --alerts-topic rtls.alerts --backend onnx
docker exec redpanda rpk topic consume rtls.alerts     # 경보 확인
```

## 메모
* 파티션 키를 `zone_id` 로 두면 한 구역의 태그가 항상 같은 파티션·소비자로 가므로 `FrameAssembler` 가
  구역 그래프를 온전히 볼 수 있다. 소비자를 늘릴 때는 파티션 수 ≥ 소비자 수.
* `KafkaSource` 는 새 빈(ts_ms // 400)이 시작될 때 이전 빈을 닫는다. 생산자가 멈추면 1 s 뒤에 열린 빈을 강제로
  닫는다 (`flush_after_s`).
* confluent-kafka 가 없으면 `--source kafka` 는 설치 안내와 함께 실패하고, `--sink kafka` 는 stdout 으로 내려간다.
