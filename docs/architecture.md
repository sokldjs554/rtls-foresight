# 아키텍처

```mermaid
flowchart LR
  subgraph DATA["데이터 (foresight.data)"]
    A1[ETH/UCY 원본<br/>raw.githubusercontent + sha256] --> A2[장면 윈도우<br/>Polars + numpy, 1 s/분할]
    S1[합성 RTLS 생성기<br/>10 Hz · 수천만 행 · Parquet] --> S2[Polars lazy/streaming<br/>2.5 Hz 리샘플 · 구역 윈도우]
    A2 --> G[그래프 텐서 V, A<br/>정규화 라플라시안 (벡터화)]
    S2 --> G
  end
  subgraph TRAIN["학습·평가 (foresight.train / eval)"]
    G --> T[Social-STGCNN<br/>7,563 params · SGD 250 ep]
    T --> E[3 프로토콜 ADE/FDE<br/>+ CVM 기준선 + 재현표]
    T -. Hydra 설정 · 시드 .-> M[(MLflow SQLite<br/>run · 지표 · 레지스트리)]
    E --> R[results/*.json<br/>README 수치 동기화]
  end
  subgraph INFER["추론 최적화 (foresight.inference)"]
    T --> X[ONNX export<br/>dynamic N] --> Q[정적 INT8<br/>ORT 양자화]
    X --> B[벤치마크<br/>eager · compile · ORT fp32/int8]
  end
  subgraph SERVE["서빙·스트리밍 (foresight.serving)"]
    Q --> P[Predictor<br/>torch / onnx / onnx-int8]
    P --> API[FastAPI /predict /risk<br/>Prometheus /metrics]
    P --> ST[스트림 소비자<br/>replay 또는 Kafka]
    ST --> RK[충돌 위험 점수<br/>P(min dist < d_safe)] --> AL[경보 정책<br/>임계값·연속·쿨다운]
  end
  subgraph OPS["MLOps"]
    D[DVC 파이프라인<br/>S3/MinIO 원격] ~~~ CI[GitHub Actions<br/>lint · 테스트 · 스모크 · 이미지]
    CI ~~~ DK[Docker Compose<br/>api · mlflow · minio · redpanda]
    DK ~~~ TF[Terraform AWS<br/>S3 · ECR · ECS Fargate · ALB]
  end
```

## 모듈 경계
| 패키지 | 책임 | 계약 |
|---|---|---|
| `foresight.data` | 원본 획득·검증, 장면 윈도우, 그래프 텐서, 합성 RTLS, 대규모 파이프라인, 스키마 | `SceneSet` npz, `scene_to_graph` |
| `foresight.models` | Social-STGCNN(재현), CVM, 손실 | `forward(v, a) -> (B,5,T_pred,N)` |
| `foresight.train` | Hydra 설정 학습 루프, MLflow | 체크포인트 `{state_dict, model, ...}` |
| `foresight.eval` | ADE/FDE 프로토콜, 재현표, 그림, 충돌 경보 평가 | `results/reproduction.json` |
| `foresight.inference` | Predictor 인터페이스, ONNX/INT8, 벤치마크 | `Prediction(params, mean_abs, samples_abs, timing_ms)` |
| `foresight.serving` | 위험 점수, 경보 정책, FastAPI, 스트림 소비자 | `POST /predict`, `POST /risk`, Kafka 토픽 |

## 요청 경로 (스트리밍)
1. 10 Hz 위치 메시지 → 0.4 s 빈으로 집계 → 태그별 8 프레임 링버퍼 (구역별 그룹).
2. 매 프레임, 구역마다 `(N, 8, 2)` 관측 → 전처리(0.3 ms) → 모델(ONNX, ~1 ms) → 20 샘플.
3. 작업자-차량 쌍마다 `P(min_t dist < d_safe)` → 경보 정책(연속 2프레임 + 5 s 쿨다운) → 경보 토픽.

지연 예산은 프레임 주기 400 ms 다. 측정값은 `docs/inference_optimization.md`, `docs/serving_streaming.md`.
