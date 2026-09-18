# rtls-foresight

**UWB/RTLS 위치 스트림에서 작업자·장비의 4.8초 뒤 궤적을 예측하고, 예측 분포로 충돌 위험을 사전 경보하는 AI 파이프라인.**

Social-STGCNN(CVPR 2020)을 PyTorch 로 처음부터 구현해 ETH/UCY 5개 분할의 ADE/FDE 를 재현하고, 수천만 행의 합성 공장 RTLS
스트림으로 전이·미세조정한 뒤, ONNX/INT8 추론 최적화 → Kafka 스트리밍 소비자 → FastAPI 서빙까지 한 저장소 안에서 연결했습니다.

> 데이터는 공개 보행자 궤적(ETH/UCY)과 **합성** RTLS 스트림입니다. 실제 현장·고객 데이터는 사용하지 않았습니다.

`1. 문제` · `2. 왜 이 주제인가` · `3. 무엇이 다른가` · `4. 결과` · `5. 아키텍처` · `6. 실행` · `7. 실험 기록` · `8. 배운 것` · `9. 한계`

---

## 1. 문제

공장·물류창고·광산의 스마트 안전 플랫폼은 UWB 태그로 작업자와 지게차의 위치를 초당 여러 번 받는다. 대부분의 제품은
**"지금 거리가 r 미만이면 경보"** 라는 지오펜스 규칙을 쓴다. 이 규칙은 이미 가까워진 뒤에 울린다.

이 프로젝트가 답하려는 질문은 하나다.

> **지금까지 3.2초의 움직임으로, 4.8초 뒤 두 사람(또는 사람과 지게차)이 1 m 안으로 들어올 확률을 미리 말할 수 있는가?**

궤적을 점이 아니라 **분포**로 예측해야 "확률"을 말할 수 있다. 그래서 각 시점의 위치를 이변량 가우시안으로 내는 Social-STGCNN 을
골랐고, 그 분포에서 뽑은 같은 미래(joint sample) 안에서 두 에이전트의 최소 거리를 세어 위험 점수를 만든다.

## 2. 왜 이 주제인가

- 지원 공고의 주요업무(모델 설계·학습 / 전처리 파이프라인 / 평가·개선 / 추론 최적화 / 실험 문서화)와 우대사항(PyTorch / MLOps / 클라우드 /
  대규모 데이터 / 논문 재현)을 **한 문제 안에서 전부** 쓸 수 있는 주제를 찾았다.
- 국내 AI 포트폴리오 지형을 6개 각도(부트캠프 파이널·취업 포트폴리오·데이콘·GitHub 한국어 README·글로벌 리스트·회사 조사)로 조사했을 때
  감성분석·YOLO·RAG 챗봇·추천·집값·이탈 예측은 포화 상태였고, **궤적 예측과 충돌 위험 예측은 사실상 0건**이었다.
- 논문 재현·추론 최적화·대규모 처리·MLflow/DVC 는 우대사항인데도 포트폴리오에서 거의 보이지 않는 공백이었다.
- 회사가 파는 제품이 UWB RTLS 안전 플랫폼(충돌 방지·동선 추적)이고 Kafka·Databricks 파트너라, 이 파이프라인이 그 제품의 핵심 루프와 같은 모양이다.

의사결정 기록: [`docs/adr/0001-topic-selection.md`](docs/adr/0001-topic-selection.md)

## 3. 무엇이 다른가

| | 흔한 접근 | 이 프로젝트 |
|---|---|---|
| 논문 재현 | 라이브러리 모델 fine-tuning, "비슷하게 나옴" | 처음부터 구현 → **공식 체크포인트를 로드해 출력이 비트 단위로 같음**을 테스트로 고정 → 5분할×3시드 재현표 → 논문 서술과 공식 코드가 다른 3곳 ablation |
| 평가 | best-of-20 ADE/FDE 한 줄 | per-agent / joint / 결정적 세 프로토콜 + CVM 기준선을 나란히. **오라클 지표와 서비스 지표를 분리** |
| 데이터 | CSV 한 장을 pandas 로 | 공식 로더 대비 **200×** 빠른 벡터화 전처리(동일성 검증), 합성 RTLS **수천만 행**을 Polars 스트리밍 + DuckDB 로 4 GB 안에서 처리 |
| 추론 | 학습 코드로 그대로 서빙 | ONNX(dynamic N) → 정적 INT8, eager/compile/ORT 벤치, **전처리까지 포함한** 지연 예산 |
| 서비스 | 노트북에서 끝 | FastAPI + Prometheus, Kafka/Redpanda 스트림 소비자, 위험 점수 → 경보 정책(연속 프레임 + 쿨다운) |
| 문서 | README 수치 손으로 복사 | MLflow run·레지스트리, DVC 파이프라인, README 수치를 `results/*.json` 에서 **자동 동기화**(CI 검사) |

## 4. 결과

### 4.1 논문 재현 (Social-STGCNN Table 1, ADE/FDE m, best-of-20)

<!-- REPRODUCTION_TABLE:START -->
| 분할 | 논문 (ADE/FDE) | 공식 ckpt 재평가 | 우리 학습 best-of-20 | joint best-of-20 | 결정적(μ) | CVM | CVM-S(20) |
|---|---|---|---|---|---|---|---|
| eth | 0.64/1.11 | 0.73/1.21 | 0.74±0.01/1.32±0.01 | 0.87/1.72 | 1.01/1.99 | 1.00/2.23 | 0.85/1.89 |
| hotel | 0.49/0.85 | 0.41/0.69 | 0.46±0.00/0.77±0.01 | 0.62/1.20 | 0.72/1.42 | 0.32/0.62 | 0.24/0.46 |
| univ | 0.44/0.79 | 0.49/0.91 | 0.47±0.00/0.86±0.00 | 0.69/1.32 | 0.68/1.34 | 0.52/1.17 | 0.39/0.82 |
| zara1 | 0.34/0.53 | 0.33/0.52 | 0.34±0.00/0.54±0.00 | 0.48/0.92 | 0.54/1.08 | 0.43/0.96 | 0.31/0.62 |
| zara2 | 0.30/0.48 | 0.30/0.48 | 0.33±0.00/0.55±0.00 | 0.49/0.93 | 0.48/0.97 | 0.33/0.73 | 0.23/0.48 |
| **avg** | **0.44/0.75** | 0.45/0.76 | **0.47/0.81** | 0.63/1.22 | 0.69/1.36 | 0.52/1.14 | 0.40/0.85 |
<!-- REPRODUCTION_TABLE:END -->

세 열을 같이 읽어야 한다. **논문**은 저자가 보고한 값, **공식 ckpt 재평가**는 저자의 가중치를 이 저장소의 평가기로 돌린 값,
**우리 학습**은 처음부터 구현한 코드로 CPU 에서 250 epoch 학습한 값이다. 자세한 논의와 ablation: [`docs/paper_reproduction.md`](docs/paper_reproduction.md)

### 4.2 같은 모델, 다른 프로토콜 — 숫자가 어떻게 달라지는가

5분할 평균 ADE/FDE (m). best-of-20 은 정답을 알고 고르는 **오라클** 지표이고, 결정적(μ)은 서비스가 실제로 내놓는 한 점이다.

| 프로토콜 | 우리 학습 | 등속 모델(CVM) |
|---|---|---|
| per-agent best-of-20 (논문) | <!-- num:reproduction.ours.avg.best_of_k_per_agent.ade -->0.47<!-- /num --> / <!-- num:reproduction.ours.avg.best_of_k_per_agent.fde -->0.81<!-- /num --> | CVM-S(20): <!-- num:reproduction.ours.avg.cvm_sampled.ade -->0.40<!-- /num --> / <!-- num:reproduction.ours.avg.cvm_sampled.fde -->0.85<!-- /num --> |
| joint best-of-20 (장면 단위 한 샘플) | <!-- num:reproduction.ours.avg.best_of_k_joint.ade -->0.63<!-- /num --> / <!-- num:reproduction.ours.avg.best_of_k_joint.fde -->1.22<!-- /num --> | – |
| 결정적 μ (샘플 없음) | <!-- num:reproduction.ours.avg.deterministic.ade -->0.69<!-- /num --> / <!-- num:reproduction.ours.avg.deterministic.fde -->1.36<!-- /num --> | <!-- num:reproduction.ours.avg.cvm.ade -->0.52<!-- /num --> / <!-- num:reproduction.ours.avg.cvm.fde -->1.14<!-- /num --> |

결정적 예측만 보면 학습 모델이 등속 모델보다 **나쁘다**. 모델의 가치는 분포에 있고, 그래서 충돌 위험은 μ 가 아니라 같은 미래(joint sample)의
최소 거리로 계산한다. 논의: [`docs/evaluation_methodology.md`](docs/evaluation_methodology.md)

### 4.3 대규모 RTLS 스트림 처리 (합성, 200 태그 × 12 시간 × 10 Hz)

| 항목 | 값 |
|---|---|
| 원시 행 | <!-- num:data_pipeline/full_pipeline_meta.rows_raw:,d -->85,423,957<!-- /num --> (1.6 GB Parquet, 시간 파티션) |
| 2.5 Hz 프레임 → 장면 | <!-- num:data_pipeline/full_pipeline_meta.rows_frames:,d -->21,576,189<!-- /num --> 프레임 → train/val/test <!-- num:data_pipeline/full_pipeline_meta.scenes.train:,d -->443,180<!-- /num --> / <!-- num:data_pipeline/full_pipeline_meta.scenes.val:,d -->378,260<!-- /num --> / <!-- num:data_pipeline/full_pipeline_meta.scenes.test:,d -->378,943<!-- /num --> 장면 |
| 벽시계 · 처리량 | <!-- num:data_pipeline/full_pipeline_meta.timings.total:.0f -->92<!-- /num --> s · 약 <!-- num:data_pipeline/full_pipeline_meta.rows_per_s:,.0f -->933,275<!-- /num --> rows/s (Polars streaming, 2 스레드) |
| 피크 RSS | <!-- num:data_pipeline/full_pipeline_meta.peak_rss_gb:.2f -->2.24<!-- /num --> GB (상한 4 GB) — 처음 설계는 10.9 GB 로 OOM, 시간 파티션 단위로 바꿔 해결 |

생성기·품질 규칙·리샘플·윈도우와 pandas/DuckDB 비교: [`docs/data_pipeline.md`](docs/data_pipeline.md)

### 4.4 추론 최적화 (CPU 1 스레드, 전처리+모델+20 샘플 후처리 포함, N=20 장면)

| 백엔드 | p50 (ms) | p95 (ms) | 처리량 (장면/s) |
|---|---|---|---|
| PyTorch eager | <!-- num:benchmark.backends.torch-eager/t1.20.k20.p50_ms -->2.32<!-- /num --> | <!-- num:benchmark.backends.torch-eager/t1.20.k20.p95_ms -->3.16<!-- /num --> | <!-- num:benchmark.backends.torch-eager/t1.20.k20.throughput_scenes_per_s:,.0f -->413<!-- /num --> |
| torch.compile | <!-- num:benchmark.backends.torch-compile/t1.20.k20.p50_ms -->1.88<!-- /num --> | <!-- num:benchmark.backends.torch-compile/t1.20.k20.p95_ms -->1.98<!-- /num --> | <!-- num:benchmark.backends.torch-compile/t1.20.k20.throughput_scenes_per_s:,.0f -->528<!-- /num --> |
| **ONNX Runtime fp32** | **<!-- num:benchmark.backends.onnx-fp32/t1.20.k20.p50_ms -->1.40<!-- /num -->** | <!-- num:benchmark.backends.onnx-fp32/t1.20.k20.p95_ms -->1.49<!-- /num --> | <!-- num:benchmark.backends.onnx-fp32/t1.20.k20.throughput_scenes_per_s:,.0f -->712<!-- /num --> |
| ONNX Runtime INT8 (정적, TXP-CNN) | <!-- num:benchmark.backends.onnx-int8/t1.20.k20.p50_ms -->1.45<!-- /num --> | <!-- num:benchmark.backends.onnx-int8/t1.20.k20.p95_ms -->1.55<!-- /num --> | <!-- num:benchmark.backends.onnx-int8/t1.20.k20.throughput_scenes_per_s:,.0f -->683<!-- /num --> |

- 전처리 벡터화: 공식 networkx 경로 대비 **×<!-- num:benchmark.preprocess.20.speedup:.0f -->100<!-- /num -->** (N=20), 결과 차이 < 1e-6.
- INT8 은 7.6K 파라미터 모델에서 **속도 이득이 없다** (Q/DQ 오버헤드). 정확도 손실 ADE +0.04(TXP-CNN 만) ~ +0.09(전체).
- 토치 4 스레드는 OpenMP 스핀 대기로 100배 느려진다 → 서빙은 1 스레드 + 프로세스 확장. 논문 주장 0.002 s/frame 대비 CPU 에서 N≤20 이면 2 ms 이내.
- 자세한 표·그림: [`docs/inference_optimization.md`](docs/inference_optimization.md) · 서빙 API/스트리밍/부하테스트: [`docs/serving_streaming.md`](docs/serving_streaming.md)

### 4.5 RTLS 전이와 충돌 경보 (합성 스트림 테스트 1.8 h)

<!-- RTLS_TRANSFER_TABLE:START -->
(`foresight evaluate-rtls` 가 채운다)
<!-- RTLS_TRANSFER_TABLE:END -->

<!-- COLLISION_TABLE:START -->
(`foresight evaluate-rtls` 가 채운다)
<!-- COLLISION_TABLE:END -->

양성 = 4.8 s 안에 작업자-차량 거리 < 1.0 m 인 쌍. 지오펜스(현재 거리)는 "이미 가까운" 쌍을 정확히 잡지만 선행시간이 0.4 s 안팎이고,
분포 기반 위험 점수는 선행시간을 산다. 정의·경보 정책·해석: [`docs/serving_streaming.md`](docs/serving_streaming.md) §3–4

## 5. 아키텍처

[`docs/architecture.md`](docs/architecture.md) 에 전체 다이어그램과 모듈 경계가 있다.

## 6. 실행

```bash
make setup && source .venv/bin/activate          # venv + CPU torch + 의존성 (이후 명령은 venv 안에서)
foresight download && foresight prepare          # ETH/UCY (14 MB, sha256 검증) → 장면 npz
foresight train dataset=eth train=paper seed=0   # 논문 설정 학습 (CPU 1스레드 ~40분) — MLflow sqlite:///mlflow.db
scripts/train_all.sh                             # 5분할 병렬
foresight evaluate                               # 재현표 results/reproduction.json
foresight simulate --profile full && foresight prepare-rtls --in-dir data/rtls/full/raw   # 합성 RTLS 85M 행 → 장면 (small 은 수 초 스모크)
foresight export && foresight benchmark          # ONNX/INT8 + 벤치마크
foresight serve --backend onnx                   # http://localhost:8000/docs
foresight stream --source replay --replay-file data/processed/rtls/test.npz
docker compose up --build                        # api + mlflow + minio
docker compose --profile stream up --build       # + redpanda + 재생 생산자 + 스트리밍 소비자
```

## 7. 실험 기록

[`docs/experiment_log.md`](docs/experiment_log.md) · MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

## 8. 배운 것

- **"재현"의 기준은 논문 본문이 아니라 공식 코드다.** 커널(위치 vs 속도), 축 교환(`permute` vs `view`), 평가(per-agent vs joint)
  세 곳에서 서술과 코드가 달랐고, 수치는 코드에서 나왔다. 코드 동작을 기본값으로, 서술을 ablation 으로 두니 차이가 설명된다.
- **비트 단위 동일성 테스트가 가장 싼 보험이다.** 커널 거리를 float64 로 계산하면 특정 장면에서 라플라시안이 완전히 달라지는데,
  공식 로더와의 동일성 검사가 없었다면 "학습이 조금 다르게 됐나 보다"로 묻혔을 것이다.
- **공식 체크포인트 + 같은 평가기 열이 없으면 학습 결과를 해석할 수 없다.** hotel 은 논문보다 좋고 univ 는 나쁜데, 저자의 가중치도
  같은 방향으로 달랐다 — 우리 학습의 문제가 아니라 평가 프로토콜·테스트 분할 크기가 만드는 분산이다.
- **best-of-20 은 오라클이고, 결정적 평균은 등속 모델보다 나쁘다.** 모델의 가치는 "정확한 한 점"이 아니라 "보정된 분포"에 있고,
  그래서 충돌 위험은 평균이 아니라 같은 미래(joint sample)의 최소 거리로 계산한다.
- **스트리밍이라고 메모리가 상한되는 것은 아니다.** `unique` + `group_by` 가 85M 행에서 10.9 GB 를 쌓아 OOM 으로 죽었다.
  키와 빈이 시간 경계를 넘지 않는다는 데이터의 성질을 쓰면 시간 파티션 단위 처리가 정확하고 2.3 GB 로 끝난다.
- **7.6K 파라미터 모델은 INT8 이 오히려 느리다.** Q/DQ 커널 오버헤드가 연산 절감보다 크다. 속도는 ONNX Runtime fp32 + 1 스레드가
  최선이었고, 병목은 모델이 아니라 20 샘플 후처리와 전처리였다. 토치 4 스레드는 OpenMP 스핀 대기로 100배 느려진다.
- **CPU 로 충분했다.** 연산이 아니라 파이썬 루프(공식 데이터 준비 211 s → 1 s)와 프로세스 병렬(분할당 1 스레드)이 시간을 결정했다.

## 9. 한계

- ETH/UCY 는 보행자 데이터고 RTLS 는 **합성**이다. 지게차의 운동학·현장 레이아웃·행동은 단순화됐다. RTLS 결과는 파이프라인과
  평가 절차의 시연이지 검증된 안전 성능이 아니다. 실제 배치 전에는 현장 데이터로 재학습·재보정이 필요하다.
- 충돌 라벨은 측정 위치(σ 0.15 m) 기준이라 d_safe 경계 근처가 흔들린다. 상대 비교는 공정하지만 절대 수치는 그만큼 보수적으로 읽어야 한다.
- best-of-20 재현 수치는 시드에 따라 ±0.01 정도 흔들리고, ETH 는 테스트가 작아(181명) 더 크게 흔들린다. 표에 표준편차를 같이 둔다.
- 벤치마크는 학습 프로세스와 같은 4 vCPU 에서 쟀다. 1 스레드 수치와 상대 비교는 안정적이지만 절대값은 유휴 머신에서 더 낮다.
- Kafka/Redpanda 경로는 파일 재생과 동일한 소비자 코드로 구현했지만, 실제 브로커 부하·재처리·정확히 한 번 의미론은 다루지 않았다.
- 클라우드는 Terraform 정의와 Compose 토폴로지까지다. 실제 AWS 배포·비용 측정은 하지 않았다.

---
문서: [논문 재현](docs/paper_reproduction.md) · [평가 방법론](docs/evaluation_methodology.md) · [데이터 파이프라인](docs/data_pipeline.md) · [추론 최적화](docs/inference_optimization.md) · [서빙·스트리밍](docs/serving_streaming.md) · [클라우드 배포](deploy/terraform/README.md) · [모델 카드](docs/model_card.md) · [데이터 카드](docs/data_card.md) · [ADR](docs/adr/)
