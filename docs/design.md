# 설계 문서 (착수 시점)

> 구현 전에 쓴 문서다. 이후 바뀐 부분(모듈 이름·CLI·데이터 경로)은 [architecture.md](architecture.md) 와 [cli.md](cli.md) 가 우선한다.

## 한 줄 요약
UWB/RTLS 위치 스트림에서 작업자·장비의 **4.8초 뒤 궤적을 예측**하고, 예측 분포로부터 **충돌 위험을 사전 경보**하는 AI 파이프라인.
Social-STGCNN(CVPR 2020)을 PyTorch로 처음부터 구현해 ETH/UCY 5개 분할의 ADE/FDE를 재현하고,
합성 공장 RTLS 스트림(수천만 행)에 전이·미세조정한 뒤 ONNX/INT8 추론 최적화, Kafka 스트리밍 소비자, FastAPI 서빙까지 연결한다.

## 왜 이 주제인가 (근거)
- 회사(데이터플로)의 자체 제품 TPAM = UWB RTLS 기반 스마트 안전 플랫폼(출입통제·충돌방지·동선추적·이상징후). Confluent(Kafka)·Databricks·Verta 파트너.
- 국내 포트폴리오 조사(부트캠프·취준·데이콘·GitHub 6개 각도, 200+ 주제 집계): 궤적 예측·충돌 위험 예측은 사실상 0건. 논문 재현/추론 최적화/대규모 처리/MLOps도 공백.
- 지원자 기존 저장소(로그 이상탐지, RUL, 웨이퍼/PCB 비전, NPU 경량화, PPE YOLO, Text-to-SQL, RAG)와 겹치지 않음.

## 공고 스킬 → 산출물 매핑
| 공고 항목 | 산출물 |
|---|---|
| AI 모델 설계 및 학습 | `foresight/models/social_stgcnn.py` (from scratch), CVM·LSTM 베이스라인, 충돌 위험 헤드 |
| 데이터 전처리 파이프라인 | `foresight/data/*`: 다운로드→Parquet→장면 윈도우→그래프(V,A) 벡터화, Pandera 스키마, DVC 스테이지 |
| 모델 성능 평가 및 개선 | ADE/FDE(best-of-20, per-scene, deterministic), 평가 프로토콜 비판, 미세조정/전이, 충돌 경보 PR·lead time |
| 서비스 적용을 위한 추론 최적화 | ONNX export, ORT fp32/INT8(static), torch.compile, 전처리 벡터화, p50/p95/p99 벤치 |
| 실험 결과 문서화 및 공유 | MLflow 실험 추적·모델 레지스트리, docs/ 리포트, 모델·데이터 카드, ADR, README 수치 자동 동기화 |
| Python / ML 이론 / 데이터 분석·전처리 | 전체, EDA 노트북, bivariate Gaussian NLL 유도 문서 |
| Git 기반 협업 | conventional commits, PR 템플릿, CI, pre-commit, CODEOWNERS |
| PyTorch | 모델·학습·양자화 |
| MLOps 도구 | MLflow, DVC, Hydra, GitHub Actions, pre-commit, Docker Compose, Prometheus |
| 클라우드 환경 운영 | Terraform(AWS S3+ECR+ECS Fargate+ALB), MinIO(S3 호환) 로컬, GHCR 이미지, render.yaml |
| 대규모 데이터 처리 | 합성 RTLS 생성기(10Hz, 수천만 행, 파티션 Parquet) + Polars lazy/streaming + DuckDB, 메모리 상한 검증 |
| 논문 구현 및 재현 | Social-STGCNN 재현표(5분할×3시드), 공식 체크포인트 로드 검증, CVM(Schöller 2020) 재현 |

## 데이터
- ETH/UCY (SGAN 포맷 `frame ped x y`, 2.5fps): Social-STGCNN 공식 저장소 vendored 파일을 raw.githubusercontent.com 에서 다운로드(sha256 고정). 5 leave-one-out 분할.
- 합성 RTLS: 공장 레이아웃(구역·통로), 작업자(보행 1.0–1.6 m/s) + 지게차/AGV(최대 3 m/s, 회전반경), 목표지향 이동 + 회피(social-force 근사), UWB 노이즈(σ≈0.15m), 드롭아웃, 10Hz. near-miss(작업자-장비 거리<1.0m) 이벤트 라벨. 규모: `--hours 12 --tags 200` ≈ 86M rows.

## 재현 대상 (Social-STGCNN Table 1, ADE/FDE, best-of-20)
ETH 0.64/1.11 · HOTEL 0.49/0.85 · UNIV 0.44/0.79 · ZARA1 0.34/0.53 · ZARA2 0.30/0.48 · AVG 0.44/0.75. 파라미터 7,563(=7.6K).
학습 설정: SGD lr 0.01, 250 epoch, StepLR(150, ×0.2), 장면 1개=배치 1 + 128 장면 누적, BN은 장면 단위, val loss 최소 체크포인트.
공식 코드 특이점(문서화): 인접행렬 커널이 위치가 아니라 **상대 변위(속도)** 간 1/거리; 평가는 **보행자 단위 best-of-20**.

## 인터페이스 계약
- 장면 텐서: `V (T, N, 2)` 상대 변위, `A (T, N, N)` 정규화 라플라시안, `obs_abs (N, 2)` 마지막 관측 절대좌표.
- 모델 출력: `(T_pred, N, 5)` = (mu_x, mu_y, log_sx, log_sy, corr_logit) 상대 변위 분포.
- Predictor 프로토콜: `predict(obs_abs: np.ndarray[N,8,2]) -> PredictionResult(mu[N,12,2], cov[N,12,2,2], samples[K,N,12,2])`.
- 위험 점수: `risk(pred_a, pred_b, d_safe) = P(min_t ||x_a(t)-x_b(t)|| < d_safe)` MC 추정, 경보 = risk>τ & 쿨다운.

## 디렉터리
(README 참조) src/foresight/{data,models,train,eval,inference,serving,utils}, configs/, tests/, docs/, notebooks/, deploy/{terraform,compose}, results/, scripts/, tools/.
