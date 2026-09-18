# 운영 런북 (Runbook)

서비스: `rtls-foresight` API (`/predict`, `/risk`, `/health`, `/metrics`) + 스트림 소비자. 이 문서는 "운영 중 무엇을 보고 무엇을 하는가"만 다룬다.
아키텍처는 `architecture.md`, 배포 정의는 `deploy/terraform`(AWS ECS Fargate + ALB), `docker-compose.yml`(로컬 토폴로지), `render.yaml`(Render).

## 1. SLO 와 경보

| 지표 | SLO | Prometheus 경보 (`deploy/monitoring/alerts.yml`) |
|---|---|---|
| `/risk` p95 지연 (동시 사용자 10, 워커 1, K = 20) | ≤ 100 ms | `ForesightHighLatency` (2 분 지속 시 warning) |
| 비 2xx 응답 비율 | ≤ 1 % | `ForesightErrorRate` (5 분, critical) |
| 요청 유입 | 10 분 내 1건 이상 | `ForesightNoTraffic` (소비자·게이트웨이 사망 감지) |
| 임계 초과 쌍(경보) 환산 | ≤ 2,000 / h | `ForesightAlertBurst` (설정 변경·노이즈 급증 감지) |

지연 SLO 의 근거: 단일 요청 벤치마크는 p50 1.4 ms(ONNX fp32, 1 스레드, `results/benchmark.json`)이고, 동시 사용자 10 명의 부하 테스트(2절)에서
워커 1개의 p95 가 55 ms 였다(학습 프로세스 2개와 CPU 를 나눠 쓴 값). 100 ms 는 그 2 배 여유이며 RTLS 프레임 주기(400 ms) 안에서
구역 수십 개를 순차 처리해도 남는 예산이다. 넘으면 워커를 늘린다(프로세스 확장).

대시보드: `docker compose up prometheus grafana` → Grafana http://localhost:3000 (익명 Viewer, 대시보드 "rtls-foresight API").
패널은 p50/p95/p99 지연, 엔드포인트·상태별 RPS, 예측 에이전트 수, 경보 환산치, 요청당 에이전트 분포, 오류 비율.

## 2. 부하 테스트 (Locust, `scripts/locustfile.py`)

<!-- LOADTEST_TABLE:START -->
| 엔드포인트 | 요청 | 실패 | RPS | p50 (ms) | p95 (ms) | p99 (ms) | 최대 (ms) |
|---|---|---|---|---|---|---|---|
| /health | 700 | 0 | 23.3 | 8 | 16 | 22 | 107 |
| /predict | 700 | 0 | 23.3 | 24 | 51 | 110 | 351 |
| /risk | 5,856 | 0 | 195.2 | 27 | 55 | 110 | 357 |
| Aggregated | 7,256 | 0 | 241.9 | 25 | 54 | 100 | 357 |

동시 사용자 10 · 30 s · 백엔드 onnx-fp32 · 워커 1 · 요청당 에이전트 5–20, K=20 · 학습 프로세스 2개(4 vCPU 중 2개 점유)와 동시 실행한 값
<!-- LOADTEST_TABLE:END -->

재현: `foresight serve --backend onnx --port 8000` 을 띄운 뒤
`locust -f scripts/locustfile.py --headless -u 10 -r 10 -t 30s --host http://127.0.0.1:8000 --csv results/loadtest/run`,
`python scripts/loadtest_summary.py --users 10 --seconds 30`.
읽는 법: 워커 1개(단일 프로세스)의 상한이다. RPS 를 올리려면 `--workers N`(프로세스 확장)이 맞고, 스레드 확장은 오히려 느리다
(`inference_optimization.md` §3의 OpenMP 스핀 대기 문제).

## 3. 배포와 롤백

| 환경 | 배포 | 롤백 |
|---|---|---|
| 로컬 | `make docker-up` (api + mlflow + minio + redpanda + replay/consumer + prometheus + grafana) | `make docker-down` |
| Render | `render.yaml` 블루프린트 — main 푸시 시 자동 | Render 대시보드에서 이전 배포 "Rollback" |
| AWS | `cd deploy/terraform && terraform apply -var image=ghcr.io/sokldjs554/rtls-foresight:<sha>` | 같은 명령에 이전 `<sha>` — ECS 가 새 태스크 정의로 롤링 교체 |

이미지 태그는 커밋 SHA 를 쓴다(`latest` 는 로컬 편의용). CI 가 main 마다 `ghcr.io/sokldjs554/rtls-foresight:<sha>` 와 `:latest` 를 올린다.
Terraform 은 CI 의 `terraform` 잡이 `fmt -check`·`validate` 를 돌린다(실제 `apply` 는 자격 증명이 있는 사람이 수동으로).

배포 전 확인: `GET /health` 가 `{"status": "ok", "backend": "onnx-fp32", ...}` 를 돌려주고, `/metrics` 에 `foresight_request_latency_seconds` 가 있다.
배포 후 10 분간 대시보드의 p95 와 오류 비율을 본다.

## 4. 모델 교체

1. `foresight train …` → MLflow 레지스트리 `social-stgcnn-<split>` 에 새 버전 등록(`mlflow.register: true`).
2. `foresight evaluate` / `foresight evaluate-rtls` 로 표를 갱신하고 `python tools/check_readme_numbers.py --check` 로 문서 일치 확인.
3. `foresight export --ckpt <best.pth> --out artifacts/onnx` → `artifacts/onnx/manifest.json` 의 패리티(`max_abs_diff`)와 INT8 정확도 차이를 확인.
4. 이미지 빌드(체크포인트·ONNX 는 이미지에 포함) → 3절의 배포. 되돌릴 때는 이전 이미지 태그.

## 5. 장애 대응

| 증상 | 먼저 볼 것 | 조치 |
|---|---|---|
| p95 지연 급증 | 컨테이너 CPU, `OMP_NUM_THREADS`/`torch` 스레드, 요청당 에이전트 수 분포 | 워커 수 증가(`FORESIGHT_WORKERS`), 스레드 1 유지, 100 에이전트 상한 확인 |
| 422 급증 | 클라이언트 페이로드(8 프레임 미만, 좌표 범위, K > 50) | 스키마 한도는 `serving/schemas.py`; 클라이언트 쪽 수정 |
| 5xx | 서버 로그의 트레이스백, 백엔드 파일 존재(`artifacts/onnx/*.onnx`) | `FORESIGHT_BACKEND=torch` 로 임시 전환 후 이미지 재빌드 |
| 경보 폭주 | `foresight_alerts_total` 기울기, 최근 설정 변경(threshold, d_safe), 태그 품질 통계 | 소비자 `AlertPolicy` 의 `min_consecutive`·`cooldown_s` 상향, 임계 복원 |
| 스트림 지연 | Kafka 소비자 lag, `replay --speed`, 구역당 프레임 크기 | 소비자 파티션 추가(구역 키), 배치 크기 조정 |
| 요청 없음 | 소비자 컨테이너 상태, 브로커 연결, 게이트웨이 | 소비자 재시작, `stream --source replay` 로 경로 검증 |

## 6. 데이터·모델 거버넌스

- 위치 데이터는 근로자 개인정보다: 보존 기간·접근 통제·동의 절차를 두고, API 로그에 좌표를 남기지 않는다(요청 ID 만).
- 모델 카드(`model_card.md`)의 한계 — 합성 데이터 시연, d_safe 1 m 에서 지오펜스 우세 — 를 운영 설정 결정(지오펜스 반경 안에서 risk 로 우선순위화)에 반영한다.
