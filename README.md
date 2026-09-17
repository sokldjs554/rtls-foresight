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
(학습 완료 후 `foresight evaluate` 가 채운다)
<!-- REPRODUCTION_TABLE:END -->

세 열을 같이 읽어야 한다. **논문**은 저자가 보고한 값, **공식 ckpt 재평가**는 저자의 가중치를 이 저장소의 평가기로 돌린 값,
**우리 학습**은 처음부터 구현한 코드로 CPU 에서 250 epoch 학습한 값이다. 자세한 논의와 ablation: [`docs/paper_reproduction.md`](docs/paper_reproduction.md)

### 4.2 평가 프로토콜에 따른 차이 · 4.3 대규모 처리 · 4.4 추론 최적화 · 4.5 충돌 경보

(각 절은 해당 문서에서 수치를 동기화한다 — `docs/evaluation_methodology.md`, `docs/data_pipeline.md`, `docs/inference_optimization.md`, `docs/serving_streaming.md`)

## 5. 아키텍처

[`docs/architecture.md`](docs/architecture.md) 에 전체 다이어그램과 모듈 경계가 있다.

## 6. 실행

```bash
make setup                       # venv + CPU torch + 의존성
foresight download && foresight prepare          # ETH/UCY (14 MB, sha256 검증) → 장면 npz
foresight train dataset=eth train=paper seed=0   # 논문 설정 학습 (CPU 1스레드 ~40분) — MLflow sqlite:///mlflow.db
scripts/train_all.sh                             # 5분할 병렬
foresight evaluate                               # 재현표 results/reproduction.json
foresight simulate --profile small && foresight prepare-rtls   # 합성 RTLS → 장면
foresight export && foresight benchmark          # ONNX/INT8 + 벤치마크
foresight serve --backend onnx                   # http://localhost:8000/docs
foresight stream --source replay --replay-file data/processed/rtls/test.npz
docker compose up --build                        # api + mlflow + minio + redpanda + replay/consumer
```

## 7. 실험 기록

[`docs/experiment_log.md`](docs/experiment_log.md) · MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

## 8. 배운 것

- **"재현"의 기준은 논문 본문이 아니라 공식 코드다.** 커널(위치 vs 속도), 축 교환(`permute` vs `view`), 평가(per-agent vs joint)
  세 곳에서 서술과 코드가 달랐고, 수치는 코드에서 나왔다. 코드 동작을 기본값으로, 서술을 ablation 으로 두니 차이가 설명된다.
- **비트 단위 동일성 테스트가 가장 싼 보험이다.** 커널 거리를 float64 로 계산하면 특정 장면에서 라플라시안이 완전히 달라지는데,
  공식 로더와의 동일성 검사가 없었다면 "학습이 조금 다르게 됐나 보다"로 묻혔을 것이다.
- **공식 체크포인트 + 같은 평가기 열이 없으면 학습 결과를 해석할 수 없다.** 분할별로 논문과 다른 방향의 차이가 났지만
  공식 가중치도 같은 방향으로 달랐다 — 우리 학습의 문제가 아니라 평가·데이터 쪽 분산이다.
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
