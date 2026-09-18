# 실험 로그

시간순 기록. 수치는 `results/` 의 JSON 이 원본이고, 여기에는 **무엇을 왜 했고 무엇을 배웠는지**를 남긴다.
MLflow 실험 이름: `social-stgcnn` (본 실험), `social-stgcnn-ablation`, `social-stgcnn-rtls`.

## E0 · 공식 구현 해부와 동일성 고정
- 공식 저장소(`abduallahmohamed/Social-STGCNN`)의 데이터 로더·모델·평가 코드를 읽고, 논문 서술과 다른 세 지점을 찾았다
  (ADR-0002: 속도 커널, `view` 축 교환, per-agent best-of-20).
- 장면 윈도우와 그래프 텐서를 numpy 브로드캐스트로 다시 구현하고 공식 로더 결과와 **비트 단위로 같음**을 확인했다
  (eth/test 70, zara1/val 605, hotel/test 301, univ/test 947 장면 전부 일치).
  - 데이터 준비 시간: 공식 ETH 학습 분할 **211 s** → **0.95 s** (파이썬 이중 루프 + networkx → 벡터화).
  - 함정 하나: 커널 거리는 **float32** 로 빼야 한다. 두 보행자의 변위가 float32 에서 정확히 같은 장면(ETH test 52번)이 있는데,
    float64 로 계산하면 1e-17 차이가 1e17 가중치가 되어 라플라시안이 완전히 달라진다.
- 모델을 처음부터 구현하고 공식 체크포인트를 로드해 출력이 **비트 단위로 같음**을 테스트로 고정했다 (`tests/test_model.py`).
  `time_channel_swap=view` 일 때만 같고, 논문 그림대로 `permute` 하면 다른 함수가 된다.

## E1 · 공식 체크포인트 재평가 (평가기 검증)
- 공식 ETH 체크포인트를 공식 평가 코드로 CPU 에서 돌리면 ADE/FDE **0.73 / 1.22** (시드 0). 논문 표는 0.64 / 1.11.
  같은 체크포인트·같은 코드인데 차이가 난다 — best-of-20 샘플링의 분산과 ETH 테스트의 작은 크기(181명)가 원인 후보다.
  5개 분할 × 3 시드 재평가 결과는 `results/reproduction.json` 의 `official_checkpoint` 열에 있다.
- 교훈: "논문 수치와 다르다"는 말을 하기 전에 **공식 체크포인트 + 같은 평가기** 열을 먼저 만들어야 한다. 그래야 학습의 문제와 평가의 문제를 가를 수 있다.

## E2 · 학습 파이프라인 스모크
- `train=smoke` (2 epoch, 64 장면): 학습→검증→best-of-20 평가→MLflow run·레지스트리 등록까지 9 초. CI 가 매번 이 경로를 돈다.
- MLflow 3 은 파일 스토어를 거부한다 → SQLite 백엔드로 전환 (ADR-0004). `mlflow.pytorch.log_model` 기본 직렬화(pt2)는
  입력 예시를 요구해 `pickle` 로 지정했다.

## E3 · 논문 설정 재현 (5 분할, 250 epoch, CPU)
- 설정: `train=paper` (SGD 0.01, 128 장면 누적, StepLR 150/0.2), 분할당 1 스레드, 3 프로세스 병렬.
- 결과와 논의: `docs/paper_reproduction.md`.

## E4 · 시드 분산
- 재현표의 ± 는 **평가(샘플링) 시드 3개**의 표준편차다. **학습 시드** 1·2 는 별도 실행(`SEEDS="1 2" scripts/train_all.sh`)이며
  결과는 `results/seeds.json` 과 `docs/paper_reproduction.md` 의 시드 표에 넣는다.

## E5 · Ablation (eth)
- `permute`(논문 그림) vs `view`(공식 코드), 위치 커널 vs 속도 커널, 버킷 배치(BN 통계 배치 단위) vs 장면 배치, 로그 영역 NLL vs 클램프 NLL.

## E6 · 합성 RTLS 전이·미세조정, 충돌 경보 평가
- ETH/UCY 학습 모델을 RTLS 테스트에 그대로 적용(zero-shot) → 미세조정(`train=finetune`) → CVM 과 비교.
- 경보 품질: 정밀도/재현율/선행시간, 지오펜스 규칙과 비교. `results/collision_eval.json`.

## E7 · 추론 최적화
- `docs/inference_optimization.md`. 요점: ONNX Runtime fp32 가 eager 대비 2배 빠르고, **정적 INT8 은 이 7.6K 모델에서 오히려 느리다**
  (Q/DQ 오버헤드 > 연산 절감). TXP-CNN 만 양자화하면 정확도 손실이 절반(ADE +0.038)이지만 속도 이득은 없다. 전처리 벡터화 ×100.
  토치 4 스레드는 OpenMP spin-wait 로 100배 느려짐 → 서빙은 1 스레드 + 프로세스 확장.

## E8 · 대규모 RTLS 파이프라인 — OOM 과 시간 파티션
- 85M 행 full 프로파일에서 단일 lazy 쿼리(`unique`+`group_by`+PartitionBy sink)가 10.9 GB 로 OOM. 시간 파티션 단위로 바꿔
  결과 동일(small 장면 수 일치) · 91.5 s · 피크 2.29 GB. 상세: `docs/data_pipeline.md` §4.
