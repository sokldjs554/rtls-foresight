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

## E5 · Ablation (eth, 250 epoch)
- 기본 0.743/1.317 · permute 0.815/1.623 · 위치 커널 0.771/1.348 · 버킷 배치 0.805/1.606 (학습 7 분) · 로그 영역 NLL **0.736/1.263**.
- 결론: 공식 코드의 `view`·속도 커널·장면 단위 BN 이 모두 결과에 기여한다. 손실 클램프만은 없애는 편이 약간 낫다. 상세: `docs/paper_reproduction.md` §3.

## E6 · 합성 RTLS 전이·미세조정, 충돌 경보 평가
- 데이터: full 프로파일(200 태그 × 12 h) → 학습 장면 앞 60,000개(skip 4), 테스트는 1.8 h 스트림의 장면을 10개마다 하나(`--every 10`).
- 궤적 지표(best-of-20 / 결정적, CVM 1.333/2.611): zero-shot(eth 가중치) 0.735/1.288 · 1.046/2.013 →
  **미세조정 30 epoch 0.615/0.978 · 0.905/1.750**, **RTLS 처음부터(버킷 배치, 60 epoch) 0.584/0.924 · 0.834/1.644**
  (`results/rtls_transfer.json`; 평가 시드 3개 평균).
  보행자 가중치는 지게차(최대 3 m/s, 큰 회전반경)를 모른다 — RTLS 장면으로 적응하면 결정적 예측도 CVM 을 크게 이긴다
  (ETH/UCY 에서는 반대였다: 등속에 가까운 보행자에게는 μ 가 CVM 보다 나빴다).
- 경보 품질(`results/collision_eval.json`, d_safe 1.0 m, 양성 260/395,781 = 0.07 %): **지오펜스(현재 거리)가 AP 0.084 · AUROC 0.914 로 최고**,
  RTLS 학습 모델의 분포 기반 위험은 AUROC 0.903 (AP 0.026), zero-shot 0.815, CVM-S 0.797, 결정적(μ) 변형은 0.65–0.67.
  예측 오차(ADE 0.58 m)가 d_safe 와 같은 크기라 "현재 거리" 이상의 정보를 주지 못한다 — 예상과 다른 결과지만 그대로 싣는다 (README §4.5).
- d_safe 2.0 m 민감도(`results/collision_eval_dsafe2.json`, 양성 3.55 %): 순서가 뒤집힌다 — 학습 모델 AP 0.465 · F1 0.508 vs 지오펜스 0.341 · 0.383,
  비슷한 오경보율에서 재현율 0.44 vs 0.35, 선행시간 1.64 vs 1.27 s. 예측 모델이 값을 내는 조건은 "예측 오차 ≪ 안전 거리" 라는 것을 두 설정이 같이 보여 준다.
- 첫 실행은 궤적 표를 쓴 뒤 충돌 단계에서 죽었다: CLI 는 `sample_fraction` 을 넘기는데 `evaluate_collision` 시그니처에 없었다.
  단위 테스트는 함수만, CI 스모크는 `evaluate-rtls` 를 안 돌려서 둘 다 놓쳤다. 인자를 구현(오경보/시간 외삽, 제외 쌍 수 기록)하고
  테스트와 CI 스모크 단계를 추가했다.

## E7 · 추론 최적화
- `docs/inference_optimization.md`. 요점: ONNX Runtime fp32 가 eager 대비 2배 빠르고, **정적 INT8 은 이 7.6K 모델에서 오히려 느리다**
  (Q/DQ 오버헤드 > 연산 절감). TXP-CNN 만 양자화하면 정확도 손실이 절반(ADE +0.038)이지만 속도 이득은 없다. 전처리 벡터화 ×100.
  토치 4 스레드는 OpenMP spin-wait 로 100배 느려짐 → 서빙은 1 스레드 + 프로세스 확장.

## E8 · 대규모 RTLS 파이프라인 — OOM 과 시간 파티션
- 85M 행 full 프로파일에서 단일 lazy 쿼리(`unique`+`group_by`+PartitionBy sink)가 10.9 GB 로 OOM. 시간 파티션 단위로 바꿔
  결과 동일(small 장면 수 일치) · 91.5 s · 피크 2.29 GB. 상세: `docs/data_pipeline.md` §4.
