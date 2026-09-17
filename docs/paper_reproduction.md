# 논문 재현 리포트 — Social-STGCNN (CVPR 2020)

재현 대상: Mohamed et al., *Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction*,
CVPR 2020, **Table 1** (ETH/UCY 5개 분할, 관측 8 프레임 → 예측 12 프레임, ADE/FDE (m), best-of-20).

## 0. 결론 먼저

<!-- REPRODUCTION_TABLE:START -->
(`foresight evaluate` 가 채운다)
<!-- REPRODUCTION_TABLE:END -->

- **평균은 재현된다.** 저자의 공식 체크포인트를 이 저장소의 평가기로 다시 돌리면 5분할 평균이 논문과 0.01 안에서 같다.
  같은 코드를 CPU 에서 250 epoch 학습한 우리 모델은 평균이 논문보다 약간 높다(분할별 차이는 아래).
- **분할별로는 다르다.** 공식 체크포인트조차 hotel 은 논문보다 좋고(0.41/0.69 vs 0.49/0.85), univ 는 나쁘다(0.49/0.91 vs 0.44/0.79).
  논문의 분할별 수치가 어떤 실행에서 나왔는지는 재현할 수 없고, "평균이 맞는다"가 정직한 표현이다.
- **ETH 는 가장 어렵고 가장 불안정하다.** 테스트 보행자가 181명뿐이라 best-of-20 표준편차가 크고, 공식 체크포인트도 0.73/1.21 이다.
- **등속 모델(CVM)을 이기는 것은 분포뿐이다.** 결정적 평균(μ)만 쓰면 CVM 보다 나쁘다(대부분의 분할). 20 샘플 CVM-S 는 hotel·zara2 에서 학습 모델보다도 좋다.
  이는 Schöller et al. (RA-L 2020) 의 결론과 같고, 충돌 위험 점수를 **평균이 아니라 샘플 분포**로 계산하는 이유다.

## 1. 무엇을 그대로 했고 무엇을 바꿨나

| 항목 | 공식 코드 | 이 저장소 | 근거 |
|---|---|---|---|
| 데이터·분할 | Social-GAN 포맷, leave-one-out | 동일 파일(sha256 고정), 장면 집합 비트 동일 | `tests/test_graph.py`, `docs/experiment_log.md` E0 |
| 그래프 커널 | 상대 변위 간 1/거리 (논문 본문은 위치) | 동일 (기본값), 위치 커널은 ablation | ADR-0002 |
| (C,T) 축 교환 | `view` (논문 그림은 permute) | 동일 (기본값), permute 는 ablation | ADR-0002 |
| 손실 | 이변량 NLL, pdf 클램프 1e-20 | 동일 (기본값), 로그 영역 NLL 은 ablation | `models/losses.py` |
| 학습 | SGD 0.01, 250 ep, StepLR 150/0.2, 장면 128 누적, val 최소 체크포인트 | 동일. 자투리 배치는 실제 개수로 나눔(공식은 128 고정) | `train/train.py` |
| 데이터 준비 | 파이썬 루프 + networkx (ETH 학습 211 s) | numpy 벡터화 (0.95 s), 결과 동일 | E0 |
| 평가 | per-agent best-of-20, 시드 1개 | 동일 프로토콜 + joint + 결정적, 시드 3개 평균±표준편차 | `eval/metrics.py` |
| 하드웨어 | GPU | CPU 4코어, 분할당 1 스레드 프로세스 | ADR-0003 |

## 2. 학습 비용 (CPU)

| 분할 | 학습 장면 | epoch 시간 (3 프로세스 동시) | 250 epoch | 최적 epoch |
|---|---|---|---|---|
<!-- TRAIN_COST:START -->
(학습 로그 `results/checkpoints/<split>/seed0/history.json` 에서 채운다)
<!-- TRAIN_COST:END -->

## 3. Ablation — 공식 코드의 세 가지 특이점은 결과에 영향을 주는가 (eth)

<!-- ABLATION_TABLE:START -->
(`scripts/run_ablations.sh` 후 `python scripts/collect_ablation.py` 가 채운다)
<!-- ABLATION_TABLE:END -->

## 4. 재현 과정에서 배운 것

1. **"재현"의 기준을 먼저 고정하라.** 논문 본문·그림·코드가 서로 다를 때 수치는 코드에서 나온다. 코드 동작을 기본값으로 두고 논문 서술을
   ablation 으로 두면, 차이가 "구현 실수"인지 "저자의 선택"인지 분리된다.
2. **공식 체크포인트 + 같은 평가기 열이 없으면 학습 결과를 해석할 수 없다.** 이 열이 있었기에 hotel/univ 의 차이가 우리 학습이 아니라
   평가·데이터 쪽에 있음을 알 수 있었다.
3. **best-of-20 은 오라클이다.** 서비스에서 쓰는 수치는 결정적 μ 또는 경보 품질이고, 그 기준에서는 CVM 이 강력한 경쟁자다.
   모델의 가치는 "얼마나 정확한 한 점"이 아니라 "얼마나 잘 보정된 분포"에 있다.
4. **float32 와 float64 의 경계가 그래프를 바꾼다.** 커널 거리를 float64 로 계산하면 특정 장면에서 라플라시안이 완전히 달라진다.
   비트 동일성 테스트가 없었다면 이 차이는 "학습이 조금 다르게 됐나 보다"로 묻혔을 것이다.
5. **CPU 로도 충분하다 — 병목은 파이썬 루프였다.** 7.6K 파라미터 모델의 학습 비용은 연산이 아니라 데이터 준비와 per-scene 루프에 있었다.
