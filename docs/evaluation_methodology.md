# 평가 방법론 — best-of-20 이 감추는 것

궤적 예측 논문의 표준 지표는 **ADE/FDE (best-of-20)** 다. 20개 샘플을 뽑아 정답에 가장 가까운 것을 채점한다.
이 프로젝트는 같은 모델을 세 가지 프로토콜로 평가해 숫자가 어떻게 달라지는지 보인다.

| 프로토콜 | 정의 | 누가 쓰나 |
|---|---|---|
| **per-agent best-of-20** | 보행자마다 20개 중 최적 샘플 (서로 다른 보행자가 다른 샘플을 골라도 됨) | Social-STGCNN 공식 코드, 논문 표 |
| **joint best-of-20** | 장면(모든 보행자) 단위로 하나의 샘플 선택 | 다중 에이전트 상호작용을 평가하려면 이쪽 |
| **deterministic (μ)** | 샘플 없이 분포 평균 | 서비스에서 "한 번의 예측"을 보여줄 때 |

여기에 **CVM(등속 모델)** 을 두 가지로 둔다: 결정적 CVM, 그리고 방향을 σ=25° 로 흔든 20 샘플 CVM-S (Schöller et al., RA-L 2020).

## 왜 중요한가
- best-of-20 은 **오라클 선택**이다. 실제 서비스는 정답을 모르므로 20개 중 무엇을 쓸지 고를 수 없다.
- per-agent 선택은 joint 보다 항상 좋거나 같다(테스트 `test_per_agent_best_is_never_worse_than_joint`). 논문 표의 수치는 그중 가장 관대한 쪽이다.
- 충돌 위험은 **하나의 미래 안에서** 두 에이전트의 위치를 같이 봐야 한다. 그래서 위험 점수는 joint 샘플(k 번째 샘플끼리)로 계산한다.

## 이 프로젝트의 수치
재현표(`docs/paper_reproduction.md`)에 세 프로토콜과 CVM 을 나란히 둔다. README 수치는 `results/reproduction.json` 에서
자동으로 동기화된다(`tools/check_readme_numbers.py`).

## 충돌 경보의 평가
궤적 오차가 아니라 **경보 품질**로 평가한다 (`docs/serving_streaming.md`, `results/collision_eval.json`):
- 양성 정의: 예측 시점 t0 이후 4.8 s 안에 작업자-차량 거리 < d_safe(1.0 m) 인 (작업자, 차량) 쌍.
- 지표: 임계값별 정밀도/재현율/F1, PR 곡선(AUPRC), 경보 선행시간(lead time) 분포, 시간당 오경보 수.
- 기준선: "현재 거리 < r" 규칙(대부분의 RTLS 제품이 쓰는 지오펜스), CVM 기반 위험 점수.
