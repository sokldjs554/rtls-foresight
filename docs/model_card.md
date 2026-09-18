# 모델 카드 — Social-STGCNN (rtls-foresight 재현본)

## 모델 개요
| 항목 | 내용 |
|---|---|
| 구조 | ST-GCNN 1층 (그래프 합성곱 + 시간 합성곱, 채널 2→5) + TXP-CNN 5층 (시간축 외삽 8→12) |
| 파라미터 | 7,563 (논문 7.6K) |
| 입력 | 장면 내 N 명의 최근 8 프레임(3.2 s, 2.5 Hz) 상대 변위 + 프레임별 정규화 라플라시안 (N×N) |
| 출력 | 12 프레임(4.8 s) 상대 변위의 이변량 가우시안 파라미터 (μx, μy, log σx, log σy, atanh ρ) |
| 손실 | 이변량 가우시안 NLL (공식 코드와 같은 pdf 클램프 경로) |
| 학습 | SGD lr 0.01, 250 epoch, StepLR(150, ×0.2), 장면 128개 누적, 검증 손실 최소 체크포인트 |
| 원 논문 | Mohamed, Qian, Elhoseiny, Claudel. *Social-STGCNN*, CVPR 2020 — 공식 체크포인트 5개를 `assets/official_checkpoints/` 에 원 저장소 라이선스(`LICENSE.social-stgcnn`)와 함께 vendoring |
| 구현 | `src/foresight/models/social_stgcnn.py` — 공식 코드의 `view` 축 교환·속도 커널을 기본값으로 재현 (ADR-0002) |

## 학습 데이터
ETH(eth, hotel) + UCY(univ, zara1, zara2) 보행자 궤적, Social-GAN leave-one-out 분할(학습 4 장면 → 테스트 1 장면). 좌표는 미터,
장면 = 20 프레임 안에 2명 이상이 연속 등장하는 윈도우. 데이터 카드(`docs/data_card.md`) 참조.

## 성능
`docs/paper_reproduction.md` 의 재현표가 유일한 출처다. README 의 수치는 `results/reproduction.json` 에서 자동 동기화된다.
세 프로토콜(per-agent best-of-20 / joint best-of-20 / 결정적 μ)과 CVM 기준선을 항상 함께 읽는다 (`docs/evaluation_methodology.md`).

## 의도된 사용
- 2.5 Hz 위치 스트림에서 4.8 s 뒤 위치 **분포**를 예측해 작업자-차량 충돌 **위험 점수**를 만드는 용도 (`foresight.serving.risk`).
- 서비스 결정은 위험 점수 + 경보 정책(임계값·연속 프레임·쿨다운)이 내리며, 모델 단독 출력은 결정이 아니다.

## 의도되지 않은 사용 / 한계
- 보행자 데이터로 학습했다. 지게차·AGV 의 운동학(회전 반경, 가속)은 학습 분포에 없다. 합성 RTLS 미세조정은 시연이지 검증이 아니다.
- 관측 8 프레임이 모두 있어야 예측한다. UWB 드롭아웃이 길면(>1 프레임) 해당 태그는 그 프레임에서 제외된다.
- 좌표계·프레임 주기(0.4 s)가 다르면 재학습이 필요하다. 미터 단위, 2.5 Hz 리샘플이 전제다.
- best-of-20 지표는 오라클 선택이다. 서비스 지표는 결정적 μ 와 경보 품질(정밀도/재현율/선행시간)을 본다.
- 합성 RTLS 충돌 사전 경보에서 학습 모델의 위험 점수는 d_safe 1.0 m 에서는 "현재 거리" 지오펜스를 넘지 못했고
  (AUROC 0.903 vs 0.914, AP 0.026 vs 0.084), 2.0 m 에서는 넘었다(AP 0.465 vs 0.341). 예측 오차(ADE ≈ 0.6 m)가 안전 거리보다
  충분히 작은 스케일에서만 쓰고, 지오펜스를 대체하기보다 그 안에서 우선순위를 매기는 신호로 쓴다.
- 안전 시스템의 단독 근거로 쓰면 안 된다. 실제 배치 전에는 현장 데이터로 재평가·재보정이 필요하다.

## 윤리·개인정보
ETH/UCY 는 공개 연구용 데이터(익명 궤적). 합성 RTLS 는 실제 사람의 데이터가 아니다. 실제 배치 시 위치 데이터는 근로자 개인정보이므로
보존 기간·접근 통제·동의 절차가 필요하다.

## 버전
MLflow 레지스트리 `social-stgcnn-<split>` 버전 = 학습 run. 체크포인트에는 Hydra 설정이 동봉된다 (`torch.load(...)["model"]`).
