# 데이터 카드

## 1. ETH/UCY 보행자 궤적 (재현용)
| 항목 | 내용 |
|---|---|
| 출처 | ETH (Pellegrini et al., ICCV 2009: eth, hotel), UCY (Lerner et al., CGF 2007: univ, zara1, zara2) |
| 획득 경로 | Social-STGCNN 공식 저장소가 vendoring 한 Social-GAN 포맷 파일 (raw.githubusercontent.com), 74개 파일 14 MB, sha256 매니페스트 고정 (`src/foresight/data/manifests/ethucy.json`) |
| 포맷 | `frame_id  ped_id  x  y` (탭, 미터, 2.5 fps) |
| 분할 | leave-one-out: 테스트 분할 하나를 뺀 나머지 장면들이 학습/검증 |
| 전처리 | 20 프레임 슬라이딩 윈도우(stride 1), 20 프레임 연속 등장 보행자만, 2명 이상 장면만, 좌표 4자리 반올림 — 공식 로더와 장면 집합·텐서가 비트 단위로 동일 (`tests/test_graph.py`) |
| 규모 | 학습 2,076~2,785 장면(분할별), 테스트 70(eth)~947(univ) 장면; EDA 는 `notebooks/01_eda_ethucy.ipynb` |
| 라이선스 | 원 데이터: 연구용 공개 (원 논문 인용). vendored 복사본과 공식 체크포인트의 출처 저장소 라이선스 원문은 `assets/official_checkpoints/LICENSE.social-stgcnn` |
| 알려진 문제 | 테스트 분할 크기 편차가 큼(ETH 181명); 픽셀→미터 호모그래피 변환 오차; 프레임 결손은 없음 |

## 2. 합성 공장 RTLS 스트림 (대규모 처리·전이·충돌 평가용)
| 항목 | 내용 |
|---|---|
| 생성기 | `foresight simulate` (`src/foresight/data/rtls_sim.py`) — 공장 레이아웃, 작업자/지게차 이동 모델, UWB 노이즈(σ 0.15 m)·이상치·드롭아웃, near-miss 주입 |
| 포맷 | 10 Hz, `ts_ms, tag_id, agent_type, zone_id, x, y, quality`, 파티션 Parquet (`date=/hour=`), 5분 청크 |
| 라벨 | 노이즈 없는 진짜 위치로 계산한 작업자-차량 거리 < d_safe 이벤트 (`_manifest.json`, `_truth/`) |
| 규모 | 프로파일 smoke/small/full — 측정값은 `docs/data_pipeline.md` |
| 전처리 | 품질 필터 → 중복 제거 → 2.5 Hz 리샘플 → 구역별 20 프레임 윈도우 → SceneSet (시간 기준 train/val/test, 경계 윈도우 제거) |
| 한계 | **합성 데이터다.** 이동 모델은 단순화되어 있고 실제 현장의 레이아웃·행동·센서 특성을 대표하지 않는다. 결과는 파이프라인과 평가 절차의 시연이다. |
