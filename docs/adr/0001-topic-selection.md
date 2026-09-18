# ADR-0001 · 주제 선정: RTLS 궤적 예측 기반 충돌 사전 경보

- 상태: 채택 (2026-09-17)

## 맥락
지원 공고(데이터플로, 인공지능 AI 모델 개발자)의 주요업무는 모델 설계·학습, 전처리 파이프라인, 평가·개선, 추론 최적화,
실험 문서화이고 우대사항은 PyTorch, MLOps, 클라우드, 대규모 데이터, 논문 구현·재현이다.
회사의 자체 제품 TPAM 은 UWB RTLS 기반 스마트 안전 플랫폼(출입통제·**충돌 방지**·동선 추적·이상징후 탐지)이며,
Confluent(Kafka)·Databricks·Verta 파트너다.

국내 포트폴리오 지형(부트캠프 파이널·취업 포트폴리오·데이콘·GitHub 한국어 README, 6개 각도 200+ 주제 집계)에서
궤적 예측과 작업자-장비 충돌 위험 예측은 사실상 0건이었다. 반면 감성분석·YOLO·RAG 챗봇·추천·집값·이탈 예측은 포화 상태였다.
지원자의 기존 저장소(로그 이상탐지, RUL, 웨이퍼/PCB 비전, NPU 경량화, PPE 검출, Text-to-SQL, RAG)와도 겹치지 않는다.

## 결정
Social-STGCNN(CVPR 2020)을 처음부터 구현해 ETH/UCY 5개 분할의 ADE/FDE 를 재현하고, 합성 공장 RTLS 스트림으로 확장해
충돌 위험 점수·경보까지 서비스 형태로 만든다.

## 대안과 기각 이유
- **다변량 센서 이상탐지(USAD/Anomaly Transformer on SMD)** — 회사 정합성은 높지만 포트폴리오에서 드물지 않고 지원자가 이미 이상탐지 저장소를 갖고 있다.
- **학습형 인덱스 / 카디널리티 추정(데이터 시스템 ML)** — 희소하지만 "AI 모델 개발자" 공고에서 모델 학습 비중이 얇다.
- **과학 ML(FNO/PhaseNet/중력파)** — 희소하지만 회사 도메인과 무관하고 합성 데이터만으로는 "데이터 분석·전처리 경험"을 보이기 어렵다.

## 결과
공고의 모든 항목이 구체적 산출물로 대응된다(README 표 참조). 위험: ETH/UCY 는 보행자 데이터라 공장 RTLS 와 분포가 다르다 —
합성 RTLS 전이 실험은 "파이프라인 시연"이지 검증된 안전 성능이 아님을 문서에 명시한다.

## 재검증 (2026-09-18)
저장소를 공개한 뒤 같은 질문을 다시 확인했다: "이 주제가 흔한가?"

- GitHub 에서 Social-STGCNN 은 공식 구현과 그 포크(STMGCN 등)만 검색되고, 이를 **처음부터 재구현해 공식 체크포인트와 출력을 맞춘
  포트폴리오**나 **RTLS/UWB 충돌 사전 경보로 확장한 저장소**는 찾지 못했다
  ([검색 1](https://github.com/abduallahmohamed/Social-STGCNN), [검색 2](https://github.com/topics/uwb-positioning?o=desc&s=stars)).
- UWB RTLS + 지게차·작업자 충돌 예측은 상용 제품과 특허(예: [Trio Mobil](https://www.triomobil.com/en/blog/forklift-collision-avoidance-tech-in-2026),
  [Ubiquicom Proximity Plus](https://www.ubiquicom.com/en/proximity-plus/), [Dmatek UWB-FAS](https://www.dmatektw.com/product/130))의 영역이고,
  공개 코드로 궤적 **분포** 예측과 경보 정책까지 잇는 프로젝트는 검색되지 않았다.
- 회사의 UWB RTLS 사례 페이지([data-flow.co.kr/uwbcase](https://data-flow.co.kr/uwbcase))가 확인되어 도메인 정합성도 유지된다.
- 국내 취업 포트폴리오 가이드류는 "기획부터 배포까지"의 실전형 사이드 프로젝트를 권하고, 주제로는 여전히 챗봇·RAG·이미지 분류·추천이
  주류다. 따라서 주제를 바꿀 이유는 없고, 대신 "희소하지만 공고와 무관"이 되지 않도록 공고 항목별 근거를 README 의 역량 지도에 둔다.
