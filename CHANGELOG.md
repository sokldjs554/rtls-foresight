# Changelog

형식: [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/). 버전은 [SemVer](https://semver.org/lang/ko/).

## [Unreleased]

## [0.1.0] - 2026-09-17
### Added
- Social-STGCNN 처음부터 구현, 공식 로더·체크포인트와 비트 단위 동일성 테스트, ETH/UCY 5분할 재현표 (`foresight evaluate`).
- 합성 공장 RTLS 생성기 + Polars/DuckDB 대규모 파이프라인 (85M 행, 시간 파티션 처리, 피크 2.3 GB).
- ONNX export(dynamic N) + 정적 INT8 + 벤치마크, FastAPI `/predict` `/risk`, 충돌 위험 점수·경보 정책, 파일/Kafka 스트림 소비자.
- MLflow(SQLite, 레지스트리), DVC 파이프라인, GitHub Actions CI, Docker/Compose, Terraform(AWS), Render 블루프린트.
- 문서: 논문 재현·평가 방법론·데이터 파이프라인·추론 최적화·서빙 리포트, 모델/데이터 카드, ADR.
