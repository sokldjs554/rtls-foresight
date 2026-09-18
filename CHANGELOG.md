# Changelog

형식: [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/). 버전은 [SemVer](https://semver.org/lang/ko/).

## [Unreleased]
### Fixed
- `foresight evaluate-rtls` 가 궤적 표를 쓴 뒤 충돌 평가 단계에서 `sample_fraction` 인자 오류로 중단되던 문제.
  부분 샘플(`--every k`) 평가의 오경보/시간 외삽과 "이미 근접해 제외한 쌍" 수를 `results/collision_eval.json` 에 기록한다.
### Added
- 학습 시드 분산 표(`results/seeds.json`, `docs/paper_reproduction.md` §2.1) — 표본 표준편차(ddof=1).
- 충돌 사전 경보의 d_safe 민감도(1.0 m / 2.0 m) 평가와 표·PR 곡선(`results/collision_eval_dsafe2.json`), `evaluate-rtls --figure` 옵션, CI 스모크에 `evaluate-rtls` 단계.
- mypy 0 오류(`src` 전체) — CI 에서 차단 검사로 전환.
- 브라우저 데모(`demo/`): Social-STGCNN 전처리·순전파·샘플링·위험 점수·경보 정책을 순수 JS 로 포팅(`demo/foresight.js`,
  Node 로 PyTorch 와 수치 비교하는 `tests/test_demo_js.py`), 합성 RTLS 구역 60 s 재생 페이지, `foresight demo` 명령, README GIF.

## [0.1.0] - 2026-09-17
### Added
- Social-STGCNN 처음부터 구현, 공식 로더와 비트 단위·공식 체크포인트 출력과 1e-6 동일성 테스트, ETH/UCY 5분할 재현표 (`foresight evaluate`).
- 합성 공장 RTLS 생성기 + Polars/DuckDB 대규모 파이프라인 (85M 행, 시간 파티션 처리, 피크 2.3 GB).
- ONNX export(dynamic N) + 정적 INT8 + 벤치마크, FastAPI `/predict` `/risk`, 충돌 위험 점수·경보 정책, 파일/Kafka 스트림 소비자.
- MLflow(SQLite, 레지스트리), DVC 파이프라인, GitHub Actions CI, Docker/Compose, Terraform(AWS), Render 블루프린트.
- 문서: 논문 재현·평가 방법론·데이터 파이프라인·추론 최적화·서빙 리포트, 모델/데이터 카드, ADR.
