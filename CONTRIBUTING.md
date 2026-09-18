# 기여 가이드

## 브랜치·커밋
- `main` 은 항상 CI 초록. 작업은 `feat/…`, `exp/…`, `fix/…` 브랜치에서 PR 로.
- 커밋은 Conventional Commits: `feat(serving): …`, `exp(eth): permute ablation`, `fix(data): float32 kernel`, `docs: …`.
- 수치를 바꾸는 변경은 PR 템플릿의 "수치 변화" 표를 채운다. README/docs 의 수치·표는 `make sync-numbers` 로만 갱신한다(손으로 고치면 CI 가 막는다).

## 개발 환경
```bash
make setup                    # venv + CPU torch + 의존성 + pre-commit
make lint test-fast           # 커밋 전
make download prepare         # 데이터 (14 MB)
```

## 실험 추가
1. Hydra 오버라이드로 먼저 돌려 본다: `foresight train dataset=eth train=paper seed=0 model.graph.kernel=position run_name=exp-poskernel`.
2. MLflow 에 자동으로 남는다 (`make mlflow-ui`). 체크포인트·metrics.json 은 `results/checkpoints/<run_name>/`.
3. 결과가 남을 실험이면 `docs/experiment_log.md` 에 한 줄, 설계 결정이면 `docs/adr/`.
4. 표에 들어갈 수치면 `scripts/collect_results.py` 에 수집 규칙을 추가하고 `make sync-numbers`.

## 테스트 원칙
- 데이터 로더·그래프는 공식 구현과 **비트 단위 동일**해야 한다 (`tests/test_graph.py`). 바꾸면 골든 픽스처를 다시 만들고 이유를 적는다.
- 모델 구조 변경은 공식 체크포인트 로드 테스트를 깨뜨린다 — 의도한 것이면 `time_channel_swap` 같은 옵션으로 분기하고 기본값은 유지한다.

## 이슈로 시작하기
- 버그·기능·실험은 이슈 템플릿(`.github/ISSUE_TEMPLATE/`)으로 연다. 실험은 **가설 → 측정 → 판단 기준**을 먼저 적는다.
- 해결한 뒤에는 이슈에 증상 → 원인 → 조치 → 검증 순으로 남기고 커밋 SHA 를 적는다. 남이 같은 문제를 만났을 때 검색되는 것이 목적이다.

## 리뷰 체크리스트 (PR 템플릿과 같다)
- 수치가 바뀌면 `results/*.json` 과 README/docs 표가 같이 바뀌었는가 (`make check-numbers`, `python scripts/sync_tables.py --check`).
- 새 CLI 옵션·설정 키는 `docs/cli.md` 에 적혔는가. 새 실험은 `docs/experiment_log.md` 에 한 줄이라도 있는가.
- 테스트는 골든 값·동일성·경계값 중 무엇을 지키는지 이름에 드러나는가. CI 가 스킵하는 테스트(데이터 필요)라면 로컬 실행 결과를 PR 에 적는다.
- 브랜치 보호(권장): `main` 은 PR 로만, CI 필수(lint·tests·docs·terraform), 리뷰 1명.

