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
