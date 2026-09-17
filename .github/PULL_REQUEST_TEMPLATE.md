## 변경 요약

<!-- 무엇을, 왜. 실험이면 가설과 결론 한 줄. -->

## 재현 명령

```bash
# 예: foresight train dataset=eth train=paper seed=0 && foresight evaluate
```

## 수치 변화

| 지표 | 이전 | 이후 | 출처 (results/*.json, MLflow run) |
|---|---|---|---|
| | | | |

## 체크리스트

- [ ] `make lint test-fast` 통과
- [ ] 수치를 바꿨다면 `make sync-numbers` 로 README/docs 갱신 (CI 의 `check-numbers` 가 확인한다)
- [ ] 새 실험은 `docs/experiment_log.md` 에 한 줄 이상
- [ ] 설계 결정은 `docs/adr/` 에 기록
