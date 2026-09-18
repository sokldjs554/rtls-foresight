# CLI 와 모듈 계약

패키지 `foresight` (src/foresight). 진입점 `foresight` (typer). 모든 명령은 `python -m foresight.cli ...` 로도 동작.

| 명령 | 역할 | 주요 옵션 |
|---|---|---|
| `foresight download` | ETH/UCY 원본 다운로드 + sha256 검증 | `--raw-dir data/raw/ethucy` |
| `foresight prepare` | 원본 → 장면 npz (`data/processed/ethucy/<split>/<subset>.npz`) + long parquet | `--raw-dir`, `--out-dir`, `--splits eth,hotel,...` |
| `foresight simulate` | 합성 RTLS 스트림 생성 (파티션 Parquet, 기본 `data/rtls/<profile>/raw`) | `--profile {smoke,small,full}`, `--hours`, `--tags`, `--seed`, `--out-dir` |
| `foresight prepare-rtls` | RTLS Parquet → 품질 필터 → 2.5 Hz 리샘플 → 구역별 장면 npz (`data/processed/rtls/{train,val,test}.npz`) | `--in-dir data/rtls/<profile>/raw`, `--out-dir`, `--train-skip`, `--quality-min`, `--smoothing` |
| `foresight train` | Hydra 학습 (`configs/`), MLflow 로깅 | `dataset=eth train=paper seed=0 ...` (Hydra override) |
| `foresight evaluate` | 체크포인트 평가, 재현표 JSON/MD | `--ckpt-dir results/checkpoints`, `--out results/reproduction.json` |
| `foresight export` | PyTorch → ONNX (+INT8) | `--ckpt`, `--out artifacts/onnx/` |
| `foresight benchmark` | 추론 벤치마크 (eager/compile/ORT/INT8) | `--out results/benchmark.json` |
| `foresight serve` | FastAPI 서버 | `--host --port --backend {torch,onnx,onnx-int8}` |
| `foresight stream` | 스트리밍 소비자 (파일 재생 / Kafka) | `--source {replay,kafka}`, `--sink {stdout,file,kafka}`, `--bootstrap`, `--topic`, `--replay-file`, `--speed`, `--max-seconds` |
| `foresight evaluate-rtls` | RTLS 테스트에서 궤적 지표 + 충돌 경보 품질 | `--ckpts a.pth,b.pth`, `--names`, `--data-dir`, `--every`, `--d-safe`, `--out`, `--collision-out`, `--figure` |

## 데이터 포맷
- **SceneSet npz** (`foresight.data.ethucy.SceneSet`): `pos (A, 20, 2)` float64 절대좌표(m), `scene_index (S, 2)`, `files`, `starts`, `obs_len=8`, `pred_len=12`, `agent_type (A,)` int8 (0 보행자/작업자, 1 차량).
- **그래프**: `foresight.data.graph.scene_to_graph(pos (N,20,2), obs_len) -> V_obs (8,N,2), A_obs (8,N,N), V_pred (12,N,2), A_pred (12,N,N)` float32.
- **모델**: `foresight.models.SocialSTGCNN` — `forward(v (B,2,8,N), a (8,N,N) | (B,8,N,N)) -> (B,5,12,N)`; `predict_params -> (B,12,N,5)` = (μx, μy, log σx, log σy, atanh ρ) 상대 변위 분포. 미래 절대좌표 = `cumsum(μ, t) + pos[:, 7]`.
- **RTLS raw Parquet** 스키마 (10 Hz): `ts_ms int64, tag_id int32, agent_type int8, zone_id int16, x float32, y float32, quality uint8` — `data/rtls/raw/date=YYYY-MM-DD/hour=HH/part-*.parquet`.
- **near-miss 라벨**: 작업자-차량 거리 < `d_safe`(기본 1.0 m) 가 되는 순간을 이벤트로 정의. 평가는 예측 시점 t0 에서 (t0, t0+4.8s] 안에 이벤트가 있는 (worker, vehicle) 쌍을 양성으로 본다.

## 설정 (Hydra, configs/)
`configs/config.yaml` 기본값 + `dataset/{eth,hotel,univ,zara1,zara2,rtls}.yaml`, `train/{paper,fast,smoke}.yaml`, `model/{social_stgcnn}.yaml`. MLflow: `MLFLOW_TRACKING_URI` (기본 `sqlite:///mlflow.db`, ADR-0004).
