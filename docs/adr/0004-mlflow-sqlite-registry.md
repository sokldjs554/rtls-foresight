# ADR-0004 · MLflow 는 SQLite 백엔드 + 모델 레지스트리

- 상태: 채택

## 맥락
MLflow 3 은 파일 스토어(`./mlruns`)를 유지보수 모드로 두고 기본적으로 예외를 던진다. 파일 스토어에는 모델 레지스트리도 없다.

## 결정
기본 `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`. 각 학습 run 은 파라미터(Hydra 설정 평탄화)·epoch 별 손실/lr/시간·최종
ADE/FDE(세 프로토콜 + CVM)·체크포인트·history.json 을 남기고, 최적 모델을 pytorch flavor 로 로그해
`social-stgcnn-<split>` 이름으로 레지스트리에 버전 등록한다. Docker Compose 에서는 MLflow 서버(+MinIO 아티팩트)로 같은 코드가 동작한다.

## 결과
`mlflow ui --backend-store-uri sqlite:///mlflow.db` 로 모든 실험을 비교할 수 있고, 서빙은 레지스트리 버전을 참조할 수 있다.
