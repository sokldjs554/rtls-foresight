# =====================================================================
# rtls-foresight — 개발/실행 진입점
#   make setup      : venv + CPU torch + 의존성
#   make reproduce  : 데이터 → 5분할 학습 → 재현표 (CPU 4코어 기준 약 2시간)
#   make serve      : ONNX 백엔드 FastAPI 를 30초 안에 띄운다
# =====================================================================
SHELL      := /bin/bash
PY         ?= python3
VENV       ?= .venv
BIN        := $(VENV)/bin
PYTHON     := $(BIN)/python
FORESIGHT  := $(BIN)/foresight
export FORESIGHT_ROOT := $(CURDIR)
export MLFLOW_DISABLE_TELEMETRY := true
export MLFLOW_DISABLE_AGENT_HINT := 1
export OMP_WAIT_POLICY := PASSIVE    # torch 스레드가 남는 코어를 스핀 대기로 태우지 않게 (docs/inference_optimization.md §3)

SEEDS      ?= 0
PAR        ?= 3
PROFILE    ?= small
BACKEND    ?= onnx
HOST       ?= 0.0.0.0
PORT       ?= 8000

.DEFAULT_GOAL := help
.PHONY: help setup download prepare simulate prepare-rtls train train-all evaluate ablations rtls-transfer export benchmark \
        serve stream test test-fast lint fmt typecheck docs figures sync-numbers check-numbers \
        docker-build docker-up docker-down dvc-repro mlflow-ui clean distclean

help: ## 사용 가능한 타깃 목록
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = "## "}; {sub(/:.*/, "", $$1); printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------- 환경
$(VENV)/bin/activate:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

setup: $(VENV)/bin/activate ## venv 생성 + CPU torch + 전체 의존성 (GPU 가 있어도 CPU 휠로 충분하다: 7.6K 파라미터)
	$(BIN)/pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(BIN)/pip install -e ".[all]"
	$(BIN)/pre-commit install || true

# --------------------------------------------------------------------- 데이터
download: ## ETH/UCY 원본 다운로드 + sha256 검증 (14 MB)
	$(FORESIGHT) download

prepare: ## 원본 → 장면 npz (분할 5개, 1초 내외)
	$(FORESIGHT) prepare

simulate: ## 합성 RTLS 스트림 생성 (PROFILE=smoke|small|full)
	$(FORESIGHT) simulate --profile $(PROFILE)

prepare-rtls: ## RTLS Parquet → 2.5 Hz 장면 npz (PROFILE 과 짝)
	$(FORESIGHT) prepare-rtls --in-dir data/rtls/$(PROFILE)/raw --out-dir data/processed/rtls

# --------------------------------------------------------------------- 학습·평가
train: ## 단일 학습 (예: make train ARGS="dataset=eth train=paper seed=0")
	$(FORESIGHT) train $(ARGS)

train-all: ## 5분할 논문 설정 학습 (SEEDS="0 1 2" PAR=3)
	SEEDS="$(SEEDS)" PAR=$(PAR) scripts/train_all.sh

evaluate: ## 재현표 results/reproduction.json (+ 공식 체크포인트 재평가)
	$(FORESIGHT) evaluate
	$(PYTHON) scripts/collect_results.py
	$(PYTHON) scripts/sync_tables.py

ablations: ## eth ablation 4종 (view/permute, 커널, 버킷 배치, 손실)
	scripts/run_ablations.sh

rtls-transfer: ## RTLS 전이: 미세조정 + 처음부터 + 충돌 경보 평가
	scripts/run_rtls_transfer.sh

reproduce: download prepare train-all evaluate ## 처음부터 재현표까지

# --------------------------------------------------------------------- 추론·서빙
export: ## PyTorch → ONNX (+ 정적 INT8)
	$(FORESIGHT) export

benchmark: ## 추론 벤치마크 (QUICK=1 이면 CI 용 45초)
	$(FORESIGHT) benchmark $(if $(QUICK),--quick,)

serve: ## FastAPI 서버 (BACKEND=onnx|onnx-int8|torch)
	$(FORESIGHT) serve --host $(HOST) --port $(PORT) --backend $(BACKEND)

stream: ## 파일 재생 스트리밍 소비자 (RTLS 테스트 장면을 10배속으로)
	$(FORESIGHT) stream --source replay --replay-file data/processed/rtls/test.npz --speed 10 --max-seconds 60

mlflow-ui: ## MLflow UI (sqlite:///mlflow.db)
	$(BIN)/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# --------------------------------------------------------------------- 품질
test: ## 전체 테스트
	$(PYTHON) -m pytest -q -p no:warnings

test-fast: ## 느린·Docker 테스트 제외
	$(PYTHON) -m pytest -q -p no:warnings -m "not slow and not docker" --ignore=tests/test_stream.py

lint: ## ruff check
	$(BIN)/ruff check src tests scripts tools

fmt: ## ruff format + import 정렬
	$(BIN)/ruff format src tests scripts tools && $(BIN)/ruff check --fix src tests scripts tools

typecheck: ## mypy (src)
	$(BIN)/mypy src/foresight

check-numbers: ## README/docs 의 수치·표가 results/*.json 과 같은지 검사 (CI)
	$(PYTHON) tools/check_readme_numbers.py --check && $(PYTHON) scripts/sync_tables.py --check

sync-numbers: ## README/docs 수치·표 동기화
	$(PYTHON) tools/check_readme_numbers.py && $(PYTHON) scripts/sync_tables.py

figures: ## results/figures 재생성
	$(PYTHON) scripts/make_figures.py

docs: ## mkdocs 정적 사이트 (site/)
	$(BIN)/mkdocs build --strict

# --------------------------------------------------------------------- 컨테이너·파이프라인
docker-build: ## 서빙 이미지 빌드
	docker build --target serve -t rtls-foresight:serve .

docker-up: ## api + mlflow + minio + redpanda + replay/consumer
	docker compose up --build -d

docker-down:
	docker compose down -v

dvc-repro: ## DVC 파이프라인 (download → … → benchmark)
	$(BIN)/dvc repro

clean: ## 산출물·캐시 정리 (데이터·체크포인트는 유지)
	rm -rf .pytest_cache .ruff_cache .mypy_cache site outputs multirun
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean ## 데이터·아티팩트까지 (다시 받거나 생성해야 함)
	rm -rf data/raw data/processed data/rtls artifacts/onnx mlflow.db mlruns mlartifacts
