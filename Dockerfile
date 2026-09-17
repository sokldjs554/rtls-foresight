# 다단계 빌드. serve 타깃은 학습 의존성(mlflow, hydra, dvc, 노트북)을 넣지 않는다 — 서빙 이미지는 작고, 학습 코드가 없어야
# "무엇이 배포되는가"가 분명하다. train 타깃은 전체 파이프라인(재현·DVC)을 컨테이너에서 돌릴 때 쓴다.
#
#   docker build --target serve -t rtls-foresight:serve .
#   docker run --rm -p 8000:8000 rtls-foresight:serve                 # onnx 백엔드, /health, /docs
#   docker build --target train -t rtls-foresight:train . && docker run --rm rtls-foresight:train foresight --help

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    FORESIGHT_ROOT=/app OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /app
# CPU 휠: 7.6K 파라미터 모델에 CUDA 런타임(2.5 GB)은 필요 없다. 프록시 뒤에서는 이 인덱스를 미러로 바꾼다.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install "numpy>=1.26" "polars>=1.0" "pyarrow>=15" "pydantic>=2.6" "pydantic-settings>=2.2" "pyyaml>=6.0" "typer>=0.12" "rich>=13.7" "tqdm>=4.66"

# ---------------------------------------------------------------- serve
FROM base AS serve
RUN pip install "onnx>=1.16" "onnxruntime>=1.18" "fastapi>=0.110" "uvicorn[standard]>=0.29" "prometheus-client>=0.20" "httpx>=0.27" "confluent-kafka>=2.3"
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY assets/official_checkpoints ./assets/official_checkpoints
# 학습된 체크포인트·ONNX 는 있으면 담고 없으면 공식 체크포인트로 폴백한다 (foresight.serving.load_predictor).
COPY results/checkpoints* ./results/checkpoints/
COPY artifacts/onnx* ./artifacts/onnx/
RUN pip install --no-deps -e . && python -c "import foresight, foresight.serving.app" \
 && useradd --create-home --uid 10001 app && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status==200 else 1)"
ENV FORESIGHT_BACKEND=onnx
ENTRYPOINT ["foresight"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000", "--backend", "onnx"]

# ---------------------------------------------------------------- train
FROM base AS train
RUN apt-get update && apt-get install -y --no-install-recommends git make && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[serve,stream,dvc,dev]"
COPY . .
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app
ENTRYPOINT ["foresight"]
CMD ["--help"]
