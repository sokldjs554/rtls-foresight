"""FastAPI 추론 서버 — ``/predict``, ``/risk``, ``/health``, ``/metrics``.

설계 메모
* 엔드포인트는 동기 ``def`` 로 둔다. Starlette 가 스레드풀에서 돌리므로 모델 호출(수 ms)이 이벤트 루프를
  막지 않고, ORT 세션·torch eager 는 동시 호출에 안전하다.
* 직렬화: FastAPI ≥ 0.14x 는 ``response_model`` 이 있으면 pydantic-core 가 JSON bytes 를 직접 만든다 (Rust 구현,
  orjson 경유보다 빠르고 ``JSONResponse`` 는 deprecated). 그래서 응답 클래스를 바꾸지 않고 스키마만 선언한다.
* Prometheus 레지스트리는 앱마다 새로 만든다 — 테스트에서 ``create_app`` 을 여러 번 불러도 "duplicated
  timeseries" 로 죽지 않게. ``/metrics`` 는 그 레지스트리만 노출한다.
* 지연 히스토그램 버킷은 ms 단위로 촘촘하게(0.5 ms~) 잡는다: 기본 버킷(5 ms~)은 이 모델의 전 구간이
  한 버킷에 몰려 p95 를 읽을 수 없다.
* 모델 sha256 을 기동 시 한 번 계산해 ``/health`` 에 싣는다 — 어떤 아티팩트가 떠 있는지 배포에서 확인하는 용도.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from foresight.inference.predictor import Prediction, Predictor
from foresight.serving import BACKENDS, load_predictor
from foresight.serving.risk import RiskMatrix, risk_from_prediction
from foresight.serving.schemas import (
    HealthResponse,
    PredictionOut,
    PredictRequest,
    PredictResponse,
    RiskPairOut,
    RiskRequest,
    RiskResponse,
)
from foresight.utils import get_logger

log = get_logger("foresight.serving.app")

LATENCY_BUCKETS = (
    0.0005,
    0.001,
    0.002,
    0.003,
    0.005,
    0.0075,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    1.0,
)


def _sha256(path: os.PathLike[str] | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _obs_array(req: PredictRequest) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray([a.obs for a in req.agents], dtype=np.float64)  # (N, 8, 2)
    types = np.asarray([a.type for a in req.agents], dtype=np.int8)
    return obs, types


def _prediction_outputs(req: PredictRequest, pred: Prediction) -> list[PredictionOut]:
    params = pred.params  # (T, N, 5)
    sigma = np.exp(params[..., 2:4]).transpose(1, 0, 2)  # (N, T, 2)
    rho = np.tanh(params[..., 4]).T  # (N, T)
    out: list[PredictionOut] = []
    for i, a in enumerate(req.agents):
        samples = None
        if req.include_samples and pred.samples_abs is not None:
            samples = pred.samples_abs[:, i].round(4).tolist()
        out.append(
            PredictionOut(
                id=a.id,
                type=a.type,
                mean=pred.mean_abs[i].round(4).tolist(),
                sigma=sigma[i].round(4).tolist(),
                rho=rho[i].round(4).tolist(),
                samples=samples,
            )
        )
    return out


def _pairs_out(req: RiskRequest, rm: RiskMatrix) -> list[RiskPairOut]:
    ids = [a.id for a in req.agents]
    return [
        RiskPairOut(
            worker_id=ids[p.worker],
            vehicle_id=ids[p.vehicle],
            risk=round(p.risk, 4),
            ttc_s=None if np.isnan(p.ttc_s) else round(p.ttc_s, 3),
            min_dist_mean=round(float(p.min_dist_mean), 4),
        )
        for p in rm.pairs()
    ]


def create_app(backend: str = "onnx", threads: int | None = None) -> FastAPI:
    """백엔드 ∈ {torch, onnx, onnx-int8}. 모델은 앱 생성 시 로드·웜업한다 (TestClient 에서도 바로 쓰이도록)."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    threads = threads or int(os.environ.get("FORESIGHT_THREADS", "1"))
    t0 = time.perf_counter()
    predictor: Predictor = load_predictor(backend, threads=threads, warmup=True)
    model_path = getattr(predictor, "model_path", "")
    model_sha = _sha256(model_path) if model_path else ""
    n_params = None
    if hasattr(predictor, "model"):
        n_params = int(sum(p.numel() for p in predictor.model.parameters()))  # type: ignore[attr-defined]
    log.info(
        "backend=%s model=%s sha=%s threads=%d loaded in %.2fs",
        predictor.name,
        model_path,
        model_sha[:12],
        threads,
        time.perf_counter() - t0,
    )

    registry = CollectorRegistry()
    h_latency = Histogram(
        "foresight_request_latency_seconds",
        "요청 처리 시간 (전처리+모델+후처리+직렬화 전)",
        ["endpoint"],
        buckets=LATENCY_BUCKETS,
        registry=registry,
    )
    c_predictions = Counter(
        "foresight_predictions_total", "예측한 에이전트 수 (누적)", ["endpoint"], registry=registry
    )
    c_requests = Counter(
        "foresight_requests_total", "요청 수", ["endpoint", "status"], registry=registry
    )
    h_agents = Histogram(
        "foresight_agents_per_request",
        "요청당 에이전트 수",
        buckets=(1, 2, 3, 5, 8, 12, 20, 30, 50, 100, 200),
        registry=registry,
    )
    c_alerts = Counter(
        "foresight_alerts_total", "/risk 에서 임계를 넘은 쌍 수 (누적)", registry=registry
    )

    app = FastAPI(
        title="rtls-foresight",
        version="0.1.0",
        description="Social-STGCNN 궤적 예측 + 작업자·차량 충돌 위험 API",
    )
    app.state.predictor = predictor
    app.state.backend = backend
    app.state.registry = registry
    app.state.model_sha = model_sha
    app.state.threads = threads

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        c_requests.labels(endpoint=request.url.path, status="500").inc()
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            backend=predictor.name,
            model_path=str(model_path),
            model_sha256=model_sha,
            warm=True,
            params=n_params,
            threads=threads,
        )

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        t_start = time.perf_counter()
        obs, _ = _obs_array(req)
        pred = predictor.predict(obs, k=req.k, seed=req.seed)
        outputs = _prediction_outputs(req, pred)
        total = time.perf_counter() - t_start
        h_latency.labels(endpoint="/predict").observe(total)
        c_predictions.labels(endpoint="/predict").inc(len(req.agents))
        c_requests.labels(endpoint="/predict", status="200").inc()
        h_agents.observe(len(req.agents))
        timing = {k: round(v, 3) for k, v in pred.timing_ms.items()}
        timing["total_ms"] = round(total * 1e3, 3)
        return PredictResponse(
            predictions=outputs, k=req.k, backend=predictor.name, timing_ms=timing
        )

    @app.post("/risk", response_model=RiskResponse)
    def risk(req: RiskRequest) -> RiskResponse:
        t_start = time.perf_counter()
        obs, types = _obs_array(req)
        pred = predictor.predict(obs, k=req.k, seed=req.seed)
        t_risk = time.perf_counter()
        rm = risk_from_prediction(
            pred, types, d_safe=req.d_safe, deterministic=req.deterministic or req.k == 0
        )
        pairs = _pairs_out(req, rm)
        alerts = [p for p in pairs if p.risk >= req.threshold]
        total = time.perf_counter() - t_start
        h_latency.labels(endpoint="/risk").observe(total)
        c_predictions.labels(endpoint="/risk").inc(len(req.agents))
        c_requests.labels(endpoint="/risk", status="200").inc()
        h_agents.observe(len(req.agents))
        c_alerts.inc(len(alerts))
        timing = {k: round(v, 3) for k, v in pred.timing_ms.items()}
        timing["risk_ms"] = round((time.perf_counter() - t_risk) * 1e3, 3)
        timing["total_ms"] = round(total * 1e3, 3)
        return RiskResponse(
            pairs=pairs,
            alerts=alerts,
            n_workers=len(rm.worker_idx),
            n_vehicles=len(rm.vehicle_idx),
            k=0 if (req.deterministic or req.k == 0) else rm.k,
            d_safe=req.d_safe,
            threshold=req.threshold,
            backend=predictor.name,
            timing_ms=timing,
        )

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": "rtls-foresight",
            "backend": predictor.name,
            "endpoints": ["/health", "/metrics", "/predict", "/risk", "/docs"],
        }

    # 422 도 카운터에 남긴다 (검증 실패율 모니터링)
    from fastapi.exception_handlers import request_validation_exception_handler
    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> Any:
        c_requests.labels(endpoint=request.url.path, status="422").inc()
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(HTTPException)
    async def _http(request: Request, exc: HTTPException) -> JSONResponse:
        c_requests.labels(endpoint=request.url.path, status=str(exc.status_code)).inc()
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    return app


def get_app() -> FastAPI:  # uvicorn --factory foresight.serving.app:get_app
    return create_app(os.environ.get("FORESIGHT_BACKEND", "onnx"))


def app_from_env():
    """환경변수 FORESIGHT_BACKEND 로 앱을 만든다 (`foresight serve --workers N`, Docker CMD)."""
    import os

    return create_app(backend=os.environ.get("FORESIGHT_BACKEND", "onnx"))
