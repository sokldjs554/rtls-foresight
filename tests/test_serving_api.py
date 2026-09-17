"""FastAPI 서빙 — 검증 오류(422), 실제 예측, 위험 점수, health/metrics."""

from __future__ import annotations

import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from foresight.utils import project_root

ROOT = project_root()
OFFICIAL = ROOT / "assets" / "official_checkpoints" / "social-stgcnn-eth.pth"
pytestmark = pytest.mark.skipif(not OFFICIAL.exists(), reason="official checkpoint missing")


def _traj(
    x0: float, y0: float, vx: float, vy: float, n: int = 8, dt: float = 0.4
) -> list[list[float]]:
    return [[round(x0 + vx * dt * t, 4), round(y0 + vy * dt * t, 4)] for t in range(n)]


def _payload(n: int = 3, k: int = 20) -> dict:
    agents = [
        {"id": f"a{i}", "type": i % 2, "obs": _traj(i * 2.0, 0.0, 1.0, 0.1 * i)} for i in range(n)
    ]
    return {"agents": agents, "k": k, "seed": 0}


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    """공식 체크포인트를 강제해 (학습 중인 best.pth 와 무관하게) 결정적인 테스트를 만든다."""
    from foresight.serving.app import create_app

    os.environ["FORESIGHT_CKPT"] = str(OFFICIAL)
    app = create_app(backend="torch", threads=1)
    return TestClient(app)


@pytest.fixture(scope="module")
def onnx_client() -> TestClient:
    from foresight.serving.app import create_app

    if not (ROOT / "artifacts" / "onnx" / "social_stgcnn_fp32.onnx").exists():
        pytest.skip("ONNX artifact missing (run `foresight export`)")
    return TestClient(create_app(backend="onnx", threads=1))


def test_health_reports_backend_and_sha(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert (
        body["backend"] == "torch"
        and body["warm"] is True
        and len(body["model_sha256"]) == 64
        and body["params"] == 7563
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["agents"][0]["obs"].pop(),  # 7 점
        lambda p: p["agents"][0]["obs"].append([0.0, 0.0]),  # 9 점
        lambda p: p.update(agents=[]),  # 0 명
        lambda p: p["agents"][0].update(type=2),  # 타입 범위
        lambda p: p.update(k=-1),
        lambda p: p.update(k=1000),
        lambda p: p["agents"][0].update(obs=[[1.0, 2.0, 3.0]] * 8),  # 3-D 점
        lambda p: p["agents"][1].update(id="a0"),  # 중복 id
        lambda p: p.update(bogus=1),  # extra=forbid
    ],
)
def test_validation_errors_are_422(client: TestClient, mutate) -> None:
    p = _payload()
    mutate(p)
    r = client.post("/predict", json=p)
    assert r.status_code == 422, r.text


def test_too_many_agents_is_422(client: TestClient) -> None:
    p = _payload(n=201)
    assert client.post("/predict", json=p).status_code == 422


def test_predict_real_model(client: TestClient) -> None:
    p = _payload(n=4, k=20)
    r = client.post("/predict", json=p)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["backend"] == "torch" and body["k"] == 20 and len(body["predictions"]) == 4
    pr = body["predictions"][0]
    assert (
        len(pr["mean"]) == 12
        and len(pr["mean"][0]) == 2
        and len(pr["sigma"]) == 12
        and len(pr["rho"]) == 12
    )
    assert pr["samples"] is None
    assert all(abs(x) < 1 for x in pr["rho"]) and all(s[0] > 0 for s in pr["sigma"])
    # 마지막 관측에서 이어지는 궤적이어야 한다 (첫 예측 스텝이 마지막 관측에서 수 m 안)
    last = p["agents"][0]["obs"][-1]
    assert abs(pr["mean"][0][0] - last[0]) < 3 and abs(pr["mean"][0][1] - last[1]) < 3
    assert {"preprocess_ms", "model_ms", "postprocess_ms", "total_ms"} <= set(body["timing_ms"])
    # 같은 시드 → 같은 결과
    assert client.post("/predict", json=p).json()["predictions"] == body["predictions"]


def test_predict_include_samples_and_k0(client: TestClient) -> None:
    p = _payload(n=2, k=5)
    p["include_samples"] = True
    body = client.post("/predict", json=p).json()
    assert (
        len(body["predictions"][0]["samples"]) == 5
        and len(body["predictions"][0]["samples"][0]) == 12
    )
    p0 = _payload(n=2, k=0)
    body0 = client.post("/predict", json=p0).json()
    assert body0["k"] == 0 and body0["predictions"][0]["samples"] is None


def test_risk_head_on_pair_alerts(client: TestClient) -> None:
    agents = [
        {"id": "worker", "type": 0, "obs": _traj(-3.0, 0.0, 1.0, 0.0)},
        {"id": "forklift", "type": 1, "obs": _traj(3.0, 0.0, -1.0, 0.0)},
        {"id": "far-forklift", "type": 1, "obs": _traj(0.0, 30.0, 1.0, 0.0)},
    ]
    r = client.post(
        "/risk", json={"agents": agents, "k": 20, "seed": 0, "d_safe": 1.0, "threshold": 0.3}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_workers"] == 1 and body["n_vehicles"] == 2 and len(body["pairs"]) == 2
    by_v = {p["vehicle_id"]: p for p in body["pairs"]}
    assert by_v["forklift"]["risk"] > 0.5 and by_v["forklift"]["ttc_s"] is not None
    assert by_v["far-forklift"]["risk"] == 0.0 and by_v["far-forklift"]["ttc_s"] is None
    assert [a["vehicle_id"] for a in body["alerts"]] == ["forklift"]
    assert "risk_ms" in body["timing_ms"]
    # 결정적 경로
    d = client.post("/risk", json={"agents": agents, "k": 20, "deterministic": True}).json()
    assert d["k"] == 0 and d["pairs"][0]["risk"] in (0.0, 1.0)


def test_risk_without_vehicles_is_empty(client: TestClient) -> None:
    p = {"agents": [{"id": "w", "type": 0, "obs": _traj(0, 0, 1, 0)}], "k": 5}
    body = client.post("/risk", json=p).json()
    assert body["pairs"] == [] and body["alerts"] == [] and body["n_vehicles"] == 0


def test_metrics_exposes_histograms_and_counters(client: TestClient) -> None:
    client.post("/predict", json=_payload(n=2, k=1))
    r = client.get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    text = r.text
    assert "foresight_request_latency_seconds_bucket" in text
    assert "foresight_predictions_total" in text and "foresight_agents_per_request_bucket" in text
    assert (
        'foresight_requests_total{endpoint="/predict",status="422"}' in text
    )  # 앞선 검증 실패도 기록됨


def test_onnx_backend_matches_torch(client: TestClient, onnx_client: TestClient) -> None:
    p = _payload(n=5, k=0)
    a = client.post("/predict", json=p).json()
    b = onnx_client.post("/predict", json=p).json()
    assert b["backend"] == "onnx-fp32"
    ma, mb = (
        np.array([x["mean"] for x in a["predictions"]]),
        np.array([x["mean"] for x in b["predictions"]]),
    )
    assert np.abs(ma - mb).max() < 1e-3
    assert onnx_client.get("/health").json()["backend"] == "onnx-fp32"


def test_unknown_backend_rejected() -> None:
    from foresight.serving.app import create_app

    with pytest.raises(ValueError):
        create_app(backend="tensorrt")
