"""데모 JS 구현(demo/foresight.js)이 PyTorch 구현과 같은 값을 내는지 Node 로 비교한다.

전처리 + 순전파(params), 평균 궤적, 같은 샘플로 계산한 위험 행렬, 경보 정책 결정을 각각 비교한다. Node 가 없으면 건너뛴다.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from foresight.demo.replay import export_weights, write_js
from foresight.eval import metrics as M
from foresight.eval.evaluate import load_model
from foresight.inference.predictor import TorchPredictor
from foresight.serving.risk import AlertPolicy, PairRisk, pairwise_risk
from foresight.utils import project_root

ROOT = project_root()
CKPT = ROOT / "results" / "checkpoints" / "rtls-scratch-fast" / "best.pth"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node 가 없다")


def _cases(rng: np.random.Generator) -> list[np.ndarray]:
    out = []
    for n in (1, 3, 7):
        start = rng.uniform(0, 20, size=(n, 1, 2))
        vel = rng.normal(0, 0.4, size=(n, 1, 2))
        steps = np.cumsum(np.repeat(vel, 8, axis=1) + rng.normal(0, 0.05, size=(n, 8, 2)), axis=1)
        out.append((start + steps).astype(np.float32).astype(np.float64))
    # 두 에이전트의 변위가 float32 에서 정확히 같은 경우(커널 0) 도 포함
    twin = out[1].copy()
    twin[1] = twin[0] + np.array([3.0, 0.0])
    out.append(twin)
    return out


def _run_node(model_js: Path, payload: dict) -> dict:
    inp = model_js.parent / "harness_input.json"
    inp.write_text(json.dumps(payload), encoding="utf-8")
    res = subprocess.run(
        [NODE, str(ROOT / "tests" / "demo_harness.js"), str(model_js), str(inp)],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return json.loads(res.stdout)


def test_js_matches_pytorch(tmp_path: Path) -> None:
    if not CKPT.exists():
        pytest.skip("RTLS 체크포인트가 없다")
    model = load_model(CKPT)
    model_js = tmp_path / "model.js"
    write_js(model_js, "FORESIGHT_MODEL", export_weights(model, source=str(CKPT.name)))
    rng = np.random.default_rng(7)
    cases = _cases(rng)
    predictor = TorchPredictor(model)
    py = [predictor.predict(obs, k=0) for obs in cases]

    # 위험 행렬: 파이썬이 뽑은 샘플을 그대로 JS 에 넘겨 같은 답이 나오는지 (샘플링 RNG 는 다르므로 분리)
    obs = cases[2]
    types = np.array([0, 1, 0, 1, 1, 0, 0], dtype=np.int8)
    params = torch.from_numpy(py[2].params)
    g = torch.Generator().manual_seed(0)
    rel = M.sample_relative(params, 20, generator=g)
    last = torch.from_numpy(obs[:, -1].astype(np.float32))
    samples = M.relative_to_absolute(rel, last).permute(0, 2, 1, 3).numpy()  # (K, N, T, 2)
    rm = pairwise_risk(samples, types, 1.0)

    # 경보 정책: 같은 위험 시퀀스 → 같은 경보 시각
    seq = [0.1, 0.35, 0.4, 0.2, 0.31, 0.5, 0.5, 0.05, 0.9, 0.9]
    frames = [[{"wid": "W1", "vid": "V1", "risk": r, "ttc": 1.0, "minDist": 0.5}] for r in seq]
    pol = AlertPolicy(threshold=0.3, cooldown_s=1.0, min_consecutive=2)
    py_alerts = []
    for i, r in enumerate(seq):
        for a in pol.update([("W1", "V1", PairRisk(0, 1, r, 1.0, 0.5))], i * 0.4):
            py_alerts.append(["W1", "V1", round(a.ts_s, 6), a.consecutive])

    out = _run_node(
        model_js,
        {
            "cases": [{"obs": c.tolist()} for c in cases],
            "risk_case": {"samples": samples.tolist(), "types": types.tolist(), "d_safe": 1.0},
            "policy_case": {
                "frames": frames,
                "opts": {"threshold": 0.3, "cooldownS": 1.0, "minConsecutive": 2},
            },
        },
    )
    for i, p in enumerate(py):
        js_params = np.array(out["params"][i])  # (T, N, 5)
        assert js_params.shape == p.params.shape
        assert np.abs(js_params - p.params).max() < 2e-4, f"case {i}: params 불일치"
        js_mean = np.array(out["mean"][i])  # (N, T, 2)
        assert np.abs(js_mean - p.mean_abs).max() < 2e-3, f"case {i}: 평균 궤적 불일치"
    js_risk = np.array(out["risk"]["risk"])
    assert js_risk.shape == rm.risk.shape and np.abs(js_risk - rm.risk).max() < 1e-9
    js_md = np.array(out["risk"]["minDist"])
    assert np.abs(js_md - rm.min_dist_mean).max() < 1e-6
    js_ttc = np.array([[np.nan if v is None else v for v in row] for row in out["risk"]["ttc"]])
    assert np.allclose(js_ttc, rm.ttc_s, equal_nan=True, atol=1e-9)
    assert [[a[0], a[1], round(a[2], 6), a[3]] for a in out["alerts"]] == py_alerts
    assert out["n_suppressed"] == pol.n_suppressed


def test_export_weights_shapes() -> None:
    if not CKPT.exists():
        pytest.skip("RTLS 체크포인트가 없다")
    model = load_model(CKPT)
    w = export_weights(model)
    assert w["meta"]["param_count"] == 7563
    assert len(w["st_gcn"]["gcn_w"]) == 5 and len(w["st_gcn"]["gcn_w"][0]) == 2
    assert np.array(w["tpcnns"][0]["w"]).shape == (12, 8, 3, 3)
    assert np.array(w["tpcnn_output"]["w"]).shape == (12, 12, 3, 3)
    assert w["st_gcn"]["res_kind"] == "conv"
