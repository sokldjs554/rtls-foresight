"""ONNX 내보내기·양자화·ORT 백엔드 — torch 와의 동등성, 동적 N, Predictor 계약."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from foresight.inference.export import (
    FP32_NAME,
    INT8_NAME,
    check_parity,
    export_onnx,
    parity_scenes,
    resolve_checkpoint,
)
from foresight.inference.onnx_backend import OnnxPredictor
from foresight.inference.predictor import TorchPredictor, preprocess
from foresight.utils import project_root

ROOT = project_root()
DATA = ROOT / "data" / "processed" / "ethucy"
OFFICIAL = ROOT / "assets" / "official_checkpoints" / "social-stgcnn-eth.pth"
ARTIFACTS = ROOT / "artifacts" / "onnx"

pytestmark = pytest.mark.skipif(not OFFICIAL.exists(), reason="official checkpoint missing")


@pytest.fixture(scope="module")
def model() -> torch.nn.Module:
    from foresight.inference.export import load_model

    return load_model(OFFICIAL)


@pytest.fixture(scope="module")
def onnx_dir(tmp_path_factory: pytest.TempPathFactory, model: torch.nn.Module) -> Path:
    """산출물이 있으면 그대로, 없으면 임시 디렉터리에 fp32 + (보정 데이터가 있을 때) INT8 을 만든다."""
    if (ARTIFACTS / FP32_NAME).exists():
        return ARTIFACTS
    d = tmp_path_factory.mktemp("onnx")
    export_onnx(model, d / FP32_NAME)
    calib = DATA / "eth" / "train.npz"
    if calib.exists():
        from foresight.data.ethucy import SceneSet
        from foresight.inference.quantize import quantize_static_scenes

        quantize_static_scenes(
            d / FP32_NAME, d / INT8_NAME, SceneSet.load(calib), n_calib=32, variant="txp-only"
        )
    return d


def test_export_is_dynamic_in_n_and_checker_passes(model: torch.nn.Module, tmp_path: Path) -> None:
    import onnx

    info = export_onnx(model, tmp_path / "m.onnx")
    assert info["opset"] >= 17
    m = onnx.load(str(tmp_path / "m.onnx"))
    dims = {
        i.name: [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]
        for i in m.graph.input
    }
    assert dims["v"][:3] == [1, 2, 8] and isinstance(dims["v"][3], str)
    assert isinstance(dims["a"][1], str) and dims["a"][1] == dims["a"][2]
    assert not (tmp_path / "m.onnx.data").exists(), "weights must be embedded, not external"


def test_parity_on_real_scenes_across_sizes(model: torch.nn.Module, onnx_dir: Path) -> None:
    scenes = parity_scenes(DATA, min_scenes=50)
    sizes = {s.shape[0] for s in scenes}
    assert len(scenes) >= 50 and min(sizes) <= 2 and max(sizes) >= 20
    res = check_parity(model, onnx_dir / FP32_NAME, scenes, tol=1e-4)
    assert res["max_abs_diff"] < 1e-4


def test_onnx_predictor_matches_torch_predictor(model: torch.nn.Module, onnx_dir: Path) -> None:
    tp = TorchPredictor(model, threads=1)
    op = OnnxPredictor(onnx_dir / FP32_NAME, threads=1)
    op.warmup()
    assert op.name == "onnx-fp32"
    rng = np.random.default_rng(1)
    for n in (2, 7, 33):
        obs = np.cumsum(rng.normal(0, 0.3, size=(n, 8, 2)), axis=1)
        a, b = tp.predict(obs, k=20, seed=3), op.predict(obs, k=20, seed=3)
        assert a.mean_abs.shape == (n, 12, 2) and b.mean_abs.shape == (n, 12, 2)
        assert np.abs(a.mean_abs - b.mean_abs).max() < 1e-4
        assert np.abs(a.params - b.params).max() < 1e-4
        assert b.samples_abs is not None and b.samples_abs.shape == (20, n, 12, 2)
        # 같은 시드 → 같은 샘플 (파라미터가 1e-6 수준으로만 다르므로)
        assert np.abs(a.samples_abs - b.samples_abs).max() < 1e-3
        assert set(b.timing_ms) == {"preprocess_ms", "model_ms", "postprocess_ms"}


def test_preprocess_matches_dataset_graph() -> None:
    """서빙 전처리가 학습 데이터 파이프라인(scene_to_graph)과 동일해야 한다."""
    from foresight.data.graph import scene_to_graph

    rng = np.random.default_rng(0)
    pos = np.cumsum(rng.normal(0, 0.3, size=(6, 20, 2)), axis=1)
    v, a = preprocess(pos[:, :8])
    vo, ao, _, _ = scene_to_graph(pos, 8)
    assert np.allclose(v[0].transpose(1, 2, 0), vo) and np.allclose(a, ao)


@pytest.mark.skipif(not (DATA / "eth" / "train.npz").exists(), reason="calibration data missing")
def test_int8_model_runs_and_is_close(model: torch.nn.Module, onnx_dir: Path) -> None:
    int8 = onnx_dir / INT8_NAME
    assert int8.exists()
    op = OnnxPredictor(int8, threads=1)
    assert op.name == "onnx-int8"
    tp = TorchPredictor(model, threads=1)
    scenes = parity_scenes(DATA, min_scenes=20)[:20]
    diffs = []
    for obs in scenes:
        a, b = tp.predict(obs, k=0), op.predict(obs, k=0)
        assert np.isfinite(b.mean_abs).all()
        diffs.append(np.abs(a.mean_abs - b.mean_abs).mean())
    # INT8 은 정확히 같지 않지만 평균 궤적 차이가 수 cm 수준이어야 한다 (미터 단위 궤적)
    assert float(np.mean(diffs)) < 0.3


def test_manifest_records_source_and_parity() -> None:
    mf = ARTIFACTS / "manifest.json"
    if not mf.exists():
        pytest.skip("run `foresight export` to produce the manifest")
    import json

    m = json.loads(mf.read_text())
    assert m["param_count"] == 7563 and m["parity"]["max_abs_diff"] < 1e-4
    assert FP32_NAME in m["files"] and len(m["files"][FP32_NAME]["sha256"]) == 64


def test_resolve_checkpoint_falls_back_to_official(tmp_path: Path) -> None:
    assert resolve_checkpoint(tmp_path / "missing.pth") == OFFICIAL
