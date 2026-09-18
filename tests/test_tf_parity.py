"""TensorFlow 이식(models/tf_port.py)이 PyTorch 와 같은 파라미터를 내는지 — tensorflow 가 없으면 건너뛴다."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foresight.demo.replay import export_weights
from foresight.eval.evaluate import load_model
from foresight.inference.predictor import TorchPredictor, preprocess
from foresight.utils import project_root

tf = pytest.importorskip("tensorflow")

CKPT = project_root() / "results" / "checkpoints" / "rtls-scratch-fast" / "best.pth"


def _scenes(rng: np.random.Generator) -> list[np.ndarray]:
    out = []
    for n in (1, 4, 9):
        start = rng.uniform(0, 20, size=(n, 1, 2))
        vel = rng.normal(0, 0.4, size=(n, 1, 2))
        steps = np.cumsum(np.repeat(vel, 8, axis=1) + rng.normal(0, 0.05, size=(n, 8, 2)), axis=1)
        out.append((start + steps).astype(np.float32).astype(np.float64))
    return out


def test_tf_forward_matches_torch(tmp_path: Path) -> None:
    if not CKPT.exists():
        pytest.skip("RTLS 체크포인트가 없다")
    from foresight.models.tf_port import SocialSTGCNNTF, predict_params

    model = load_model(CKPT)
    weights = export_weights(model)
    tf_model = SocialSTGCNNTF(weights)
    predictor = TorchPredictor(model)
    scenes = _scenes(np.random.default_rng(3))
    for obs in scenes:
        ref = predictor.predict(obs, k=0).params  # (T, N, 5)
        got = predict_params(weights, obs)
        assert got.shape == ref.shape
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)
    # SavedModel 로 내보내고 다시 읽어도 같은 값 (동적 N 서명)
    path = tf_model.export_saved_model(tmp_path / "savedmodel")
    loaded = tf.saved_model.load(str(path))
    fn = loaded.signatures["serving_default"]
    v, a = preprocess(scenes[1])
    out = fn(v=tf.constant(v), a=tf.constant(a))["params"].numpy()[0]
    np.testing.assert_allclose(out, predictor.predict(scenes[1], k=0).params, rtol=1e-4, atol=1e-4)
    assert (path / "saved_model.pb").exists()
