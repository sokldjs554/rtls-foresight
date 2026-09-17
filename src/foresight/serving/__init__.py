"""서빙: FastAPI 앱, 스트리밍 소비자, 충돌 위험 점수.

``load_predictor`` 는 앱과 스트리밍 소비자가 공유하는 백엔드 팩토리다. 모델 경로 규칙:

* ``FORESIGHT_CKPT`` → 없으면 ``results/checkpoints/eth/seed0/best.pth`` → 없으면 공식 ETH 체크포인트.
* ``FORESIGHT_ONNX_DIR`` → 없으면 ``artifacts/onnx/`` (``social_stgcnn_fp32.onnx`` / ``social_stgcnn_int8.onnx``).
  ONNX 파일이 없으면 체크포인트에서 즉석 내보내기 한다 — ``foresight serve --backend onnx`` 가 export 를
  따로 돌리지 않아도 뜨게 하려고.

무거운 import(torch, onnxruntime, fastapi)는 함수 안에서 한다 — ``foresight --help`` 가 빨라야 하고,
``foresight.serving.risk`` 만 쓰는 코드가 torch 를 끌어오지 않게.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from foresight.utils import get_logger, project_root

if TYPE_CHECKING:
    from foresight.inference.predictor import Predictor

log = get_logger("foresight.serving")

BACKENDS = ("torch", "onnx", "onnx-int8")
DEFAULT_CKPT = Path("results/checkpoints/eth/seed0/best.pth")
OFFICIAL_CKPT = Path("assets/official_checkpoints/social-stgcnn-eth.pth")


def resolve_checkpoint_path() -> Path:
    root = project_root()
    env = os.environ.get("FORESIGHT_CKPT")
    candidates = [Path(env)] if env else []
    candidates += [root / DEFAULT_CKPT, root / OFFICIAL_CKPT]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"no checkpoint found among {[str(c) for c in candidates]}")


def resolve_onnx_dir() -> Path:
    env = os.environ.get("FORESIGHT_ONNX_DIR")
    return Path(env) if env else project_root() / "artifacts" / "onnx"


def resolve_onnx_path(backend: str) -> Path:
    from foresight.inference.export import FP32_NAME, INT8_NAME

    return resolve_onnx_dir() / (INT8_NAME if backend == "onnx-int8" else FP32_NAME)


def load_predictor(backend: str = "onnx", threads: int = 1, warmup: bool = True) -> Predictor:
    """백엔드 이름 → 준비된(웜업된) Predictor. ``model_path`` 속성으로 어떤 파일을 썼는지 남긴다."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "torch":
        from foresight.inference.predictor import TorchPredictor

        ckpt = resolve_checkpoint_path()
        pred = TorchPredictor.from_checkpoint(ckpt, threads=threads)
        pred.model_path = ckpt  # type: ignore[attr-defined]
        if warmup:
            import numpy as np

            obs = np.cumsum(np.random.default_rng(0).normal(0, 0.3, size=(4, 8, 2)), axis=1)
            for _ in range(3):
                pred.predict(obs, k=0)
        return pred
    from foresight.inference.onnx_backend import OnnxPredictor

    path = resolve_onnx_path(backend)
    if not path.exists():
        _export_missing(path, backend)
    pred = OnnxPredictor(
        path,
        threads=threads,
        name=backend if backend == "onnx-int8" else "onnx-fp32",
        torch_threads=1,
    )
    pred.model_path = path  # type: ignore[attr-defined]
    if warmup:
        pred.warmup()
    return pred


def _export_missing(path: Path, backend: str) -> None:
    """ONNX 산출물이 없을 때의 즉석 내보내기 (fp32 는 항상, INT8 은 보정 데이터가 있을 때만)."""
    from foresight.inference.export import FP32_NAME, export_onnx, load_model

    ckpt = resolve_checkpoint_path()
    log.warning("%s not found — exporting from %s", path, ckpt)
    model = load_model(ckpt)
    fp32 = path.parent / FP32_NAME
    if not fp32.exists():
        export_onnx(model, fp32)
    if backend == "onnx-int8":
        from foresight.data.ethucy import SceneSet
        from foresight.inference.quantize import quantize_static_scenes

        calib = project_root() / "data" / "processed" / "ethucy" / "eth" / "train.npz"
        if not calib.exists():
            raise FileNotFoundError(
                f"{path} missing and no calibration data at {calib}; run `foresight export` first"
            )
        quantize_static_scenes(fp32, path, SceneSet.load(calib), n_calib=64, variant="txp-only")


__all__ = [
    "BACKENDS",
    "load_predictor",
    "resolve_checkpoint_path",
    "resolve_onnx_dir",
    "resolve_onnx_path",
]
