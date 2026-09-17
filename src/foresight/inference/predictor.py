"""추론 인터페이스 — 서빙·스트리밍·벤치마크가 공통으로 쓰는 계약.

입력은 항상 **절대좌표 관측** ``obs_abs (N, T_obs, 2)`` (미터). 전처리(상대 변위 + 정규화 라플라시안)는
Predictor 안에서 한다 — 서비스 지연 예산에 전처리가 포함되어야 벤치마크가 정직하다.

백엔드
* ``TorchPredictor``  — PyTorch eager (선택적으로 ``torch.compile``).
* ``OnnxPredictor``   — ONNX Runtime (fp32 또는 정적 INT8), ``foresight.inference.onnx_backend`` 에서 구현.

출력 ``Prediction``: 상대 변위 분포 파라미터, 평균 궤적(절대좌표), K 개 샘플(절대좌표), 전처리·모델 시간.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from foresight.data.graph import (
    Kernel,
    inverse_distance_kernel,
    normalized_laplacian,
    relative_displacement,
)
from foresight.eval import metrics as M
from foresight.models import SocialSTGCNN


@dataclass
class Prediction:
    params: np.ndarray  # (T_pred, N, 5) 상대 변위 분포 파라미터 (μx, μy, log σx, log σy, atanh ρ)
    mean_abs: np.ndarray  # (N, T_pred, 2) 평균(결정적) 궤적, 절대좌표
    samples_abs: np.ndarray | None = None  # (K, N, T_pred, 2)
    timing_ms: dict[str, float] = field(default_factory=dict)


def preprocess(
    obs_abs: np.ndarray, kernel: Kernel = "velocity", normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """``(N, T_obs, 2)`` → ``v (1, 2, T_obs, N)``, ``a (T_obs, N, N)`` float32 (학습 전처리와 동일)."""
    rel = relative_displacement(obs_abs.astype(np.float64))  # (N, T, 2)
    v_all = rel.transpose(1, 0, 2)  # (T, N, 2)
    feat = v_all if kernel == "velocity" else obs_abs.transpose(1, 0, 2)
    a = inverse_distance_kernel(feat)
    if normalize:
        a = normalized_laplacian(a)
    v = v_all.astype(np.float32).transpose(2, 0, 1)[None]  # (1, 2, T, N)
    return np.ascontiguousarray(v), a.astype(np.float32)


class Predictor(Protocol):
    name: str

    def predict(self, obs_abs: np.ndarray, k: int = 0, seed: int | None = None) -> Prediction: ...


class TorchPredictor:
    name = "torch"

    def __init__(
        self,
        model: SocialSTGCNN,
        compile: bool = False,
        kernel: Kernel = "velocity",
        normalize: bool = True,
        threads: int | None = None,
    ) -> None:
        self.model = model.eval()
        self.kernel, self.normalize = kernel, normalize
        if threads:
            torch.set_num_threads(threads)
        self._fn = torch.compile(self.model, dynamic=True) if compile else self.model
        if compile:
            self.name = "torch-compile"

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kw: object) -> TorchPredictor:
        from foresight.eval.evaluate import load_model

        return cls(load_model(Path(path)), **kw)  # type: ignore[arg-type]

    def predict(self, obs_abs: np.ndarray, k: int = 0, seed: int | None = None) -> Prediction:
        t0 = time.perf_counter()
        v, a = preprocess(obs_abs, self.kernel, self.normalize)
        t1 = time.perf_counter()
        with torch.no_grad():
            out = self._fn(torch.from_numpy(v), torch.from_numpy(a))  # (1, 5, T_pred, N)
            params = out.permute(0, 2, 3, 1)[0]  # (T_pred, N, 5)
        t2 = time.perf_counter()
        return finalize(
            params,
            obs_abs,
            k,
            seed,
            {"preprocess_ms": (t1 - t0) * 1e3, "model_ms": (t2 - t1) * 1e3},
        )


def finalize(
    params: torch.Tensor, obs_abs: np.ndarray, k: int, seed: int | None, timing: dict[str, float]
) -> Prediction:
    """모델 출력 → Prediction (평균 궤적 + K 샘플, 절대좌표)."""
    t0 = time.perf_counter()
    last = torch.from_numpy(obs_abs[:, -1].astype(np.float32))  # (N, 2)
    mean_abs = M.relative_to_absolute(params[..., :2], last).permute(1, 0, 2).numpy()  # (N, T, 2)
    samples = None
    if k > 0:
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        rel = M.sample_relative(params, k, generator=g)
        samples = M.relative_to_absolute(rel, last).permute(0, 2, 1, 3).numpy()  # (K, N, T, 2)
    timing["postprocess_ms"] = (time.perf_counter() - t0) * 1e3
    return Prediction(
        params=params.numpy(), mean_abs=mean_abs, samples_abs=samples, timing_ms=timing
    )
