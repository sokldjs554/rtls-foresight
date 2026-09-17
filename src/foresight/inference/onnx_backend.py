"""ONNX Runtime 백엔드 — ``Predictor`` 프로토콜 구현.

전처리(``preprocess``)와 후처리(``finalize``)는 ``predictor.py`` 의 것을 그대로 쓴다. 백엔드가 바뀌어도
입출력 의미가 같아야 서빙/스트리밍/벤치마크가 교체 가능하고, 벤치마크에서 "모델 시간"만 따로 비교할 수 있다.

세션 옵션
* ``intra_op_num_threads``: 7.6K 파라미터 모델은 스레드를 늘려도 이득이 없고 p99 꼬리만 길어진다
  (벤치마크 참조). 기본 1.
* ``ORT_ENABLE_ALL``: Conv+BN 접기, 상수 접기, 레이아웃 최적화. 로드 시 한 번만 비용을 낸다.
* ``warmup()``: 첫 run 은 메모리 계획·커널 선택으로 수 ms 가 더 걸리므로 서버 기동 시 미리 돌린다.
* ``torch_threads``: 후처리 샘플링이 torch 라서 torch 스레드 수도 여기서 고정한다 (기본 1). 안 하면 torch 기본값
  (코어 수)으로 돌아 OpenMP 스핀 대기 때문에 후처리가 20 배 느려진다 — 서빙에서 실제로 겪은 회귀.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from foresight.data.graph import Kernel
from foresight.inference.predictor import Prediction, finalize, preprocess


class OnnxPredictor:
    name = "onnx-fp32"

    def __init__(
        self,
        path: str | Path,
        threads: int = 1,
        kernel: Kernel = "velocity",
        normalize: bool = True,
        obs_len: int = 8,
        name: str | None = None,
        torch_threads: int = 1,
    ) -> None:
        import onnxruntime as ort

        # 후처리(K 샘플 추출)는 torch 로 돌아간다. torch 의 기본 스레드 수(=코어 수)를 그대로 두면 OpenMP 스핀 대기가
        # 0.9 ms 짜리 후처리를 20 ms 이상으로 늘린다 (벤치마크 문서 §3). 소형 텐서 연산이라 1 스레드가 항상 최선.
        torch.set_num_threads(int(torch_threads))
        self.path = Path(path)
        self.kernel, self.normalize, self.obs_len = kernel, normalize, obs_len
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
        names = [i.name for i in self.session.get_inputs()]
        if names != ["v", "a"]:
            raise ValueError(f"unexpected ONNX inputs {names}; expected ['v', 'a']")
        self.threads = int(threads)
        if name:
            self.name = name
        elif "int8" in self.path.name:
            self.name = "onnx-int8"

    def warmup(self, sizes: tuple[int, ...] = (2, 10, 40), iters: int = 3) -> None:
        """여러 N 으로 미리 돌려 동적 축 관련 첫 호출 비용을 없앤다."""
        rng = np.random.default_rng(0)
        for n in sizes:
            obs = np.cumsum(rng.normal(0, 0.3, size=(n, self.obs_len, 2)), axis=1)
            for _ in range(iters):
                self.predict(obs, k=0)

    def run_raw(self, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"v": v, "a": a})[0]

    def predict(self, obs_abs: np.ndarray, k: int = 0, seed: int | None = None) -> Prediction:
        t0 = time.perf_counter()
        v, a = preprocess(obs_abs, self.kernel, self.normalize)
        t1 = time.perf_counter()
        out = self.run_raw(v, a)  # (1, 5, T_pred, N)
        params = torch.from_numpy(
            np.ascontiguousarray(out.transpose(0, 2, 3, 1)[0])
        )  # (T_pred, N, 5)
        t2 = time.perf_counter()
        return finalize(
            params,
            obs_abs,
            k,
            seed,
            {"preprocess_ms": (t1 - t0) * 1e3, "model_ms": (t2 - t1) * 1e3},
        )
