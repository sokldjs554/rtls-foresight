"""ONNX 정적 INT8 양자화 + fp32/INT8 정확도 비교.

왜 **정적**(static) 양자화인가
    ORT 의 동적 양자화(``quantize_dynamic``)는 MatMul/Gemm/LSTM 계열만 INT8 로 바꾸고 **Conv 는 그대로 둔다**.
    Social-STGCNN 은 1x1 Conv + 시간축 Conv + TXP-CNN 3x3 Conv 가 연산의 거의 전부라 동적 양자화는
    사실상 아무것도 바꾸지 않는다. 정적 양자화는 보정 데이터로 활성값 범위를 미리 재서 Conv 를
    QLinearConv(또는 QDQ 로 감싼 Conv)로 바꾼다.

보정(calibration)
    학습 분할에서 무작위 200 장면을 골라 ``preprocess`` 로 (v, a) 를 만든다 — 서빙 입력과 완전히 같은 경로.
    노드 수 N 이 장면마다 달라 텐서 모양이 제각각이므로 히스토그램 기반(Entropy/Percentile) 보정은
    ORT 구현이 ``np.array(list_of_ragged)`` 에서 실패한다. 그래서 MinMax 만 쓴다.

부분 양자화 ``nodes_to_exclude``
    ST-GCNN 블록(첫 Reshape 이전 노드들)은 입력이 상대 변위(대개 ±0.5 m) 이고 Einsum 으로 정규화
    라플라시안을 곱하는데, 여기서 8-bit 격자가 미세한 변위 차이를 뭉개 ADE 가 눈에 띄게 나빠진다.
    TXP-CNN 만 양자화하면 정확도 손실이 절반 이하로 준다 — 어느 쪽이 나은지는 ``export_all`` 이 실제
    테스트 세트로 재서 고른다.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import onnx
import torch

from foresight.data.ethucy import SceneSet
from foresight.eval import metrics as M
from foresight.inference.predictor import preprocess
from foresight.utils import get_logger

log = get_logger("foresight.inference.quantize")


def stgcnn_node_names(model: onnx.ModelProto) -> list[str]:
    """첫 번째 Reshape(= (C,T) 축 교환) 이전의 노드 이름 — ST-GCNN 블록에 해당한다."""
    names: list[str] = []
    for node in model.graph.node:
        if node.op_type == "Reshape":
            break
        names.append(node.name)
    return names


class SceneCalibrationReader:
    """``onnxruntime.quantization.CalibrationDataReader`` 구현 (지연 import 를 위해 덕타이핑)."""

    def __init__(
        self,
        scenes: SceneSet,
        indices: np.ndarray,
        kernel: Literal["velocity", "position"] = "velocity",
        normalize: bool = True,
    ) -> None:
        self.scenes, self.indices = scenes, indices
        self.kernel, self.normalize = kernel, normalize
        self._it: Iterator[int] = iter(indices.tolist())

    def get_next(self) -> dict[str, np.ndarray] | None:
        i = next(self._it, None)
        if i is None:
            return None
        v, a = preprocess(
            self.scenes.scene(i)[:, : self.scenes.obs_len], self.kernel, self.normalize
        )
        return {"v": v, "a": a}

    def rewind(self) -> None:
        self._it = iter(self.indices.tolist())


@dataclass
class QuantizeResult:
    path: Path
    variant: str  # "full" | "txp-only"
    n_calib: int
    excluded_nodes: list[str] = field(default_factory=list)
    quant_seconds: float = 0.0
    n_quantized_nodes: int = 0


def preprocess_for_quant(fp32_path: Path, out_path: Path) -> Path:
    """BN 접기·상수 접기 등 ORT 의 양자화 전처리 (``quant_pre_process``). 양자화 품질과 속도 둘 다 좋아진다."""
    from onnxruntime.quantization.shape_inference import quant_pre_process

    quant_pre_process(str(fp32_path), str(out_path), skip_symbolic_shape=False)
    return out_path


def quantize_static_scenes(
    fp32_path: Path,
    out_path: Path,
    calib: SceneSet,
    n_calib: int = 200,
    variant: str = "txp-only",
    seed: int = 0,
    quant_format: str = "QDQ",
    per_channel: bool = True,
) -> QuantizeResult:
    """정적 INT8 양자화. ``variant="full"`` 은 전체, ``"txp-only"`` 는 ST-GCNN 을 제외한다."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    pre = out_path.with_suffix(".pre.onnx")
    preprocess_for_quant(fp32_path, pre)
    model = onnx.load(str(pre))
    excluded = stgcnn_node_names(model) if variant == "txp-only" else []
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(calib), size=min(n_calib, len(calib)), replace=False)
    reader = SceneCalibrationReader(calib, idx)
    t0 = time.perf_counter()
    quantize_static(
        str(pre),
        str(out_path),
        reader,
        quant_format=QuantFormat.QDQ if quant_format == "QDQ" else QuantFormat.QOperator,
        per_channel=per_channel,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=excluded or None,
    )
    dt = time.perf_counter() - t0
    pre.unlink(missing_ok=True)
    q = onnx.load(str(out_path))
    onnx.checker.check_model(q)
    nq = sum(
        1 for n in q.graph.node if n.op_type in ("QuantizeLinear", "QLinearConv", "QLinearAdd")
    )
    log.info(
        "INT8 %s: %d calib scenes, %d excluded nodes, %d quant nodes, %.1fs -> %s",
        variant,
        len(idx),
        len(excluded),
        nq,
        dt,
        out_path,
    )
    return QuantizeResult(out_path, variant, len(idx), excluded, dt, nq)


@dataclass
class OnnxAccuracy:
    ade_det: float
    fde_det: float
    ade_bo20: float
    fde_bo20: float
    n_scenes: int
    n_agents: int
    k: int
    seed: int


def evaluate_onnx_accuracy(
    onnx_path: Path, test: SceneSet, k: int = 20, seed: int = 0, threads: int = 1
) -> OnnxAccuracy:
    """ONNX 모델의 ADE/FDE (결정적 μ 경로 + 고정 시드 best-of-K, 보행자 단위 평균)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    g = torch.Generator().manual_seed(seed)
    det: list[M.DisplacementErrors] = []
    bok: list[M.DisplacementErrors] = []
    obs_len = test.obs_len
    for i in range(len(test)):
        pos = test.scene(i)
        v, a = preprocess(pos[:, :obs_len])
        params = torch.from_numpy(sess.run(None, {"v": v, "a": a})[0]).permute(0, 2, 3, 1)[
            0
        ]  # (T_pred, N, 5)
        gt = torch.from_numpy(pos[:, obs_len:].astype(np.float32)).permute(1, 0, 2)
        last = torch.from_numpy(pos[:, obs_len - 1].astype(np.float32))
        det.append(M.deterministic(params, last, gt))
        rel = M.sample_relative(params, k, generator=g)
        bok.append(M.best_of_k_per_agent(M.relative_to_absolute(rel, last), gt))
    d, b = M.summarize(det), M.summarize(bok)
    return OnnxAccuracy(
        d["ade"], d["fde"], b["ade"], b["fde"], len(test), int(d["n_agents"]), k, seed
    )
