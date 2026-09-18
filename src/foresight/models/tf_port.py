"""Social-STGCNN 의 TensorFlow 이식 — 같은 가중치, 같은 수식, SavedModel 로 내보내기.

서빙 팀이 TF Serving/TFLite 를 쓰는 경우를 위한 경로다. PyTorch 학습 결과(``export_weights``)를 받아 tf 연산으로
순전파를 다시 구성한다. 학습은 하지 않는다(가중치 동결). ``tests/test_tf_parity.py`` 가 PyTorch 출력과 1e-4 안에서 같은지 확인한다.

구현 메모
* CPU 의 tf.nn.conv2d 는 NHWC 만 지원하므로 내부 배치는 NHWC 로 두고, 공식 코드의 ``view`` 축 교환은
  NCHW 로 돌린 뒤 ``tf.reshape`` 로 메모리를 재해석한다(행 우선이라 torch ``view`` 와 같다).
* BatchNorm 은 eval 모드 아핀 변환으로 접는다. PReLU 는 스칼라 기울기.
* 마지막 TXP-CNN 층과 PReLU 는 공식 코드처럼 쓰지 않는다(``for k in range(1, n_txpcnn - 1)``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _tf() -> Any:
    try:
        import tensorflow as tf
    except ImportError as e:  # pragma: no cover - 선택 의존성
        raise ImportError("tensorflow 가 필요합니다: pip install 'rtls-foresight[tf]'") from e
    return tf


def _bn_affine(bn: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    w, b, m, v = (np.asarray(bn[k], dtype=np.float32) for k in ("w", "b", "m", "v"))
    scale = w / np.sqrt(v + np.float32(bn["eps"]))
    return scale, b - m * scale


class SocialSTGCNNTF:
    """``export_weights`` 사전으로 만든 TensorFlow 순전파. ``__call__(v, a)`` → params ``(1, T_pred, N, 5)``."""

    def __init__(self, weights: dict[str, Any]) -> None:
        tf = _tf()
        meta = weights["meta"]
        if meta.get("time_channel_swap", "view") != "view":
            raise ValueError("TF 이식은 time_channel_swap='view' 만 지원합니다")
        self.obs_len, self.pred_len = int(meta["obs_len"]), int(meta["pred_len"])
        self.c_in, self.c_out = int(meta["in_channels"]), int(meta["out_channels"])
        st = weights["st_gcn"]
        f32 = lambda x: tf.constant(np.asarray(x, dtype=np.float32))  # noqa: E731
        # 1x1 conv (C, Cin) → NHWC 커널 (1, 1, Cin, C)
        self.gcn_w = f32(np.asarray(st["gcn_w"], dtype=np.float32).T[None, None])
        self.gcn_b = f32(st["gcn_b"])
        self.bn1 = tuple(f32(x) for x in _bn_affine(st["bn1"]))
        self.prelu1 = float(st["prelu1"])
        # 시간축 conv (C_out, C_in, kt) → (kt, 1, C_in, C_out)
        tcn = np.asarray(st["tcn_w"], dtype=np.float32)
        self.t_kernel = tcn.shape[2]
        self.tcn_w = f32(np.transpose(tcn, (2, 0, 1))[:, None, :, :].transpose(0, 1, 3, 2))
        self.tcn_b = f32(st["tcn_b"])
        self.bn2 = tuple(f32(x) for x in _bn_affine(st["bn2"]))
        self.res_kind = st.get("res_kind", "zero")
        if self.res_kind == "conv":
            self.res_w = f32(np.asarray(st["res_w"], dtype=np.float32).T[None, None])
            self.res_b = f32(st["res_b"])
            self.res_bn = tuple(f32(x) for x in _bn_affine(st["res_bn"]))
        self.prelu_out = float(st["prelu_out"])
        # TXP-CNN 3x3 (Cout, Cin, 3, 3) → (3, 3, Cin, Cout)
        self.tp = [
            (
                f32(np.transpose(np.asarray(layer["w"], dtype=np.float32), (2, 3, 1, 0))),
                f32(layer["b"]),
            )
            for layer in weights["tpcnns"]
        ]
        self.tp_prelus = [float(p) for p in weights["prelus"]]
        out = weights["tpcnn_output"]
        self.tp_out = (
            f32(np.transpose(np.asarray(out["w"], dtype=np.float32), (2, 3, 1, 0))),
            f32(out["b"]),
        )
        self._tf = tf

    @staticmethod
    def _prelu(x: Any, a: float) -> Any:
        tf = _tf()
        return tf.where(x > 0, x, a * x)

    def __call__(self, v: Any, a: Any) -> Any:
        """v ``(1, 2, T_obs, N)``, a ``(T_obs, N, N)`` (float32) → ``(1, T_pred, N, 5)``."""
        tf = self._tf
        v = tf.convert_to_tensor(v, tf.float32)
        a = tf.convert_to_tensor(a, tf.float32)
        x = tf.transpose(v, [0, 2, 3, 1])  # NHWC: (1, T, N, Cin)
        y = tf.nn.conv2d(x, self.gcn_w, 1, "VALID") + self.gcn_b  # (1, T, N, C)
        z = tf.einsum("ntvc,tvw->ntwc", y, a)  # 프레임별 A_t 곱 (공식 einsum 'nctv,tvw->nctw')
        z = z * self.bn1[0] + self.bn1[1]
        z = self._prelu(z, self.prelu1)
        pad = (self.t_kernel - 1) // 2
        zp = tf.pad(z, [[0, 0], [pad, pad], [0, 0], [0, 0]])
        u = tf.nn.conv2d(zp, self.tcn_w, 1, "VALID") + self.tcn_b
        u = u * self.bn2[0] + self.bn2[1]
        if self.res_kind == "conv":
            r = tf.nn.conv2d(x, self.res_w, 1, "VALID") + self.res_b
            u = u + (r * self.res_bn[0] + self.res_bn[1])
        elif self.res_kind == "identity":
            u = u + x
        h = self._prelu(u, self.prelu_out)  # (1, T, N, C)
        n = tf.shape(h)[2]
        # 공식 코드의 view: (1, C, T, N) 메모리를 (1, T, C, N) 으로 재해석
        h_nchw = tf.transpose(h, [0, 3, 1, 2])  # (1, C, T, N)
        h_view = tf.reshape(h_nchw, [1, self.obs_len, self.c_out, n])  # (1, T, C, N) = NCHW(채널 T)
        t = tf.transpose(h_view, [0, 2, 3, 1])  # NHWC: (1, C, N, T)
        w0, b0 = self.tp[0]
        t = self._prelu(tf.nn.conv2d(t, w0, 1, "SAME") + b0, self.tp_prelus[0])
        for k in range(1, len(self.tp) - 1):
            wk, bk = self.tp[k]
            t = self._prelu(tf.nn.conv2d(t, wk, 1, "SAME") + bk, self.tp_prelus[k]) + t
        wo, bo = self.tp_out
        t = tf.nn.conv2d(t, wo, 1, "SAME") + bo  # (1, C, N, T_pred)
        t_nchw = tf.transpose(t, [0, 3, 1, 2])  # (1, T_pred, C, N)
        out = tf.reshape(t_nchw, [1, self.c_out, self.pred_len, n])  # view → (1, 5, T_pred, N)
        return tf.transpose(out, [0, 2, 3, 1])  # (1, T_pred, N, 5)

    def as_module(self) -> Any:
        """``tf.Module`` (SavedModel 서명: v (1,2,T_obs,None), a (T_obs,None,None))."""
        tf = self._tf
        outer = self

        class _Module(tf.Module):  # type: ignore[name-defined]
            @tf.function(
                input_signature=[
                    tf.TensorSpec([1, outer.c_in, outer.obs_len, None], tf.float32, name="v"),
                    tf.TensorSpec([outer.obs_len, None, None], tf.float32, name="a"),
                ]
            )
            def serve(self, v: Any, a: Any) -> dict[str, Any]:
                return {"params": outer(v, a)}

        return _Module()

    def export_saved_model(self, path: str | Path) -> Path:
        tf = self._tf
        path = Path(path)
        module = self.as_module()
        tf.saved_model.save(module, str(path), signatures={"serving_default": module.serve})
        return path


def predict_params(weights: dict[str, Any], obs_abs: np.ndarray) -> np.ndarray:
    """편의 함수: 절대좌표 관측 ``(N, T_obs, 2)`` → params ``(T_pred, N, 5)`` (파이썬 전처리 + TF 순전파)."""
    from foresight.inference.predictor import preprocess

    v, a = preprocess(obs_abs)
    return SocialSTGCNNTF(weights)(v, a).numpy()[0]
