"""장면 → 시공간 그래프 텐서 (V, A).

Social-STGCNN 은 프레임 t 마다 보행자 노드 그래프를 만든다.

* 노드 특징 ``V[t, i] = p_i(t) - p_i(t-1)`` (상대 변위; t=0 은 0).
* 커널 ``a_ij(t) = 1 / ||v_i(t) - v_j(t)||`` (0 이면 0), 대각 ``a_ii = 1``.
  논문 본문은 "위치 간 역거리"라고 쓰지만 **공식 코드는 상대 변위(속도) 간 역거리**를 쓴다.
  재현 대상은 공식 코드이므로 여기서도 상대 변위를 쓰고, 위치 기반 커널은 ``kernel="position"``
  옵션으로 남겨 ablation 에 쓴다.
* ``A[t] = L_sym = I - D^{-1/2} A D^{-1/2}`` (networkx ``normalized_laplacian_matrix`` 와 동일, D 는 자기 루프 포함 행 합).

공식 구현(장면당 파이썬 이중 루프 + networkx)과 달리 numpy 브로드캐스트로 한 번에 계산한다.
ETH 학습 분할 기준 200 s → 0.3 s. 동일성은 테스트에서 networkx 결과와 1e-6 이내로 확인한다.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Kernel = Literal["velocity", "position"]


def relative_displacement(pos: np.ndarray) -> np.ndarray:
    """``(N, T, 2)`` 절대좌표 → 같은 모양의 상대 변위 (t=0 은 0)."""
    rel = np.zeros_like(pos)
    rel[:, 1:] = pos[:, 1:] - pos[:, :-1]
    return rel


def inverse_distance_kernel(feat: np.ndarray) -> np.ndarray:
    """``(T, N, 2)`` → ``(T, N, N)`` 역거리 커널 (대각 1, 거리 0 은 0).

    거리는 **float32 로 뺀 뒤** 제곱합하고 float64 로 제곱근을 취한다. 공식 코드가 float32 텐서를
    받아 파이썬 float 로 계산하기 때문인데, 두 보행자의 변위가 float32 에서 정확히 같으면(ETH test 에
    실제로 존재) 커널이 0 이어야 한다. float64 로 계산하면 1e-17 차이가 1e17 가중치가 되어
    라플라시안이 완전히 달라진다 — 재현 실험에서 가장 찾기 어려운 종류의 차이다.
    """
    f32 = feat.astype(np.float32)
    diff = f32[:, :, None, :] - f32[:, None, :, :]
    sq = (diff**2).sum(-1)  # float32
    dist = np.sqrt(sq.astype(np.float64))
    with np.errstate(divide="ignore"):
        a = np.where(dist > 0, 1.0 / dist, 0.0)
    n_nodes = feat.shape[1]
    idx = np.arange(n_nodes)
    a[:, idx, idx] = 1.0
    return a


def normalized_laplacian(a: np.ndarray) -> np.ndarray:
    """배치 ``(T, N, N)`` 인접행렬 → 대칭 정규화 라플라시안 ``I - D^{-1/2} A D^{-1/2}``."""
    d = a.sum(-1)  # (T, N), 자기 루프 포함
    with np.errstate(divide="ignore"):
        dinv = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    eye = np.eye(a.shape[-1])[None]
    return eye - dinv[:, :, None] * a * dinv[:, None, :]


def scene_to_graph(
    pos_abs: np.ndarray, obs_len: int, kernel: Kernel = "velocity", normalize: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """장면 절대좌표 ``(N, seq_len, 2)`` → ``(V_obs, A_obs, V_pred, A_pred)``.

    Returns:
        V_obs ``(obs_len, N, 2)``, A_obs ``(obs_len, N, N)``, V_pred ``(pred_len, N, 2)``, A_pred ``(pred_len, N, N)``.
        V_pred[0] 은 마지막 관측 위치 기준 첫 변위이므로 ``cumsum(V_pred) + pos[:, obs_len-1]`` 이 미래 절대좌표다.
    """
    rel = relative_displacement(pos_abs)  # (N, T, 2)
    v_all = rel.transpose(1, 0, 2)  # (T, N, 2)
    feat = v_all if kernel == "velocity" else pos_abs.transpose(1, 0, 2)
    a_all = inverse_distance_kernel(feat)
    if normalize:
        a_all = normalized_laplacian(a_all)
    return (
        v_all[:obs_len].astype(np.float32),
        a_all[:obs_len].astype(np.float32),
        v_all[obs_len:].astype(np.float32),
        a_all[obs_len:].astype(np.float32),
    )
