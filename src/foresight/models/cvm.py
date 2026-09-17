"""Constant Velocity Model (Schöller et al., RA-L 2020) — 학습 없는 기준선.

마지막 관측 변위를 12 스텝 반복한다. ``angle_std_deg > 0`` 이면 방향을 정규분포 각도로 흔든 K 개
샘플을 만든다(논문의 OUR-S, σ=25°). 딥러닝 모델의 best-of-20 ADE/FDE 가 이 기준선을 얼마나 이기는지가
"학습된 것이 있는가"의 가장 정직한 척도다.
"""

from __future__ import annotations

import numpy as np


def constant_velocity(
    obs_abs: np.ndarray,
    pred_len: int = 12,
    k: int = 1,
    angle_std_deg: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """obs_abs ``(N, T_obs, 2)`` → ``(K, N, pred_len, 2)`` 절대좌표 예측."""
    rng = rng or np.random.default_rng(0)
    delta = obs_abs[:, -1] - obs_abs[:, -2]  # (N, 2)
    last = obs_abs[:, -1]
    out = np.empty((k, obs_abs.shape[0], pred_len, 2), dtype=np.float64)
    for i in range(k):
        d = delta
        if angle_std_deg > 0:
            theta = np.deg2rad(rng.normal(0.0, angle_std_deg))
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array([[c, s], [-s, c]])
            d = delta @ rot.T
        steps = np.arange(1, pred_len + 1)[None, :, None]
        out[i] = last[:, None, :] + d[:, None, :] * steps
    return out
