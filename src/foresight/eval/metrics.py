"""ADE / FDE 와 샘플링 프로토콜.

세 가지 프로토콜을 모두 계산한다. 숫자가 어떻게 달라지는지가 이 프로젝트의 "평가 방법론" 논의다.

* ``best_of_k_per_agent`` — **논문/공식 코드 프로토콜**. 20개 샘플 중 보행자마다 가장 좋은 샘플을 골라 평균.
  같은 장면의 보행자들이 서로 다른 샘플을 고를 수 있으므로 실제로는 존재하지 않는 "조합"을 평가하는 셈이다.
* ``best_of_k_joint`` — 장면 단위로 하나의 샘플(모든 보행자 공통)을 고른다. 다중 에이전트 예측이라면 이쪽이
  현실적이다 (충돌 위험은 한 미래 안에서 두 사람의 위치를 같이 봐야 하므로).
* ``deterministic`` — 샘플 없이 분포의 평균 μ 를 쓴다. 서비스에서 "한 번의 예측"을 보여줄 때의 성능.

ADE = 예측 구간 12 스텝의 평균 L2 오차(m), FDE = 마지막 스텝 오차(m). 집계는 공식 코드처럼
**보행자 단위 평균** (장면 크기와 무관하게 모든 보행자가 같은 가중치).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from foresight.models.losses import split_params


def build_covariance(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(..., 5)`` → (mean ``(..., 2)``, cov ``(..., 2, 2)``)."""
    mux, muy, sx, sy, corr = split_params(params)
    cov = torch.zeros(*params.shape[:-1], 2, 2, dtype=params.dtype, device=params.device)
    cov[..., 0, 0] = sx * sx
    cov[..., 0, 1] = corr * sx * sy
    cov[..., 1, 0] = corr * sx * sy
    cov[..., 1, 1] = sy * sy
    mean = torch.stack([mux, muy], dim=-1)
    return mean, cov


def sample_relative(
    params: torch.Tensor, k: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """``(T, N, 5)`` 상대 변위 분포에서 K 개 샘플 ``(K, T, N, 2)`` (공식 코드와 같은 MultivariateNormal)."""
    mean, cov = build_covariance(params)
    dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
    if generator is None:
        return dist.sample((k,))
    # torch.distributions 는 generator 인자를 받지 않으므로 표준정규 샘플을 직접 뽑아 변환한다.
    scale_tril = torch.linalg.cholesky(cov)
    eps = torch.randn((k, *mean.shape), generator=generator, dtype=mean.dtype)
    return mean.unsqueeze(0) + (scale_tril.unsqueeze(0) @ eps.unsqueeze(-1)).squeeze(-1)


def relative_to_absolute(rel: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
    """``(..., T, N, 2)`` 상대 변위 → 절대좌표 (``last_obs`` ``(N, 2)`` 기준 누적합)."""
    return torch.cumsum(rel, dim=-3) + last_obs


@dataclass
class DisplacementErrors:
    ade: np.ndarray  # (N,) 보행자별
    fde: np.ndarray  # (N,)


def errors_per_agent(
    pred_abs: torch.Tensor, gt_abs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """pred ``(K, T, N, 2)`` vs gt ``(T, N, 2)`` → ade ``(K, N)``, fde ``(K, N)``."""
    d = torch.linalg.norm(pred_abs - gt_abs.unsqueeze(0), dim=-1)  # (K, T, N)
    return d.mean(dim=1), d[:, -1]


def best_of_k_per_agent(pred_abs: torch.Tensor, gt_abs: torch.Tensor) -> DisplacementErrors:
    ade, fde = errors_per_agent(pred_abs, gt_abs)
    return DisplacementErrors(ade.min(dim=0).values.numpy(), fde.min(dim=0).values.numpy())


def best_of_k_joint(pred_abs: torch.Tensor, gt_abs: torch.Tensor) -> DisplacementErrors:
    """장면 평균 ADE 가 가장 좋은 샘플 하나를 고르고 그 샘플의 ADE·FDE 를 모두 보고한다.

    FDE 를 따로 최소화하지 않는 것은 의도다: "하나의 미래"를 고른 뒤 그 미래의 두 지표를 읽어야 joint 다.
    """
    ade, fde = errors_per_agent(pred_abs, gt_abs)
    k = ade.mean(dim=1).argmin()  # 장면 평균 ADE 가 가장 좋은 샘플 하나
    return DisplacementErrors(ade[k].numpy(), fde[k].numpy())


def deterministic(
    params: torch.Tensor, last_obs: torch.Tensor, gt_abs: torch.Tensor
) -> DisplacementErrors:
    mean = params[..., :2]
    pred_abs = relative_to_absolute(mean.unsqueeze(0), last_obs)
    ade, fde = errors_per_agent(pred_abs, gt_abs)
    return DisplacementErrors(ade[0].numpy(), fde[0].numpy())


def summarize(errs: list[DisplacementErrors]) -> dict[str, float]:
    ade = np.concatenate([e.ade for e in errs])
    fde = np.concatenate([e.fde for e in errs])
    return {"ade": float(ade.mean()), "fde": float(fde.mean()), "n_agents": len(ade)}
