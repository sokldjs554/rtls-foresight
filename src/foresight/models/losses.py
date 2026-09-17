"""이변량 가우시안 음의 로그가능도 (Social-LSTM / Social-STGCNN 손실).

출력 5채널 = (μx, μy, log σx, log σy, atanh ρ). ``exact=True`` 는 공식 코드와 수치까지 같은 경로
(pdf 계산 후 1e-20 로 클램프 → -log) 이고, ``exact=False`` 는 로그 영역에서 직접 계산하는
수치적으로 안정한 버전이다(ablation 용).
"""

from __future__ import annotations

import math

import torch


def split_params(
    out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(..., 5)`` → (μx, μy, σx, σy, ρ)."""
    mux, muy = out[..., 0], out[..., 1]
    sx, sy = torch.exp(out[..., 2]), torch.exp(out[..., 3])
    corr = torch.tanh(out[..., 4])
    return mux, muy, sx, sy, corr


def bivariate_nll(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, exact: bool = True
) -> torch.Tensor:
    """pred ``(..., 5)``, target ``(..., 2)`` → 스칼라 평균 NLL. mask ``(...)`` 가 있으면 유효 항만 평균."""
    mux, muy, sx, sy, corr = split_params(pred)
    normx = target[..., 0] - mux
    normy = target[..., 1] - muy
    sxsy = sx * sy
    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * (corr * normx * normy) / sxsy
    neg_rho = 1 - corr**2
    if exact:
        result = torch.exp(-z / (2 * neg_rho))
        denom = 2 * math.pi * (sxsy * torch.sqrt(neg_rho))
        result = result / denom
        nll = -torch.log(torch.clamp(result, min=1e-20))
    else:
        nll = z / (2 * neg_rho) + torch.log(sxsy) + 0.5 * torch.log(neg_rho) + math.log(2 * math.pi)
    if mask is None:
        return nll.mean()
    m = mask.to(nll.dtype)
    return (nll * m).sum() / m.sum().clamp_min(1.0)
