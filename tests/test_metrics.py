"""ADE/FDE 프로토콜과 샘플링."""

from __future__ import annotations

import numpy as np
import torch

from foresight.eval import metrics as M
from foresight.models import constant_velocity


def test_perfect_prediction_has_zero_error() -> None:
    gt = torch.cumsum(torch.ones(12, 3, 2) * 0.1, dim=0)
    pred = gt.unsqueeze(0).repeat(5, 1, 1, 1)
    e = M.best_of_k_per_agent(pred, gt)
    assert np.allclose(e.ade, 0) and np.allclose(e.fde, 0)


def test_per_agent_best_is_never_worse_than_joint() -> None:
    torch.manual_seed(0)
    gt = torch.randn(12, 4, 2)
    pred = torch.randn(20, 12, 4, 2)
    pa, jo = M.best_of_k_per_agent(pred, gt), M.best_of_k_joint(pred, gt)
    assert pa.ade.mean() <= jo.ade.mean() + 1e-6
    assert pa.fde.mean() <= jo.fde.mean() + 1e-6


def test_sampling_generator_is_reproducible_and_matches_distribution() -> None:
    params = torch.zeros(12, 2, 5)
    params[..., 2:4] = np.log(0.5)  # σ = 0.5
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    s1, s2 = M.sample_relative(params, 3, g1), M.sample_relative(params, 3, g2)
    torch.testing.assert_close(s1, s2)
    big = M.sample_relative(params, 4000, torch.Generator().manual_seed(0))
    assert abs(big.std().item() - 0.5) < 0.02


def test_relative_to_absolute_cumsum() -> None:
    rel = torch.ones(1, 12, 2, 2) * 0.1
    last = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    abs_ = M.relative_to_absolute(rel, last)
    assert torch.allclose(abs_[0, -1, 0], torch.tensor([2.2, 3.2]))


def test_constant_velocity_extrapolates() -> None:
    obs = np.stack([np.arange(8) * 0.5, np.zeros(8)], axis=-1)[None]  # 한 명, x 방향 0.5/step
    pred = constant_velocity(obs, pred_len=12, k=1)
    assert pred.shape == (1, 1, 12, 2)
    assert np.allclose(pred[0, 0, -1], [3.5 + 6.0, 0.0])
    sampled = constant_velocity(
        obs, pred_len=12, k=20, angle_std_deg=25.0, rng=np.random.default_rng(0)
    )
    assert sampled.shape == (20, 1, 12, 2)
    assert np.allclose(np.linalg.norm(sampled[:, 0, 0] - obs[0, -1], axis=-1), 0.5)  # 속력 보존
