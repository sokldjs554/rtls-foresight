"""충돌 위험 점수 + 경보 정책 — 답을 아는 합성 직선 궤적으로 검증."""

from __future__ import annotations

import numpy as np
import pytest

from foresight.serving.risk import (
    STEP_SECONDS,
    AlertPolicy,
    PairRisk,
    pairwise_risk,
    risk_deterministic,
)

T = 12


def straight(
    x0: float, y0: float, vx: float, vy: float, t: int = T, dt: float = STEP_SECONDS
) -> np.ndarray:
    ts = np.arange(1, t + 1) * dt
    return np.stack([x0 + vx * ts, y0 + vy * ts], axis=-1)  # (T, 2)


def make_samples(
    trajs: list[np.ndarray], k: int = 20, noise: float = 0.0, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.stack(trajs)  # (N, T, 2)
    return base[None] + rng.normal(0, noise, size=(k, *base.shape))


def test_head_on_approach_has_risk_one_and_sensible_ttc() -> None:
    worker = straight(-2.0, 0.0, 1.0, 0.0)  # 오른쪽으로 1 m/s
    vehicle = straight(2.0, 0.0, -1.0, 0.0)  # 왼쪽으로 1 m/s → 상대속도 2 m/s, 4 m 간격
    rm = pairwise_risk(make_samples([worker, vehicle], noise=0.02), np.array([0, 1]), d_safe=1.0)
    assert rm.risk.shape == (1, 1)
    assert rm.risk[0, 0] == pytest.approx(1.0)
    # 거리 4 - 2t < 1 → t > 1.5 s → 첫 스텝 1.6 s (0.4 s 격자)
    assert rm.ttc_s[0, 0] == pytest.approx(1.6, abs=0.2)
    assert rm.min_dist_mean[0, 0] < 0.5


def test_parallel_far_apart_has_zero_risk_and_nan_ttc() -> None:
    worker = straight(0.0, 0.0, 1.0, 0.0)
    vehicle = straight(0.0, 5.0, 1.0, 0.0)  # 5 m 옆에서 나란히
    rm = pairwise_risk(make_samples([worker, vehicle], noise=0.05), np.array([0, 1]), d_safe=1.0)
    assert rm.risk[0, 0] == 0.0
    assert np.isnan(rm.ttc_s[0, 0])
    assert rm.min_dist_mean[0, 0] == pytest.approx(5.0, abs=0.2)


def test_partial_risk_counts_joint_samples_not_cross_product() -> None:
    """샘플 k 끼리 짝지어야 한다: 절반의 미래에서만 충돌하면 risk=0.5."""
    worker = straight(0.0, 0.0, 0.0, 0.0)
    vehicle = straight(3.0, 0.0, 0.0, 0.0)
    s = make_samples([worker, vehicle], k=10)
    s[5:, 1, :, 0] = 0.2  # 차량 샘플 5~9 만 작업자 위로 옮긴다
    rm = pairwise_risk(s, np.array([0, 1]), d_safe=1.0)
    assert rm.risk[0, 0] == pytest.approx(0.5)
    assert rm.ttc_s[0, 0] == pytest.approx(STEP_SECONDS)  # 충돌 샘플은 1 스텝부터 가깝다


def test_matrix_shape_and_pairs_only_worker_vehicle() -> None:
    trajs = [straight(i * 10.0, 0, 0, 0) for i in range(5)]
    types = np.array([0, 1, 0, 1, 1])
    rm = pairwise_risk(make_samples(trajs), types, d_safe=1.0)
    assert rm.risk.shape == (2, 3)
    assert set(rm.worker_idx.tolist()) == {0, 2} and set(rm.vehicle_idx.tolist()) == {1, 3, 4}
    assert len(rm.pairs()) == 6 and len(rm.pairs(min_risk=0.5)) == 0


def test_no_vehicles_gives_empty_matrix() -> None:
    rm = pairwise_risk(make_samples([straight(0, 0, 1, 0)]), np.array([0]), d_safe=1.0)
    assert rm.empty and rm.pairs() == []


def test_deterministic_variant_matches_noise_free_mc() -> None:
    worker = straight(-2.0, 0.0, 1.0, 0.0)
    vehicle = straight(2.0, 0.0, -1.0, 0.0)
    det = risk_deterministic(np.stack([worker, vehicle]), np.array([0, 1]), d_safe=1.0)
    mc = pairwise_risk(make_samples([worker, vehicle], noise=0.0), np.array([0, 1]), d_safe=1.0)
    assert det.risk[0, 0] == 1.0 and det.k == 1
    assert det.ttc_s[0, 0] == pytest.approx(mc.ttc_s[0, 0])
    assert det.min_dist_mean[0, 0] == pytest.approx(mc.min_dist_mean[0, 0])


# ----------------------------------------------------------------------------- AlertPolicy
def _pr(risk: float) -> PairRisk:
    return PairRisk(0, 1, risk, 1.2, 0.5)


def test_policy_requires_consecutive_frames_then_cooldown() -> None:
    pol = AlertPolicy(threshold=0.3, cooldown_s=5.0, min_consecutive=2)
    assert pol.update([("w", "v", _pr(0.9))], now_s=0.0) == []  # 1 프레임 — 아직
    alerts = pol.update([("w", "v", _pr(0.9))], now_s=0.4)  # 2 연속 → 경보
    assert len(alerts) == 1 and alerts[0].worker_id == "w" and alerts[0].consecutive == 2
    assert pol.update([("w", "v", _pr(0.9))], now_s=0.8) == []  # 쿨다운 중
    assert pol.n_suppressed == 1
    assert len(pol.update([("w", "v", _pr(0.9))], now_s=5.5)) == 1  # 쿨다운 지남
    assert pol.n_alerts == 2


def test_policy_single_spike_is_suppressed() -> None:
    pol = AlertPolicy(threshold=0.3, min_consecutive=2)
    assert pol.update([("w", "v", _pr(0.95))], 0.0) == []
    assert pol.update([("w", "v", _pr(0.05))], 0.4) == []  # clear 아래 → 리셋
    assert pol.update([("w", "v", _pr(0.95))], 0.8) == []  # 다시 1 프레임뿐
    assert pol.state_of("w", "v")[0] == 1


def test_policy_hysteresis_band_keeps_counter() -> None:
    pol = AlertPolicy(threshold=0.3, min_consecutive=3, clear_threshold=0.1)
    pol.update([("w", "v", _pr(0.5))], 0.0)
    pol.update([("w", "v", _pr(0.2))], 0.4)  # 0.1 ≤ 0.2 < 0.3 → 카운터 유지(1)
    assert pol.state_of("w", "v")[0] == 1
    pol.update([("w", "v", _pr(0.5))], 0.8)
    assert len(pol.update([("w", "v", _pr(0.5))], 1.2)) == 1


def test_policy_pairs_are_independent_and_gc_drops_stale() -> None:
    pol = AlertPolicy(threshold=0.3, min_consecutive=1, cooldown_s=0.0, stale_after_s=10.0)
    a = pol.update([("w1", "v", _pr(0.9)), ("w2", "v", _pr(0.1))], 0.0)
    assert [x.worker_id for x in a] == ["w1"]
    assert len(pol) == 2
    assert pol.gc(now_s=20.0) == 2 and len(pol) == 0


def test_policy_validation() -> None:
    with pytest.raises(ValueError):
        AlertPolicy(threshold=0.3, clear_threshold=0.5)
    with pytest.raises(ValueError):
        AlertPolicy(min_consecutive=0)
