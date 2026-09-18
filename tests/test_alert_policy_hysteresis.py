"""경보 정책: 히스테리시스 밴드 안에서는 새 경보를 내지 않는다 (카운터만 유지)."""

from __future__ import annotations

import math

from foresight.serving.risk import AlertPolicy, PairRisk


def _obs(r: float) -> list[tuple[str, str, PairRisk]]:
    return [("w1", "v1", PairRisk(worker=0, vehicle=1, risk=r, ttc_s=math.nan, min_dist_mean=2.0))]


def test_no_alert_inside_hysteresis_band() -> None:
    pol = AlertPolicy(threshold=0.3, cooldown_s=0.0, min_consecutive=2, clear_threshold=0.15)
    assert pol.update(_obs(0.5), now_s=0.0) == []  # 1번째 초과
    assert len(pol.update(_obs(0.5), now_s=0.4)) == 1  # 2번째 → 경보
    # 밴드(0.15 ≤ r < 0.3): 카운터는 유지하지만 경보는 없다
    assert pol.update(_obs(0.2), now_s=0.8) == []
    assert pol.state_of("w1", "v1")[0] >= 2
    # 다시 임계 초과 → 경보 (쿨다운 0)
    assert len(pol.update(_obs(0.4), now_s=1.2)) == 1
    # clear 아래 → 리셋
    pol.update(_obs(0.05), now_s=1.6)
    assert pol.state_of("w1", "v1")[0] == 0
