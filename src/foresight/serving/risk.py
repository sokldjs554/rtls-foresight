"""충돌 위험 점수 + 경보 정책.

예측 분포에서 "작업자–차량 쌍이 4.8 초 안에 ``d_safe`` 보다 가까워질 확률"을 구한다.

정의 (DESIGN.md 인터페이스 계약)
    risk(w, v) = P( min_t ||x_w(t) − x_v(t)|| < d_safe ),  t = 1..T_pred (0.4 s 간격)

MC 추정은 **같은 샘플 인덱스 k 끼리** 짝을 짓는다 — 샘플 k 는 모델이 한 번에 뽑은 "하나의 미래"이므로
작업자 샘플 k 와 차량 샘플 k 를 같은 미래로 보는 것이 맞다. (작업자 20 × 차량 20 = 400 조합을 보면
서로 다른 미래의 위치를 섞어 비교하게 되어 위험을 과대/과소 추정한다.) 이 점이 ``best_of_k_joint``
프로토콜을 별도로 두는 이유이기도 하다.

부가 지표
* ``ttc_s``: 충돌한 샘플들에서 **처음** d < d_safe 가 되는 시각의 평균 (초). 충돌 조건부이므로 risk=0 이면 NaN.
* ``min_dist_mean``: 샘플별 최소 거리의 평균 (m). 경보 임계 근처의 "얼마나 아슬아슬한지"를 보여 준다.
* 결정적 변형 ``risk_deterministic``: 평균 궤적(μ)만 써서 같은 지표를 계산 (risk ∈ {0, 1}). 샘플링이 없어
  빠르고, 예측 분포가 좁을 때(장비가 직진 중) MC 와 거의 같은 답을 준다 — 서비스에서 K 를 줄일 근거.

경보 정책 ``AlertPolicy``
    RTLS 는 0.4 s 마다 새 예측이 나오므로 단발 임계 초과만으로 경보하면 오경보가 잦다. 이전 PPE 검출
    프로젝트에서 검증한 "연속 N 프레임 + 쿨다운" 규칙을 쌍 단위 상태로 옮겼다.
    * ``min_consecutive``: 연속으로 임계를 넘은 프레임 수가 이 값 이상일 때만 경보 (한 프레임 튐 억제).
    * ``clear_threshold`` (히스테리시스): 임계 바로 아래로 잠깐 내려가도 카운터를 유지하고, 이 값 아래로
      떨어져야 리셋한다. 임계 근처에서 진동하는 쌍이 카운터를 매번 잃지 않도록.
    * ``cooldown_s``: 한 쌍에 경보를 낸 뒤 이 시간 동안은 같은 쌍에 다시 경보하지 않는다 (알림 폭주 방지).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from foresight.inference.predictor import Prediction

WORKER = 0
VEHICLE = 1
STEP_SECONDS = 0.4  # ETH/UCY 2.5 fps = RTLS 리샘플 주기


@dataclass
class PairRisk:
    worker: int  # 장면 내 인덱스
    vehicle: int
    risk: float
    ttc_s: float  # 충돌 조건부 기대 최초 접근 시각(초); risk=0 이면 nan
    min_dist_mean: float


@dataclass
class RiskMatrix:
    """(작업자 W × 차량 V) 행렬 형태의 결과. 인덱스는 장면 내 에이전트 인덱스."""

    worker_idx: np.ndarray  # (W,)
    vehicle_idx: np.ndarray  # (V,)
    risk: np.ndarray  # (W, V)
    ttc_s: np.ndarray  # (W, V) nan = 충돌 샘플 없음
    min_dist_mean: np.ndarray  # (W, V)
    k: int = 0

    def pairs(self, min_risk: float = 0.0) -> list[PairRisk]:
        out: list[PairRisk] = []
        for wi, w in enumerate(self.worker_idx):
            for vi, v in enumerate(self.vehicle_idx):
                r = float(self.risk[wi, vi])
                if r >= min_risk:
                    out.append(
                        PairRisk(
                            int(w),
                            int(v),
                            r,
                            float(self.ttc_s[wi, vi]),
                            float(self.min_dist_mean[wi, vi]),
                        )
                    )
        return out

    @property
    def empty(self) -> bool:
        return self.risk.size == 0


def _empty(worker_idx: np.ndarray, vehicle_idx: np.ndarray, k: int) -> RiskMatrix:
    shape = (len(worker_idx), len(vehicle_idx))
    return RiskMatrix(
        worker_idx, vehicle_idx, np.zeros(shape), np.full(shape, np.nan), np.full(shape, np.inf), k
    )


def pairwise_risk(
    samples_abs: np.ndarray, types: np.ndarray, d_safe: float = 1.0, dt: float = STEP_SECONDS
) -> RiskMatrix:
    """``samples_abs (K, N, T, 2)`` 절대좌표 샘플 + ``types (N,)`` → 작업자×차량 위험 행렬.

    (K, W, V, T) 거리 텐서를 한 번에 만든다: K=20, W=V=100, T=12 여도 ~20 MB 라 numpy 브로드캐스트가
    파이썬 이중 루프보다 두 자릿수 빠르다.
    """
    samples_abs = np.asarray(samples_abs, dtype=np.float64)
    if samples_abs.ndim == 3:  # (N, T, 2) 단일 궤적 → K=1
        samples_abs = samples_abs[None]
    types = np.asarray(types)
    w_idx = np.flatnonzero(types == WORKER)
    v_idx = np.flatnonzero(types == VEHICLE)
    k = samples_abs.shape[0]
    if len(w_idx) == 0 or len(v_idx) == 0:
        return _empty(w_idx, v_idx, k)
    diff = samples_abs[:, w_idx, None, :, :] - samples_abs[:, None, v_idx, :, :]  # (K, W, V, T, 2)
    dist = np.sqrt((diff**2).sum(-1))  # (K, W, V, T)
    close = dist < d_safe
    hit = close.any(-1)  # (K, W, V)
    risk = hit.mean(0)
    # 최초 접근 스텝: argmax 는 첫 True 를 돌려준다 (hit 가 False 인 샘플은 마스크로 제외)
    first_step = close.argmax(-1) + 1  # 1..T → 초 단위 t·dt
    # 충돌 조건부 평균: 충돌한 샘플의 최초 접근 시각 합 / 충돌 샘플 수 (충돌 샘플이 없으면 nan)
    n_hit = hit.sum(0)
    ttc_sum = np.where(hit, first_step * dt, 0.0).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(n_hit > 0, ttc_sum / np.maximum(n_hit, 1), np.nan)
    min_dist_mean = dist.min(-1).mean(0)
    return RiskMatrix(w_idx, v_idx, risk, ttc, min_dist_mean, k)


def risk_deterministic(
    mean_abs: np.ndarray, types: np.ndarray, d_safe: float = 1.0, dt: float = STEP_SECONDS
) -> RiskMatrix:
    """평균 궤적 ``(N, T, 2)`` 만으로 계산한 결정적 위험 (risk ∈ {0, 1})."""
    return pairwise_risk(np.asarray(mean_abs)[None], types, d_safe, dt)


def risk_from_prediction(
    pred: Prediction, types: np.ndarray, d_safe: float = 1.0, deterministic: bool = False
) -> RiskMatrix:
    """``Prediction`` 편의 래퍼: 샘플이 없으면 자동으로 결정적 경로를 쓴다."""
    if deterministic or pred.samples_abs is None:
        return risk_deterministic(pred.mean_abs, types, d_safe)
    return pairwise_risk(pred.samples_abs, types, d_safe)


@dataclass
class Alert:
    worker_id: str
    vehicle_id: str
    risk: float
    ttc_s: float
    min_dist_mean: float
    ts_s: float
    zone_id: int | None = None
    consecutive: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "worker_id": self.worker_id,
            "vehicle_id": self.vehicle_id,
            "risk": round(self.risk, 4),
            "ttc_s": None if np.isnan(self.ttc_s) else round(self.ttc_s, 2),
            "min_dist_mean": round(self.min_dist_mean, 3),
            "ts_s": round(self.ts_s, 3),
            "zone_id": self.zone_id,
            "consecutive": self.consecutive,
        }


@dataclass
class _PairState:
    consecutive: int = 0
    last_alert_s: float = -np.inf
    last_seen_s: float = -np.inf


@dataclass
class AlertPolicy:
    """쌍 단위 상태를 가진 경보 정책 (연속 N 프레임 + 히스테리시스 + 쿨다운)."""

    threshold: float = 0.3
    cooldown_s: float = 5.0
    min_consecutive: int = 2
    clear_threshold: float | None = None  # 기본 threshold/2
    stale_after_s: float = 60.0  # 이 시간 동안 안 보인 쌍의 상태는 버린다 (메모리 상한)
    _state: dict[tuple[str, str], _PairState] = field(default_factory=dict, repr=False)
    n_alerts: int = 0
    n_suppressed: int = 0

    def __post_init__(self) -> None:
        if self.clear_threshold is None:
            self.clear_threshold = self.threshold / 2
        if not (0 <= self.clear_threshold <= self.threshold <= 1):
            raise ValueError("0 <= clear_threshold <= threshold <= 1 이어야 한다")
        if self.min_consecutive < 1:
            raise ValueError("min_consecutive >= 1")

    def update(
        self,
        observations: list[tuple[str, str, PairRisk]],
        now_s: float,
        zone_id: int | None = None,
    ) -> list[Alert]:
        """이번 프레임의 (worker_id, vehicle_id, PairRisk) 목록을 반영하고 새 경보를 돌려준다.

        관측에 없는 쌍은 상태를 건드리지 않는다 (같은 구역에서 잠깐 태그가 끊겨도 카운터 유지).
        """
        alerts: list[Alert] = []
        assert self.clear_threshold is not None
        for wid, vid, pr in observations:
            st = self._state.setdefault((wid, vid), _PairState())
            st.last_seen_s = now_s
            if pr.risk >= self.threshold:
                st.consecutive += 1
            elif pr.risk < self.clear_threshold:
                st.consecutive = 0
            # clear_threshold <= risk < threshold : 카운터 유지 (히스테리시스 밴드)
            if st.consecutive >= self.min_consecutive and pr.risk >= self.threshold:
                # 히스테리시스 밴드(clear ≤ risk < threshold)에서는 카운터만 유지하고 새 경보는 내지 않는다
                if now_s - st.last_alert_s >= self.cooldown_s:
                    st.last_alert_s = now_s
                    self.n_alerts += 1
                    alerts.append(
                        Alert(
                            wid,
                            vid,
                            pr.risk,
                            pr.ttc_s,
                            pr.min_dist_mean,
                            now_s,
                            zone_id,
                            st.consecutive,
                        )
                    )
                else:
                    self.n_suppressed += 1
        return alerts

    def gc(self, now_s: float) -> int:
        """오래 안 보인 쌍의 상태를 지운다. 지운 개수를 돌려준다."""
        stale = [k for k, st in self._state.items() if now_s - st.last_seen_s > self.stale_after_s]
        for k in stale:
            del self._state[k]
        return len(stale)

    def state_of(self, worker_id: str, vehicle_id: str) -> tuple[int, float]:
        st = self._state.get((worker_id, vehicle_id))
        return (st.consecutive, st.last_alert_s) if st else (0, -np.inf)

    def __len__(self) -> int:
        return len(self._state)
