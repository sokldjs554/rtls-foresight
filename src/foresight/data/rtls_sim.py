"""합성 공장 RTLS(UWB) 위치 스트림 생성기.

왜 합성 데이터인가
------------------
실제 TPAM 류 RTLS 로그는 공개돼 있지 않다. 이 모듈은 "수천만 행 규모의 10 Hz 위치 스트림을 메모리
상한 안에서 처리한다"는 것을 보이기 위한 **입력 생성기**이며, 사람·지게차의 실제 행동을 재현한다고
주장하지 않는다(한계는 ``docs/data_pipeline.md`` 참고). 다만 하류 파이프라인이 다뤄야 하는 성질은
갖추도록 만들었다.

* 구역(zone) 격자의 통로(aisle) 를 따라 이동하는 목표지향 이동 — 작업자는 스테이션에서 10~60 s 대기,
  차량(지게차/AGV)은 헤딩 변화율이 회전반경으로 제한돼 더 부드럽게 움직인다.
* social-force 근사 반발력으로 대부분의 근접을 회피하되, 에이전트별 "주의 산만(distracted)" 에피소드
  동안 회피가 꺼져 실제 near-miss(작업자-차량 거리 < ``d_safe``) 가 일정 비율로 발생한다.
* UWB 측정 잡음(σ 0.15 m) + 드문 이상치(≤ 2 m) + 패킷 드롭아웃(iid + 버스트) + ``quality`` 지표.
* 10 Hz, 파티션 Parquet(``date=YYYY-MM-DD/hour=HH/part-k.parquet``), zstd, 5분 청크 단위 기록.

메모리 설계
-----------
시뮬레이션 상태는 에이전트 수 N 에만 비례하고(위치·속도·목표 등 (N, 2) 배열), 출력은 청크
(기본 5분 = 3,000 스텝 × N 행) 단위로 만들어 바로 디스크에 내려보낸다. 총 시간이 12시간이든
120시간이든 피크 RSS 는 같다. 시간 스텝 루프는 동역학 의존성 때문에 불가피하지만, 스텝 안의 모든
연산은 에이전트 축으로 벡터화돼 있고(쌍별 거리 (N, N)) 측정 잡음·직렬화는 청크 단위로 벡터화된다.

결정성
------
모든 난수는 하나의 ``numpy.random.Generator(seed)`` 에서 고정된 순서로 뽑으므로 같은 seed·설정이면
바이트 단위로 같은 Parquet 이 나온다(``tests/test_rtls_sim.py``).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def get_logger(name: str = "foresight.data") -> logging.Logger:
    """``foresight.utils.get_logger`` 와 같은 포맷의 로거.

    ``foresight.utils`` 는 torch 를 import 하는데(RSS +400 MB, +1 s) 데이터 경로에는 torch 가 필요 없고
    대규모 처리의 피크 RSS 측정을 왜곡하므로 여기서는 표준 logging 만 쓴다.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("FORESIGHT_LOGLEVEL", "INFO"))
        logger.propagate = False
    return logger


log = get_logger(__name__)

WORKER = 0
VEHICLE = 1

# CLI_CONTRACT.md 의 RTLS raw Parquet 스키마 — 열 순서·dtype 을 그대로 고정한다.
RAW_SCHEMA = pa.schema(
    [
        ("ts_ms", pa.int64()),
        ("tag_id", pa.int32()),
        ("agent_type", pa.int8()),
        ("zone_id", pa.int16()),
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("quality", pa.uint8()),
    ]
)
# 잡음 없는 참값 — 라벨링·평가 전용. 스트림에는 절대 섞지 않는다.
TRUTH_SCHEMA = pa.schema(
    [("ts_ms", pa.int64()), ("tag_id", pa.int32()), ("x", pa.float32()), ("y", pa.float32())]
)

PROFILES: dict[str, dict[str, float]] = {
    "smoke": {"hours": 2 / 60, "tags": 12},  # CI 용: 10 s 이내
    "small": {"hours": 1.0, "tags": 60},  # ≈ 2.2M rows
    "full": {"hours": 12.0, "tags": 200},  # ≈ 86M rows
}
MANIFEST_NAME = "_manifest.json"
TRUTH_DIR = "_truth"


@dataclass(frozen=True)
class PlantLayout:
    """공장 평면과 구역 격자.

    통로는 각 구역 셀의 중심선을 지나며 통로 교차점(node)이 이동 목표가 된다. ``zone_id`` 는
    행 우선(row-major) 셀 번호 — 파이프라인이 리샘플 뒤 같은 식으로 다시 계산하므로 여기 하나만
    바꾸면 두 곳이 같이 바뀐다.
    """

    width_m: float = 120.0
    height_m: float = 80.0
    zone_cell_m: float = 20.0
    aisle_spacing_m: float = (
        10.0  # 통로 간격 — 구역 셀보다 촘촘하다 (200명이 24개 교차로에 몰리면 정체)
    )

    @property
    def n_cols(self) -> int:
        return math.ceil(self.width_m / self.zone_cell_m)

    @property
    def n_rows(self) -> int:
        return math.ceil(self.height_m / self.zone_cell_m)

    @property
    def n_zones(self) -> int:
        return self.n_cols * self.n_rows

    @property
    def n_node_cols(self) -> int:
        return math.ceil(self.width_m / self.aisle_spacing_m)

    @property
    def n_node_rows(self) -> int:
        return math.ceil(self.height_m / self.aisle_spacing_m)

    def zone_of(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """좌표 → zone_id (int16). 평면 밖 좌표는 가장 가까운 셀로 클립한다."""
        cx = np.clip(x, 0.0, self.width_m - 1e-3)
        cy = np.clip(y, 0.0, self.height_m - 1e-3)
        col = np.floor(cx / self.zone_cell_m).astype(np.int64)
        row = np.floor(cy / self.zone_cell_m).astype(np.int64)
        return (row * self.n_cols + col).astype(np.int16)

    def node_xy(self, col: np.ndarray, row: np.ndarray) -> np.ndarray:
        """통로 교차점 좌표 ``(K, 2)``."""
        sp = self.aisle_spacing_m
        return np.stack([col * sp + sp / 2, row * sp + sp / 2], axis=-1)


@dataclass(frozen=True)
class SimConfig:
    """시뮬레이터 파라미터. 기본값은 200 태그 기준으로 near-miss 1~3 건/차량-시간이 나오도록 맞췄다."""

    hours: float = 1.0
    tags: int = 60
    seed: int = 0
    start_iso: str = "2025-03-03T00:00:00+00:00"
    hz: int = 10
    # 평면
    plant_width_m: float = 120.0
    plant_height_m: float = 80.0
    zone_cell_m: float = 20.0
    aisle_spacing_m: float = 10.0
    # 에이전트 구성
    vehicle_frac: float = 0.3
    worker_speed_mean: float = 1.3
    worker_speed_std: float = 0.25
    worker_speed_clip: tuple[float, float] = (0.5, 2.0)
    vehicle_speed_range: tuple[float, float] = (1.5, 3.0)
    vehicle_turn_radius_m: float = 3.0
    vehicle_accel_max: float = 1.5
    vehicle_brake_max: float = 3.0  # 제동은 가속보다 세다 — 운전자의 비상 정지
    vehicle_lookahead_m: float = 6.0  # 전방 원뿔 제동 거리
    # 행동
    station_prob: float = 0.2  # 작업자가 노드 도착 시 스테이션으로 빠질 확률
    idle_s: tuple[float, float] = (10.0, 60.0)
    vehicle_stop_prob: float = 0.1
    vehicle_stop_s: tuple[float, float] = (5.0, 20.0)
    # 회피 / near-miss
    d_safe: float = 1.0
    distracted_per_hour: float = 2.5  # 에이전트당 주의 산만 에피소드 발생률
    distracted_s: tuple[float, float] = (5.0, 30.0)
    # UWB 측정 모델
    noise_sigma_m: float = 0.15
    outlier_p: float = 0.002
    outlier_max_m: float = 2.0
    dropout_p: float = 0.01
    duplicate_p: float = (
        0.002  # at-least-once 전달로 생기는 중복 패킷 (파이프라인 dedupe 규칙의 존재 이유)
    )
    burst_start_p: float = 1 / 3000  # 태그·스텝당 버스트 시작 확률 (≈ 5분에 1회)
    burst_len_steps: tuple[int, int] = (10, 30)
    burst_dropout_p: float = 0.5
    # 출력
    chunk_minutes: int = 5
    write_truth: bool = True

    @property
    def layout(self) -> PlantLayout:
        return PlantLayout(
            self.plant_width_m, self.plant_height_m, self.zone_cell_m, self.aisle_spacing_m
        )

    @property
    def dt(self) -> float:
        return 1.0 / self.hz

    @property
    def start_ms(self) -> int:
        return int(datetime.fromisoformat(self.start_iso).timestamp() * 1000)


@dataclass
class SimResult:
    out_dir: Path
    rows: int
    seconds: float
    rows_per_s: float
    n_events: int
    manifest_path: Path
    config: SimConfig
    files: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)


STATE_NAMES = ("aware", "idle", "distracted")


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


class _World:
    """에이전트 상태와 한 스텝의 동역학.

    작업자가 앞, 차량이 뒤에 오도록 배열해 두면 작업자-차량 거리 행렬이 쌍별 거리 행렬의 슬라이스(뷰)가
    돼 near-miss 판정에 복사가 없다. 통로에는 우측통행 차선이 있다 — 작업자는 진행 방향 오른쪽
    2~3 m 보행로, 차량은 오른쪽 0.7 m — 그래서 정상 상황에서는 서로 다른 차선을 쓰고, near-miss 는
    주로 교차로 횡단·스테이션 이동·주의 산만 상태에서 생긴다(실제 공장의 사고 패턴과 같은 구조).
    """

    LANE_VEHICLE = 0.7
    LANE_WORKER = (1.6, 2.4)
    STATION_MIN_OFFSET = 3.0  # 통로 중심선에서 스테이션까지 최소 거리 [m]
    STUCK_STEPS = 20  # 이만큼(2 s) 멈춰 있으면 옆걸음으로 정체를 푼다
    CREEP_AFTER_STEPS = 50  # 차량이 5 s 이상 전방 제동으로 서 있으면 서행으로 진입
    PASS_BIAS_RAD = 0.35  # 반발력을 진행 방향 오른쪽으로 20° 기울여 정면 대칭을 깬다(우측 통행)
    LOOKAHEAD = 5.0  # 차량 pure pursuit 전방 거리 [m]

    def __init__(self, cfg: SimConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self.layout = cfg.layout
        n = cfg.tags
        if n < 2:
            raise ValueError("tags must be >= 2 (at least one worker and one vehicle)")
        n_veh = min(max(1, round(n * cfg.vehicle_frac)), n - 1)
        n_wk = n - n_veh
        self.n, self.n_wk, self.n_veh = n, n_wk, n_veh
        self.is_vehicle = np.zeros(n, dtype=bool)
        self.is_vehicle[n_wk:] = True
        veh = self.is_vehicle
        # 태그 ID: 1xxx 작업자, 2xxx 차량 — EDA 에서 한눈에 구분된다.
        self.tag_id = np.concatenate([1001 + np.arange(n_wk), 2001 + np.arange(n_veh)]).astype(
            np.int32
        )
        self.agent_type = veh.astype(np.int8)

        # 에이전트별 상수 (작업자/차량)
        lo, hi = cfg.worker_speed_clip
        self.speed_pref = np.where(
            veh,
            rng.uniform(*cfg.vehicle_speed_range, size=n),
            np.clip(rng.normal(cfg.worker_speed_mean, cfg.worker_speed_std, size=n), lo, hi),
        )
        self.v_max = np.where(veh, cfg.vehicle_speed_range[1], 2.2)
        self.tau = np.where(veh, 1.0, 0.5)  # 희망 속도로의 완화 시간
        # 몸체 반경: 차량은 반폭(0.6 m). d_safe=1.0 m 가 점 기준이므로 반경을 크게 잡으면
        # near-miss 가 물리적으로 불가능해진다.
        radius = np.where(veh, 0.6, 0.3)
        self.r0 = radius[:, None] + radius[None, :]  # 쌍별 접촉 거리
        self.rep_a = np.where(veh, 6.0, 8.0)  # 반발 강도 [m/s^2]
        self.rep_b = np.where(veh, 0.8, 0.8)  # 반발 감쇠 길이 [m]
        self.r_int = np.where(veh, cfg.vehicle_lookahead_m, 4.0)  # 상호작용 반경
        self.lam = np.where(veh, 0.1, 0.3)  # 시야 비등방성 λ (뒤쪽 에이전트에는 λ 만큼만 반응)
        self.sigma_acc = np.where(veh, 0.25, 0.6)  # 가속도 프로세스 잡음
        # 작업자는 점 목표(반경 0.6 m), 차량은 세그먼트 진행률로 도착을 판정한다(아래 pure pursuit).
        self.reach_tol = np.where(veh, -1.0, 0.6)
        self.lane = np.where(veh, self.LANE_VEHICLE, rng.uniform(*self.LANE_WORKER, size=n))

        # 통로 격자 상태
        self.node = np.stack(
            [
                rng.integers(0, self.layout.n_node_cols, n),
                rng.integers(0, self.layout.n_node_rows, n),
            ],
            axis=1,
        )
        self.seg_a = self.node.copy()  # 현재 통로 세그먼트의 시작 노드 (차량 pure pursuit 용)
        self.direction = np.zeros((n, 2), dtype=np.int64)
        self.pos = self.layout.node_xy(self.node[:, 0], self.node[:, 1]) + rng.normal(
            0, 1.0, (n, 2)
        )
        self.goal = np.empty((n, 2))
        self.goal_is_station = np.zeros(n, dtype=bool)
        self.idle_until = np.zeros(n, dtype=np.int64)
        self.distracted_until = np.zeros(n, dtype=np.int64)
        self.vel = np.zeros((n, 2))
        self.speed = np.zeros(n)
        self.stuck = np.zeros(n, dtype=np.int64)  # 활성인데 멈춰 있는 연속 스텝 수
        self.sidestep_until = np.zeros(n, dtype=np.int64)
        self.sidestep = np.zeros((n, 2))
        self.t = 0
        self._pick_next_node(np.arange(n))
        to_goal = self.goal - self.pos
        self.theta = np.arctan2(to_goal[:, 1], to_goal[:, 0])

        # near-miss 에피소드 상태 (작업자 × 차량)
        self.nm_active = np.zeros((n_wk, n_veh), dtype=bool)
        self.nm_start = np.zeros((n_wk, n_veh), dtype=np.int64)
        self.nm_min = np.full((n_wk, n_veh), np.inf)
        self.nm_tmin = np.zeros((n_wk, n_veh), dtype=np.int64)
        self.nm_state = np.zeros((n_wk, n_veh, 2), dtype=np.int8)
        self.events: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- 목표 선택
    def _lane_goal(self, idx: np.ndarray) -> np.ndarray:
        """현재 노드 + 진행 방향 오른쪽 차선 오프셋."""
        d = self.direction[idx]
        right = np.stack([d[:, 1], -d[:, 0]], axis=1).astype(np.float64)
        return (
            self.layout.node_xy(self.node[idx, 0], self.node[idx, 1]) + right * self.lane[idx, None]
        )

    def _pick_next_node(self, idx: np.ndarray) -> None:
        """idx 에이전트의 다음 통로 노드를 고른다. 차량은 직진 선호·역주행 회피, 작업자는 무작위."""
        rng = self.rng
        k = len(idx)
        if k == 0:
            return
        dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        node = self.node[idx]
        cand = node[:, None, :] + dirs[None, :, :]  # (k, 4, 2)
        valid = (
            (cand[:, :, 0] >= 0)
            & (cand[:, :, 0] < self.layout.n_node_cols)
            & (cand[:, :, 1] >= 0)
            & (cand[:, :, 1] < self.layout.n_node_rows)
        )
        prev = self.direction[idx]
        same = (dirs[None, :, :] == prev[:, None, :]).all(-1)  # (k, 4)
        opposite = (dirs[None, :, :] == -prev[:, None, :]).all(-1)
        veh = self.is_vehicle[idx]
        # 차량: 다른 유효 방향이 있으면 U턴 금지
        no_uturn = valid & ~opposite
        valid = np.where(veh[:, None] & no_uturn.any(1, keepdims=True), no_uturn, valid)
        keep_p = np.where(veh, 0.7, 0.4)
        keep = (rng.random(k) < keep_p) & (same & valid).any(1)
        # 무작위 유효 방향 선택 (누적합 트릭 — 행별 choice 를 벡터화)
        r = rng.random(k)
        cum = np.cumsum(valid, axis=1)
        pick = np.argmax(cum > (r * cum[:, -1])[:, None], axis=1)
        pick = np.where(keep, np.argmax(same & valid, axis=1), pick)
        d = dirs[pick]
        self.direction[idx] = d
        self.seg_a[idx] = node
        self.node[idx] = node + d
        self.goal[idx] = self._lane_goal(idx)
        self.goal_is_station[idx] = False

    def _on_reached(self, idx: np.ndarray) -> None:
        """목표에 도착한 에이전트 처리: 스테이션 대기 / 차량 정차 / 다음 노드."""
        cfg, rng = self.cfg, self.rng
        veh = self.is_vehicle[idx]
        at_station = self.goal_is_station[idx]
        # (1) 스테이션 도착 → 대기 후 통로(차선)로 복귀
        s_idx = idx[at_station]
        if len(s_idx):
            lo, hi = cfg.idle_s
            self.idle_until[s_idx] = self.t + rng.integers(
                int(lo * cfg.hz), int(hi * cfg.hz) + 1, len(s_idx)
            )
            self.goal[s_idx] = self._lane_goal(s_idx)
            self.goal_is_station[s_idx] = False
        # (2) 노드 도착 작업자 → 일부는 셀 안 스테이션으로
        w_idx = idx[~at_station & ~veh]
        if len(w_idx):
            to_station = rng.random(len(w_idx)) < cfg.station_prob
            st = w_idx[to_station]
            if len(st):
                # 통로 중심선에서 충분히 떨어진 위치 — 통로 위에 서 있는 비현실적 상황을 막는다.
                lo_m = self.STATION_MIN_OFFSET
                hi_m = max(lo_m + 0.5, cfg.aisle_spacing_m / 2 - 0.8)
                mag = rng.uniform(lo_m, hi_m, (len(st), 2))
                sign = np.where(rng.random((len(st), 2)) < 0.5, -1.0, 1.0)
                self.goal[st] = self.layout.node_xy(self.node[st, 0], self.node[st, 1]) + mag * sign
                self.goal_is_station[st] = True
            self._pick_next_node(w_idx[~to_station])
        # (3) 노드 도착 차량 → 일부 정차 후 다음 노드
        v_idx = idx[~at_station & veh]
        if len(v_idx):
            stop = rng.random(len(v_idx)) < cfg.vehicle_stop_prob
            if stop.any():
                lo, hi = cfg.vehicle_stop_s
                self.idle_until[v_idx[stop]] = self.t + rng.integers(
                    int(lo * cfg.hz), int(hi * cfg.hz) + 1, int(stop.sum())
                )
            self._pick_next_node(v_idx)

    # ---------------------------------------------------------------- 동역학
    def _repulsion(
        self, dx: np.ndarray, dy: np.ndarray, d: np.ndarray, aware: np.ndarray, e_goal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """social-force 반발 가속도 ``(fx, fy)`` 와 차량 전방 제동 계수 ``gov`` (각 ``(n,)``).

        (n, n) 밀집 행렬 대신 상호작용 반경 안의 쌍만 골라(``np.nonzero``, 200명 기준 수백 쌍) 계산하고
        ``np.bincount`` 로 합친다 — 밀집 (n, n, 2) 연산 대비 스텝당 약 5배 빠르다.

        차량은 "멀리 있는 사람에게 밀려나는" 것이 아니라 **전방 원뿔(±25°) 안의 가장 가까운 에이전트까지
        거리에 따라 감속**한다. 전방향 장거리 반발을 주면 밀도가 높은 공장에서 차량이 아예 움직이지
        못하는 것을 실험으로 확인했다(평균 속력 0.4 m/s).
        """
        within = (d < self.r_int[:, None]) & aware[:, None]
        ii, jj = np.nonzero(within)
        n = self.n
        gov = np.ones(n)
        if len(ii) == 0:
            return np.zeros(n), np.zeros(n), gov
        vel = self.vel
        speed = np.hypot(vel[:, 0], vel[:, 1])
        moving = speed > 0.1
        ux = np.where(moving, vel[:, 0] / np.maximum(speed, 1e-9), e_goal[:, 0])[ii]
        uy = np.where(moving, vel[:, 1] / np.maximum(speed, 1e-9), e_goal[:, 1])[ii]
        dij, dxij, dyij = d[ii, jj], dx[ii, jj], dy[ii, jj]
        cosphi = -(ux * dxij + uy * dyij) / dij  # 진행 방향과 상대 방향 사이 각
        lam = self.lam[ii]
        aniso = lam + (1 - lam) * 0.5 * (1 + cosphi)
        w = self.rep_a[ii] * np.exp((self.r0[ii, jj] - dij) / self.rep_b[ii]) * aniso / dij
        fx, fy = w * dxij, w * dyij
        c, s = math.cos(self.PASS_BIAS_RAD), math.sin(self.PASS_BIAS_RAD)
        fx, fy = c * fx - s * fy, s * fx + c * fy  # 우측 통행 편향
        # 차량 전방 원뿔 제동: 1.5 m 에서 정지, lookahead 거리에서 제한 없음 (선형)
        look = self.cfg.vehicle_lookahead_m
        cone = self.is_vehicle[ii] & (cosphi > 0.9) & (dij < look)
        if cone.any():
            factor = np.clip((dij[cone] - 1.5) / (look - 1.5), 0.0, 1.0)
            np.minimum.at(gov, ii[cone], factor)
        return np.bincount(ii, fx, n), np.bincount(ii, fy, n), gov

    def _pursuit_goals(self) -> np.ndarray:
        """차량 목표를 현재 통로 세그먼트 위 ``LOOKAHEAD`` 앞 지점으로 갱신하고(pure pursuit), 세그먼트 끝을
        지난 차량 마스크 ``(n,)`` 를 돌려준다.

        점 목표 + 회전반경 제약을 같이 쓰면 목표를 살짝 지나친 차량이 반경 3 m 로 맴돌며 교차로에
        머문다(평균 속력 0.4 m/s, 10분간 노드 통과 7회). 차선 위 전방 점을 쫓게 하면 교차로에서 자연스러운
        호를 그리며 다음 통로로 넘어간다.
        """
        veh = self.is_vehicle
        a_xy = self.layout.node_xy(self.seg_a[veh, 0], self.seg_a[veh, 1])
        d = self.direction[veh].astype(np.float64)
        right = np.stack([d[:, 1], -d[:, 0]], axis=1)
        prog = ((self.pos[veh] - a_xy) * d).sum(1)
        seg_len = self.cfg.aisle_spacing_m
        target = np.minimum(prog + self.LOOKAHEAD, seg_len)
        self.goal[veh] = a_xy + d * target[:, None] + right * self.lane[veh, None]
        done = np.zeros(self.n, dtype=bool)
        done[veh] = prog >= seg_len - 0.2
        return done

    def step(self) -> None:
        cfg, rng = self.cfg, self.rng
        n, dt, t = self.n, cfg.dt, self.t
        pos, vel = self.pos, self.vel
        active = t >= self.idle_until

        seg_done = self._pursuit_goals()
        to_goal = self.goal - pos
        dist_goal = np.hypot(to_goal[:, 0], to_goal[:, 1])
        reached = active & ((dist_goal < self.reach_tol) | seg_done)
        if reached.any():
            self._on_reached(np.flatnonzero(reached))
            active = t >= self.idle_until
            self._pursuit_goals()
            to_goal = self.goal - pos
            dist_goal = np.hypot(to_goal[:, 0], to_goal[:, 1])
        e_goal = to_goal / np.maximum(dist_goal, 1e-6)[:, None]

        # --- 쌍별 거리 (n, n): 반발력과 near-miss 판정에 공용 ---
        dx = pos[:, 0][:, None] - pos[:, 0][None, :]
        dy = pos[:, 1][:, None] - pos[:, 1][None, :]
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, np.inf)
        aware = active & (t >= self.distracted_until)
        fx, fy, gov = self._repulsion(dx, dy, d, aware, e_goal)
        # --- 정체 해소: 멈춘 작업자는 옆걸음, 오래 선 차량은 서행 ---
        speed_now = np.hypot(vel[:, 0], vel[:, 1])
        stuck = active & (speed_now < 0.15) & (dist_goal > 1.0)
        self.stuck = np.where(stuck, self.stuck + 1, 0)
        veh = self.is_vehicle
        new_side = stuck & ~veh & (self.stuck >= self.STUCK_STEPS) & (t >= self.sidestep_until)
        if new_side.any():
            k = int(new_side.sum())
            sign = np.where(rng.random(k) < 0.5, -1.0, 1.0)
            perp = np.stack([-e_goal[new_side, 1], e_goal[new_side, 0]], axis=1) * sign[:, None]
            self.sidestep[new_side] = perp
            self.sidestep_until[new_side] = t + 15
            self.stuck[new_side] = 0
        side_on = (t < self.sidestep_until) & ~veh
        dir_des = np.where(side_on[:, None], e_goal * 0.4 + self.sidestep, e_goal)
        gov = np.where(veh & (self.stuck >= self.CREEP_AFTER_STEPS), np.maximum(gov, 0.15), gov)
        v_des = dir_des * (self.speed_pref * active * gov)[:, None]
        acc = (v_des - vel) / self.tau[:, None]
        acc[:, 0] += fx
        acc[:, 1] += fy
        acc += rng.normal(0.0, 1.0, (n, 2)) * self.sigma_acc[:, None]
        v_new = vel + acc * dt

        # --- 차량: 헤딩 변화율(회전반경)·가속/제동 제한 ---
        vv = v_new[veh]
        s_des = np.hypot(vv[:, 0], vv[:, 1])
        theta_des = np.arctan2(vv[:, 1], vv[:, 0])
        s_cur = self.speed[veh]
        omega_max = np.maximum(s_cur, 0.5) / cfg.vehicle_turn_radius_m * dt
        err = _wrap_angle(theta_des - self.theta[veh])
        dth = np.clip(err, -omega_max, omega_max)
        theta = self.theta[veh] + dth
        s_des = s_des * np.clip(1.0 - np.abs(err) / (np.pi / 2), 0.35, 1.0)  # 회전 중 감속
        s = s_cur + np.clip(s_des - s_cur, -cfg.vehicle_brake_max * dt, cfg.vehicle_accel_max * dt)
        s = np.clip(s, 0.0, self.v_max[veh])
        v_new[veh] = s[:, None] * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        self.theta[veh] = theta
        self.speed[veh] = s
        # --- 작업자: 속력 상한 ---
        wk = ~veh
        sp = np.hypot(v_new[wk, 0], v_new[wk, 1])
        v_new[wk] *= np.minimum(1.0, self.v_max[wk] / np.maximum(sp, 1e-9))[:, None]
        v_new[~active] = 0.0

        # --- 이동 + 경계 ---
        pos_new = pos + v_new * dt
        clipped = np.clip(pos_new, 0.5, [cfg.plant_width_m - 0.5, cfg.plant_height_m - 0.5])
        hit = clipped != pos_new
        v_new[hit] = 0.0
        self.speed[veh & hit.any(1)] = 0.0
        self.pos = clipped
        self.vel = v_new

        # --- near-miss 에피소드 (스텝 시작 시점의 참 위치 기준) ---
        d_wv = d[: self.n_wk, self.n_wk :]
        close = d_wv < cfg.d_safe
        if close.any() or self.nm_active.any():
            self._track_near_miss(close, d_wv, active)

        # --- 주의 산만 에피소드 시작 ---
        p = cfg.distracted_per_hour / (3600 * cfg.hz)
        start = (rng.random(n) < p) & (t >= self.distracted_until)
        if start.any():
            lo, hi = cfg.distracted_s
            k = int(start.sum())
            self.distracted_until[start] = t + rng.integers(
                int(lo * cfg.hz), int(hi * cfg.hz) + 1, k
            )
        self.t = t + 1

    def _track_near_miss(self, close: np.ndarray, d_wv: np.ndarray, active: np.ndarray) -> None:
        t = self.t
        enter = close & ~self.nm_active
        if enter.any():
            self.nm_start[enter] = t
            self.nm_min[enter] = np.inf
            # 진입 시점의 상태(0 aware / 1 idle / 2 distracted) — "왜 일어났나" 분석용 라벨
            state = np.where(~active, 1, np.where(t < self.distracted_until, 2, 0)).astype(np.int8)
            wi, vj = np.nonzero(enter)
            self.nm_state[wi, vj, 0] = state[wi]
            self.nm_state[wi, vj, 1] = state[self.n_wk + vj]
        upd = close & (d_wv < self.nm_min)
        self.nm_min[upd] = d_wv[upd]
        self.nm_tmin[upd] = t
        leave = self.nm_active & ~close
        if leave.any():
            self._emit_events(np.argwhere(leave), end_step=t)
        self.nm_active = close

    def _emit_events(self, pairs: np.ndarray, end_step: int) -> None:
        # 이벤트는 행 데이터가 아니라 드문 라벨(시간당 수십 건)이라 파이썬 루프여도 무방하다.
        ms = self.cfg.start_ms
        step_ms = 1000 // self.cfg.hz
        for i, j in pairs:
            self.events.append(
                {
                    "ts_ms": int(ms + self.nm_start[i, j] * step_ms),
                    "ts_min_ms": int(ms + self.nm_tmin[i, j] * step_ms),
                    "ts_end_ms": int(ms + end_step * step_ms),
                    "worker_tag": int(self.tag_id[i]),
                    "vehicle_tag": int(self.tag_id[self.n_wk + j]),
                    "min_dist": round(float(self.nm_min[i, j]), 4),
                    "worker_state": STATE_NAMES[int(self.nm_state[i, j, 0])],
                    "vehicle_state": STATE_NAMES[int(self.nm_state[i, j, 1])],
                }
            )

    def flush_events(self) -> None:
        if self.nm_active.any():
            self._emit_events(np.argwhere(self.nm_active), end_step=self.t)
            self.nm_active[:] = False


# -------------------------------------------------------------------- 측정 모델
def _burst_mask(rng: np.random.Generator, shape: tuple[int, int], cfg: SimConfig) -> np.ndarray:
    """드롭아웃 버스트 마스크 ``(T, N)``. 시작점 +1 / 종료점 -1 을 찍고 누적합 > 0 — 루프 없이 구간을 칠한다."""
    T, N = shape
    lo, hi = cfg.burst_len_steps
    starts = np.argwhere(rng.random((T, N)) < cfg.burst_start_p)
    if len(starts) == 0:
        return np.zeros((T, N), dtype=bool)
    length = rng.integers(lo, hi + 1, len(starts))
    delta = np.zeros((T + hi + 1, N), dtype=np.int32)
    np.add.at(delta, (starts[:, 0], starts[:, 1]), 1)
    np.add.at(delta, (starts[:, 0] + length, starts[:, 1]), -1)
    return np.cumsum(delta, axis=0)[:T] > 0


def _measure(
    truth: np.ndarray, t0: int, world: _World, tag_quality: np.ndarray
) -> tuple[pa.Table, np.ndarray]:
    """참 위치 ``(T, N, 2)`` → UWB 측정 행(잡음·이상치·드롭아웃·quality). 청크 단위로 완전 벡터화."""
    cfg, rng, layout = world.cfg, world.rng, world.layout
    T, N = truth.shape[:2]
    xy = truth + rng.normal(0.0, cfg.noise_sigma_m, (T, N, 2))
    outlier = rng.random((T, N)) < cfg.outlier_p
    n_out = int(outlier.sum())
    if n_out:
        ang = rng.uniform(0, 2 * np.pi, n_out)
        mag = rng.uniform(0.5, cfg.outlier_max_m, n_out)
        xy[outlier] += np.stack([mag * np.cos(ang), mag * np.sin(ang)], axis=1)
    quality = tag_quality[None, :] + rng.normal(0.0, 6.0, (T, N))
    # 앵커 커버리지가 나쁜 가장자리(3 m 이내)는 품질 저하
    edge = (
        (truth[..., 0] < 3)
        | (truth[..., 0] > cfg.plant_width_m - 3)
        | (truth[..., 1] < 3)
        | (truth[..., 1] > cfg.plant_height_m - 3)
    )
    quality -= 15.0 * edge
    burst = _burst_mask(rng, (T, N), cfg)
    drop = rng.random((T, N)) < cfg.dropout_p
    drop |= burst & (rng.random((T, N)) < cfg.burst_dropout_p)
    if n_out:
        quality[outlier] = rng.uniform(5.0, 45.0, n_out)
    in_burst = burst & ~outlier
    quality[in_burst] = rng.uniform(15.0, 60.0, int(in_burst.sum()))
    quality = np.clip(np.rint(quality), 0, 100).astype(np.uint8)

    keep = ~drop
    step_ms = 1000 // cfg.hz
    ts = (cfg.start_ms + (t0 + np.arange(T, dtype=np.int64)) * step_ms)[:, None]
    ts = np.broadcast_to(ts, (T, N))[keep]
    tag = np.broadcast_to(world.tag_id[None, :], (T, N))[keep]
    atype = np.broadcast_to(world.agent_type[None, :], (T, N))[keep]
    x = xy[..., 0][keep].astype(np.float32)
    y = xy[..., 1][keep].astype(np.float32)
    zone = layout.zone_of(x, y)
    table = pa.table(
        {
            "ts_ms": ts,
            "tag_id": tag,
            "agent_type": atype,
            "zone_id": zone,
            "x": x,
            "y": y,
            "quality": quality[keep],
        },
        schema=RAW_SCHEMA,
    )
    # 중복 패킷: 일부 행을 복제해 붙이고 (ts, tag) 순으로 다시 정렬 — 스트리밍 싱크가 남기는 모양 그대로
    dup_idx = np.flatnonzero(rng.random(table.num_rows) < cfg.duplicate_p)
    if len(dup_idx):
        table = pa.concat_tables([table, table.take(dup_idx)]).sort_by(
            [("ts_ms", "ascending"), ("tag_id", "ascending")]
        )
    return table, keep


def _truth_table(truth: np.ndarray, t0: int, world: _World) -> pa.Table:
    T, N = truth.shape[:2]
    step_ms = 1000 // world.cfg.hz
    ts = np.broadcast_to(
        (world.cfg.start_ms + (t0 + np.arange(T, dtype=np.int64)) * step_ms)[:, None], (T, N)
    ).ravel()
    tag = np.broadcast_to(world.tag_id[None, :], (T, N)).ravel()
    return pa.table(
        {
            "ts_ms": ts,
            "tag_id": tag,
            "x": truth[..., 0].ravel().astype(np.float32),
            "y": truth[..., 1].ravel().astype(np.float32),
        },
        schema=TRUTH_SCHEMA,
    )


def partition_path(out_dir: Path, ts_ms: int, part: int) -> Path:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return out_dir / f"date={dt:%Y-%m-%d}" / f"hour={dt:%H}" / f"part-{part:02d}.parquet"


def _clear_output(out_dir: Path) -> None:
    """이전 실행의 파티션이 섞이면 결정성이 깨지므로 우리가 만든 것만 지운다."""
    for p in out_dir.glob("date=*"):
        shutil.rmtree(p)
    if (out_dir / TRUTH_DIR).exists():
        shutil.rmtree(out_dir / TRUTH_DIR)
    (out_dir / MANIFEST_NAME).unlink(missing_ok=True)


def resolve_config(
    hours: float | None = None,
    tags: int | None = None,
    seed: int = 0,
    profile: str | None = None,
    config: SimConfig | None = None,
    **overrides: Any,
) -> SimConfig:
    """profile → 명시 인자 → overrides 순으로 덮어쓴 SimConfig."""
    cfg = config or SimConfig()
    if profile is not None:
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
        cfg = replace(
            cfg, hours=float(PROFILES[profile]["hours"]), tags=int(PROFILES[profile]["tags"])
        )
    if hours is not None:
        cfg = replace(cfg, hours=hours)
    if tags is not None:
        cfg = replace(cfg, tags=tags)
    cfg = replace(cfg, seed=seed, **overrides)
    if 60 % cfg.chunk_minutes != 0:
        raise ValueError("chunk_minutes must divide 60 so chunks never cross an hour partition")
    return cfg


def simulate(
    out_dir: Path | str,
    hours: float | None = None,
    tags: int | None = None,
    seed: int = 0,
    profile: str | None = None,
    config: SimConfig | None = None,
    **overrides: Any,
) -> SimResult:
    """합성 RTLS 스트림을 ``out_dir`` 에 파티션 Parquet 으로 생성한다.

    Args:
        out_dir: 출력 루트. ``date=YYYY-MM-DD/hour=HH/part-k.parquet`` + ``_manifest.json`` (+ ``_truth/``).
        hours, tags, seed: 규모·시드. ``profile`` 이 있으면 그 값이 기본이 되고 명시 인자가 우선한다.
        profile: ``smoke`` | ``small`` | ``full``.
        config, **overrides: 세부 파라미터 (``SimConfig`` 필드).

    Returns:
        생성 행 수·소요 시간·처리량(rows/s)·near-miss 이벤트 수.
    """
    cfg = resolve_config(hours, tags, seed, profile, config, **overrides)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_output(out_dir)
    rng = np.random.default_rng(cfg.seed)
    world = _World(cfg, rng)
    tag_quality = np.clip(rng.normal(85.0, 5.0, cfg.tags), 60.0, 95.0)

    steps_total = round(cfg.hours * 3600 * cfg.hz)
    chunk_steps = cfg.chunk_minutes * 60 * cfg.hz
    chunks_per_hour = 60 // cfg.chunk_minutes
    n_chunks = math.ceil(steps_total / chunk_steps)
    truth_buf = np.empty((chunk_steps, cfg.tags, 2))
    rows = 0
    files = 0
    rows_per_hour: dict[str, int] = {}
    t_start = time.perf_counter()
    log.info(
        "simulate: %.3f h × %d tags → %d steps in %d chunks",
        cfg.hours,
        cfg.tags,
        steps_total,
        n_chunks,
    )
    for c in range(n_chunks):
        t0 = c * chunk_steps
        T = min(chunk_steps, steps_total - t0)
        for i in range(T):
            truth_buf[i] = world.pos
            world.step()
        truth = truth_buf[:T]
        table, _keep = _measure(truth, t0, world, tag_quality)
        path = partition_path(out_dir, cfg.start_ms + t0 * (1000 // cfg.hz), c % chunks_per_hour)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="zstd", row_group_size=60 * cfg.hz * cfg.tags)
        if cfg.write_truth:
            tpath = out_dir / TRUTH_DIR / path.relative_to(out_dir)
            tpath.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(_truth_table(truth, t0, world), tpath, compression="zstd")
        rows += table.num_rows
        files += 1
        hour_key = path.parent.name
        rows_per_hour[hour_key] = rows_per_hour.get(hour_key, 0) + table.num_rows
        if c % 12 == 0 or c == n_chunks - 1:
            el = time.perf_counter() - t_start
            log.info(
                "chunk %d/%d rows=%d (%.0f rows/s, %d near-miss)",
                c + 1,
                n_chunks,
                rows,
                rows / max(el, 1e-9),
                len(world.events),
            )
    world.flush_events()
    seconds = time.perf_counter() - t_start
    events = sorted(world.events, key=lambda e: (e["ts_ms"], e["worker_tag"], e["vehicle_tag"]))
    vehicle_hours = world.n_veh * cfg.hours
    manifest: dict[str, Any] = {
        "profile": profile,
        "config": asdict(cfg),
        "layout": {
            "width_m": cfg.plant_width_m,
            "height_m": cfg.plant_height_m,
            "zone_cell_m": cfg.zone_cell_m,
            "n_cols": cfg.layout.n_cols,
            "n_rows": cfg.layout.n_rows,
            "n_zones": cfg.layout.n_zones,
        },
        "schema": {f.name: str(f.type) for f in RAW_SCHEMA},
        "roster": [
            {"tag_id": int(t), "agent_type": int(a), "speed_pref": round(float(s), 3)}
            for t, a, s in zip(world.tag_id, world.agent_type, world.speed_pref)
        ],
        "n_workers": world.n_wk,
        "n_vehicles": world.n_veh,
        "steps": steps_total,
        "total_rows": rows,
        "expected_rows": steps_total * cfg.tags,
        "files": files,
        "rows_per_hour": rows_per_hour,
        "truth_dir": TRUTH_DIR if cfg.write_truth else None,
        "generation": {
            "seconds": round(seconds, 3),
            "rows_per_s": round(rows / max(seconds, 1e-9), 1),
        },
        "near_miss": {
            "d_safe": cfg.d_safe,
            "count": len(events),
            "vehicle_hours": round(vehicle_hours, 4),
            "per_vehicle_hour": round(len(events) / max(vehicle_hours, 1e-9), 3),
            "events": events,
        },
    }
    manifest_path = out_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info(
        "done: %d rows in %.1fs (%.0f rows/s), %d near-miss (%.2f / vehicle-hour)",
        rows,
        seconds,
        rows / max(seconds, 1e-9),
        len(events),
        manifest["near_miss"]["per_vehicle_hour"],
    )
    return SimResult(
        out_dir,
        rows,
        seconds,
        rows / max(seconds, 1e-9),
        len(events),
        manifest_path,
        cfg,
        files,
        events,
    )


def load_manifest(out_dir: Path | str) -> dict[str, Any]:
    return json.loads((Path(out_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))


def layout_from_manifest(manifest: dict[str, Any] | None) -> PlantLayout:
    if not manifest:
        return PlantLayout()
    lay = manifest["layout"]
    return PlantLayout(lay["width_m"], lay["height_m"], lay["zone_cell_m"])


def main(argv: list[str] | None = None) -> SimResult:
    """typer 없는 진입점 — CLI 담당자가 ``foresight simulate`` 에서 그대로 감싼다."""
    p = argparse.ArgumentParser(description="합성 RTLS 스트림 생성")
    p.add_argument("--out-dir", default="data/rtls/raw")
    p.add_argument("--profile", choices=sorted(PROFILES), default=None)
    p.add_argument("--hours", type=float, default=None)
    p.add_argument("--tags", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-truth", action="store_true", help="_truth/ 참값 Parquet 을 쓰지 않는다")
    a = p.parse_args(argv)
    return simulate(a.out_dir, a.hours, a.tags, a.seed, a.profile, write_truth=not a.no_truth)


if __name__ == "__main__":
    main()
