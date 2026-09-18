"""충돌 경보 평가: 라벨 규칙, AP/AUROC, 스트림 시간 계산."""

from __future__ import annotations

import numpy as np

from foresight.data.ethucy import SceneSet
from foresight.eval.collision import (
    auroc,
    average_precision,
    collision_table,
    evaluate_collision,
    pair_labels,
    stream_hours,
)


def _scene(worker_path: np.ndarray, vehicle_path: np.ndarray) -> SceneSet:
    pos = np.stack([worker_path, vehicle_path]).astype(np.float64)  # (2, 20, 2)
    return SceneSet(
        pos=pos,
        scene_index=np.array([[0, 2]]),
        meta=[("zone=0", 100.0)],
        obs_len=8,
        pred_len=12,
        agent_type=np.array([0, 1], dtype=np.int8),
    )


def test_pair_label_positive_when_future_distance_below_d_safe() -> None:
    t = np.arange(20, dtype=np.float64)
    worker = np.stack([t * 0.5, np.zeros(20)], axis=-1)  # x 방향 이동
    vehicle = np.stack([10.0 - t * 0.5, np.zeros(20)], axis=-1)  # 정면에서 접근 → t=10 에 만남
    recs = pair_labels(_scene(worker, vehicle), d_safe=1.0)
    assert len(recs) == 1 and recs[0].label and 1 <= recs[0].first_cross_step <= 12


def test_pair_label_negative_when_far_apart() -> None:
    t = np.arange(20, dtype=np.float64)
    worker = np.stack([t * 0.5, np.zeros(20)], axis=-1)
    vehicle = np.stack([t * 0.5, np.full(20, 30.0)], axis=-1)
    recs = pair_labels(_scene(worker, vehicle), d_safe=1.0)
    assert len(recs) == 1 and not recs[0].label and recs[0].first_cross_step == 0


def test_ap_and_auroc_extremes() -> None:
    y = np.array([1, 0, 1, 0, 0])
    assert average_precision(y, np.array([0.9, 0.1, 0.8, 0.2, 0.3])) == 1.0
    assert auroc(y, np.array([0.9, 0.1, 0.8, 0.2, 0.3])) == 1.0
    assert abs(auroc(y, np.zeros(5)) - 0.5) < 1e-9


def test_stream_hours_uses_bin_span() -> None:
    pos = np.zeros((4, 20, 2))
    ss = SceneSet(
        pos=pos,
        scene_index=np.array([[0, 2], [2, 4]]),
        meta=[("zone=0", 0.0), ("zone=3", 9000.0 - 20)],
        obs_len=8,
        pred_len=12,
    )
    assert abs(stream_hours(ss) - 1.0) < 1e-9  # 9000 bins × 0.4 s = 1 h


def test_evaluate_collision_baselines_rank_head_on_higher() -> None:
    t = np.arange(20, dtype=np.float64)
    head_on = _scene(
        np.stack([t * 0.5, np.zeros(20)], -1), np.stack([10.0 - t * 0.5, np.zeros(20)], -1)
    )
    far = _scene(np.stack([t * 0.5, np.zeros(20)], -1), np.stack([t * 0.5, np.full(20, 30.0)], -1))
    pos = np.concatenate([head_on.pos, far.pos])
    ss = SceneSet(
        pos=pos,
        scene_index=np.array([[0, 2], [2, 4]]),
        meta=[("zone=0", 0.0), ("zone=0", 1.0)],
        obs_len=8,
        pred_len=12,
        agent_type=np.array([0, 1, 0, 1], dtype=np.int8),
    )
    ev = evaluate_collision(ss, {"_baselines": None}, d_safe=1.0, k=20)
    assert ev.n_positive == 1 and ev.n_pairs == 2
    assert ev.methods["cvm"]["ap"] == 1.0 and ev.methods["geofence"]["ap"] == 1.0


def test_false_alarms_per_hour_extrapolated_by_sample_fraction() -> None:
    """부분 샘플(every=k) 평가에서는 오경보/시간을 1/비율로 외삽하고, 이미 근접한 쌍은 라벨에서 뺀다."""
    t = np.arange(20, dtype=np.float64)
    walk = np.stack([t * 0.5, np.zeros(20)], -1)
    parallel = _scene(
        walk, walk + np.array([0.0, 2.0])
    )  # 2 m 간격 평행 이동: 음성, 지오펜스 r<3m 는 오경보
    already = _scene(walk, walk + np.array([0.0, 0.5]))  # 0.5 m: 이미 근접 → 제외
    ss = SceneSet(
        pos=np.concatenate([parallel.pos, already.pos]),
        scene_index=np.array([[0, 2], [2, 4]]),
        meta=[("zone=0", 0.0), ("zone=0", 8980.0)],  # 9000 빈 = 1 h
        obs_len=8,
        pred_len=12,
        agent_type=np.array([0, 1, 0, 1], dtype=np.int8),
    )
    full = evaluate_collision(ss, {"_baselines": None}, d_safe=1.0, k=20)
    half = evaluate_collision(ss, {"_baselines": None}, d_safe=1.0, k=20, sample_fraction=0.5)
    assert full.n_pairs == 1 and full.n_positive == 0 and full.n_already_close == 1
    fa_full = full.methods["geofence"]["thresholds"]["r<3.0m"]["false_alarms_per_hour"]
    fa_half = half.methods["geofence"]["thresholds"]["r<3.0m"]["false_alarms_per_hour"]
    assert abs(fa_full - 1.0) < 1e-9 and abs(fa_half - 2.0) < 1e-9
    assert half.sample_fraction == 0.5 and "부분 샘플" in collision_table(half)
