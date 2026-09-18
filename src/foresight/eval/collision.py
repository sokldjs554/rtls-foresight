"""충돌 경보 품질 평가 — 궤적 오차가 아니라 "경보가 맞았는가"로 모델을 채점한다.

장면(작업자·차량 N명, 관측 8 + 미래 12 프레임)마다 (작업자, 차량) 쌍의 정답을 만든다.

    양성 = 예측 시점에는 d_safe 밖에 있던 (작업자, 차량) 쌍이 미래 12 프레임(4.8 s) 안에 **측정 위치** 거리 d_safe 안으로
           들어오는 쌍. 예측 시점에 이미 d_safe 안인 쌍은 "사전 경보"의 대상이 아니므로 평가에서 제외한다 (개수만 보고)

각 방법이 쌍마다 위험 점수를 내면 임계값을 훑어 정밀도/재현율/F1, AP(AUPRC), AUROC, 검출된 양성의
선행시간(예측 시점 → 최초 접근까지의 시간), 시간당 오경보 수를 계산한다.

비교 대상
* ``geofence``   — 현재 거리 기반 규칙 (대부분의 RTLS 제품): score = 1 / (1 + 현재 거리)
* ``cvm``        — 등속 예측(20 샘플, 각도 σ=25°)에서 뽑은 위험 확률
* ``model``      — Social-STGCNN 분포 샘플(20)에서 뽑은 위험 확률 (``foresight.serving.risk.pairwise_risk``)
* ``model_det``  — 평균 궤적만 쓴 결정적 위험 (0/1)

정답이 "측정 위치" 기준인 이유: SceneSet 에는 노이즈 없는 진짜 위치가 없다(생성기의 ``_truth`` 는 별도).
UWB 노이즈 σ 0.15 m 가 d_safe=1.0 m 경계 근처의 라벨을 흔들지만, 모든 방법에 같은 정답이 적용되므로
상대 비교는 공정하다. 절대 수치는 이 점을 감안해 읽는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from foresight.data.ethucy import SceneSet
from foresight.inference.predictor import Prediction, Predictor
from foresight.models import constant_velocity
from foresight.serving.risk import STEP_SECONDS, VEHICLE, WORKER, pairwise_risk, risk_deterministic


@dataclass
class PairRecord:
    scene: int
    worker: int
    vehicle: int
    label: bool
    first_cross_step: int  # 최초로 d < d_safe 가 되는 미래 스텝 (1..12), 음성이면 0
    current_dist: float


@dataclass
class MethodScores:
    name: str
    scores: np.ndarray  # (P,)


@dataclass
class CollisionEval:
    d_safe: float
    n_scenes: int
    n_pairs: int
    n_positive: int
    hours: float
    sample_fraction: float = 1.0  # 평가한 장면 비율 (--every k → 1/k); 오경보/시간은 1/비율로 외삽
    n_already_close: int = 0  # 예측 시점에 이미 d_safe 안에 있어 라벨에서 뺀 쌍
    methods: dict[str, dict] = field(default_factory=dict)


def stream_hours(scenes: SceneSet) -> float:
    """테스트 스트림이 덮는 시간(h). RTLS 장면 meta 의 시작 빈(0.4 s 단위)으로 계산한다.

    장면 수 × 0.4 s 로 세면 틀린다: 같은 시각에 구역이 여러 개 있고(중복 계산), 부분 샘플(every=k)이면 과소 계산된다.
    시작 빈의 범위 + 윈도우 길이가 실제 관측 구간이다. meta 가 빈이 아니면(ETH/UCY: 파일명+프레임) 장면 수 기반으로 되돌아간다.
    """
    starts = np.array([m[1] for m in scenes.meta], dtype=np.float64)
    if len(starts) > 1 and all(str(m[0]).startswith("zone=") for m in scenes.meta):
        span_bins = starts.max() - starts.min() + scenes.obs_len + scenes.pred_len
        return float(span_bins * STEP_SECONDS / 3600.0)
    return len(scenes) * STEP_SECONDS / 3600.0


def pair_labels(
    scenes: SceneSet, d_safe: float, exclude_already_close: bool = True
) -> list[PairRecord]:
    """모든 (작업자, 차량) 쌍의 정답. ``exclude_already_close`` 면 예측 시점 거리 < d_safe 인 쌍은 뺀다."""
    out: list[PairRecord] = []
    obs_len = scenes.obs_len
    for i in range(len(scenes)):
        types = scenes.types(i)
        w = np.where(types == WORKER)[0]
        v = np.where(types == VEHICLE)[0]
        if len(w) == 0 or len(v) == 0:
            continue
        pos = scenes.scene(i)  # (N, T, 2)
        fut = pos[:, obs_len:]
        d = np.linalg.norm(fut[w][:, None] - fut[v][None], axis=-1)  # (W, V, T_pred)
        cur = np.linalg.norm(
            pos[w, obs_len - 1][:, None] - pos[v, obs_len - 1][None], axis=-1
        )  # (W, V)
        below = d < d_safe
        for a in range(len(w)):
            for b in range(len(v)):
                if exclude_already_close and cur[a, b] < d_safe:
                    continue
                lab = bool(below[a, b].any())
                first = int(np.argmax(below[a, b])) + 1 if lab else 0
                out.append(PairRecord(i, int(w[a]), int(v[b]), lab, first, float(cur[a, b])))
    return out


def _risk_lookup(rm, worker: int, vehicle: int) -> float:
    a = int(np.where(rm.worker_idx == worker)[0][0])
    b = int(np.where(rm.vehicle_idx == vehicle)[0][0])
    return float(rm.risk[a, b])


def score_pairs(
    scenes: SceneSet,
    records: list[PairRecord],
    predictor: Predictor | None,
    d_safe: float,
    k: int = 20,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """방법별 위험 점수 벡터 (records 순서)."""
    obs_len = scenes.obs_len
    rng = np.random.default_rng(seed)
    by_scene: dict[int, list[int]] = {}
    for j, r in enumerate(records):
        by_scene.setdefault(r.scene, []).append(j)
    n = len(records)
    out = {"geofence": np.zeros(n), "cvm": np.zeros(n)}
    if predictor is not None:
        out["model"] = np.zeros(n)
        out["model_det"] = np.zeros(n)
    for i, idx in by_scene.items():
        pos = scenes.scene(i)
        types = scenes.types(i)
        obs = pos[:, :obs_len]
        cv = constant_velocity(
            obs, scenes.pred_len, k=k, angle_std_deg=25.0, rng=rng
        )  # (K, N, T, 2)
        rm_cvm = pairwise_risk(cv, types, d_safe=d_safe, dt=STEP_SECONDS)
        if predictor is not None:
            pred: Prediction = predictor.predict(obs, k=k, seed=seed + i)
            rm_model = pairwise_risk(pred.samples_abs, types, d_safe=d_safe, dt=STEP_SECONDS)  # type: ignore[arg-type]
            rm_det = risk_deterministic(pred.mean_abs, types, d_safe=d_safe, dt=STEP_SECONDS)
        for j in idx:
            r = records[j]
            out["geofence"][j] = 1.0 / (1.0 + r.current_dist)
            out["cvm"][j] = _risk_lookup(rm_cvm, r.worker, r.vehicle)
            if predictor is not None:
                out["model"][j] = _risk_lookup(rm_model, r.worker, r.vehicle)
                out["model_det"][j] = _risk_lookup(rm_det, r.worker, r.vehicle)
    return out


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # 순위 기반 (동률 0.5)
    allv = np.concatenate([pos, neg])
    ranks = np.empty(len(allv))
    order = np.argsort(allv, kind="stable")
    sorted_v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate_collision(
    scenes: SceneSet,
    predictors: dict[str, Predictor | None],
    d_safe: float = 1.0,
    k: int = 20,
    hours: float | None = None,
    thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.7),
    sample_fraction: float = 1.0,
) -> CollisionEval:
    """충돌 사전 경보 품질. ``sample_fraction`` < 1 이면(장면 부분 샘플) 오경보/시간을 1/비율로 외삽한다."""
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")
    all_records = pair_labels(scenes, d_safe, exclude_already_close=False)
    records = [r for r in all_records if r.current_dist >= d_safe]
    y = np.array([r.label for r in records], dtype=np.int64)
    first = np.array([r.first_cross_step for r in records])
    if hours is None:
        hours = stream_hours(scenes)
    ev = CollisionEval(
        d_safe=d_safe,
        n_scenes=len(scenes),
        n_pairs=len(records),
        n_positive=int(y.sum()),
        hours=hours,
        sample_fraction=sample_fraction,
        n_already_close=len(all_records) - len(records),
    )
    scored: dict[str, np.ndarray] = {}
    for name, pred in predictors.items():
        parts = score_pairs(scenes, records, pred, d_safe, k=k)
        if pred is None:
            scored.update({k2: v for k2, v in parts.items() if k2 in ("geofence", "cvm")})
        else:
            scored[name] = parts["model"]
            scored[f"{name}_det"] = parts["model_det"]
            scored.setdefault("geofence", parts["geofence"])
            scored.setdefault("cvm", parts["cvm"])
    for name, s in scored.items():
        entry: dict = {"ap": average_precision(y, s), "auroc": auroc(y, s), "thresholds": {}}
        # geofence 는 점수 스케일이 다르므로 "현재 거리 < r" 규칙에 해당하는 임계값도 함께 제공
        ths = (
            thresholds
            if name != "geofence"
            else tuple(1.0 / (1.0 + r) for r in (3.0, 2.5, 2.0, 1.5, 1.0))
        )
        for th in ths:
            pred_pos = s >= th
            tp = int((pred_pos & (y == 1)).sum())
            fp = int((pred_pos & (y == 0)).sum())
            fn = int((~pred_pos & (y == 1)).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            lead = first[pred_pos & (y == 1)] * STEP_SECONDS
            label = f"r<{1.0 / th - 1.0:.1f}m" if name == "geofence" else f"{th:.2f}"
            entry["thresholds"][label] = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "false_alarms_per_hour": fp / sample_fraction / max(hours, 1e-9),
                "lead_time_mean_s": float(lead.mean()) if len(lead) else float("nan"),
            }
        best = max(entry["thresholds"].items(), key=lambda kv: kv[1]["f1"])
        entry["best_f1"] = {"threshold": best[0], **best[1]}
        ev.methods[name] = entry
    return ev


def collision_table(ev: CollisionEval) -> str:
    rows = [
        "| 방법 | AP (AUPRC) | AUROC | 최고 F1 @임계 | 정밀도 / 재현율 | 오경보/시간 | 평균 선행시간 (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    label = {"geofence": "지오펜스 (현재 거리)", "cvm": "CVM-S (20 샘플)"}
    for name, e in ev.methods.items():
        b = e["best_f1"]
        rows.append(
            f"| {label.get(name, name)} | {e['ap']:.3f} | {e['auroc']:.3f} | {b['f1']:.3f} @ {b['threshold']} | "
            f"{b['precision']:.2f} / {b['recall']:.2f} | {b['false_alarms_per_hour']:.1f} | {b['lead_time_mean_s']:.2f} |"
        )
    note = (
        f"\n장면 {ev.n_scenes:,} · 쌍 {ev.n_pairs:,} · 양성 {ev.n_positive:,} ({100 * ev.n_positive / max(ev.n_pairs, 1):.2f}%)"
        f" · 이미 근접해 제외 {ev.n_already_close:,} · 스트림 {ev.hours:.2f} h · d_safe {ev.d_safe} m"
    )
    if ev.sample_fraction < 1.0:
        note += f" · 장면 1/{round(1.0 / ev.sample_fraction)} 부분 샘플 (오경보/시간은 외삽)"
    rows.append(note)
    return "\n".join(rows)


def save_collision_eval(ev: CollisionEval, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "d_safe": ev.d_safe,
        "n_scenes": ev.n_scenes,
        "n_pairs": ev.n_pairs,
        "n_positive": ev.n_positive,
        "positive_rate": ev.n_positive / max(ev.n_pairs, 1),
        "hours": ev.hours,
        "sample_fraction": ev.sample_fraction,
        "n_already_close": ev.n_already_close,
        "methods": ev.methods,
        "table_markdown": collision_table(ev),
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")


def plot_pr_curves(
    scenes: SceneSet,
    predictors: dict[str, Predictor | None],
    path: Path,
    d_safe: float = 1.0,
    k: int = 20,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = pair_labels(scenes, d_safe)
    y = np.array([r.label for r in records], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    curves: dict[str, np.ndarray] = {}
    for name, pred in predictors.items():
        parts = score_pairs(scenes, records, pred, d_safe, k=k)
        if pred is None:
            curves["geofence (current distance)"] = parts["geofence"]
            curves["CVM-S (20 samples)"] = parts["cvm"]
        else:
            curves[f"{name} (20 samples)"] = parts["model"]
    for name, s in curves.items():
        order = np.argsort(-s, kind="stable")
        yy = y[order]
        tp = np.cumsum(yy)
        prec = tp / np.arange(1, len(yy) + 1)
        rec = tp / max(yy.sum(), 1)
        ax.plot(rec, prec, label=f"{name}  AP={average_precision(y, s):.2f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"worker-vehicle collision within 4.8 s (d_safe={d_safe} m)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


__all__ = [
    "CollisionEval",
    "evaluate_collision",
    "pair_labels",
    "plot_pr_curves",
    "save_collision_eval",
    "torch",
]
