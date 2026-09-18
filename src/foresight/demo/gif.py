"""README 용 GIF: 재생 데이터(``build_replay`` 결과)와 모델로 프레임을 그려 애니메이션으로 저장한다.

페이지(JS)와 같은 절차를 파이썬 쪽 구현으로 다시 돌린다: 8 프레임 이력 → 예측 분포 → K 샘플 → 쌍 위험 → 경보 정책.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from foresight.demo.replay import STEP_SECONDS
from foresight.utils import get_logger

log = get_logger(__name__)

WORKER_C, VEHICLE_C, RISK_C, ALERT_C, AISLE_C = (
    "#0f766e",
    "#c2410c",
    "#dc2626",
    "#b91c1c",
    "#e3e7e0",
)


def _frame_states(
    replay: dict[str, Any], model: Any, k: int, seed: int, d_safe: float, policy_kw: dict[str, Any]
) -> list[dict[str, Any]]:
    import torch

    from foresight.eval import metrics as M
    from foresight.inference.predictor import TorchPredictor
    from foresight.serving.risk import AlertPolicy, PairRisk, pairwise_risk

    agents, frames, preds = replay["agents"], replay["frames"], replay["pred"]
    obs_len = replay["meta"]["obs_len"]
    types = np.array([a["type"] for a in agents], dtype=np.int8)
    pos_at = [{a[0]: (a[1], a[2]) for a in fr["a"]} for fr in frames]
    predictor = TorchPredictor(model)
    policy = AlertPolicy(**policy_kw)
    out: list[dict[str, Any]] = []
    for f in range(len(frames)):
        st: dict[str, Any] = {
            "pos": pos_at[f],
            "ids": [],
            "mean": None,
            "samples": None,
            "pairs": [],
            "alerts": [],
        }
        pre = preds[f]
        if pre and pre["ids"]:
            ids = pre["ids"]
            hist = []
            for i in ids:
                h = [pos_at[t].get(i) for t in range(f - obs_len + 1, f + 1)]
                hist.append(h)
            if all(all(p is not None for p in h) for h in hist):
                obs = np.array(hist, dtype=np.float64)
                pred = predictor.predict(obs, k=0)
                params = torch.from_numpy(pred.params)
                g = torch.Generator().manual_seed(seed * 100003 + f)
                last = torch.from_numpy(obs[:, -1].astype(np.float32))
                if k > 0:
                    rel = M.sample_relative(params, k, generator=g)
                    samples = M.relative_to_absolute(rel, last).permute(0, 2, 1, 3).numpy()
                else:
                    samples = pred.mean_abs[None]
                rm = pairwise_risk(samples, types[ids], d_safe)
                obs_list = []
                for wi, w in enumerate(rm.worker_idx):
                    for vi, v in enumerate(rm.vehicle_idx):
                        pr = PairRisk(
                            ids[w],
                            ids[v],
                            float(rm.risk[wi, vi]),
                            float(rm.ttc_s[wi, vi]),
                            float(rm.min_dist_mean[wi, vi]),
                        )
                        st["pairs"].append(pr)
                        obs_list.append((agents[ids[w]]["id"], agents[ids[v]]["id"], pr))
                st["ids"], st["mean"], st["samples"] = (
                    ids,
                    pred.mean_abs,
                    samples if k > 0 else None,
                )
                st["alerts"] = policy.update(obs_list, f * STEP_SECONDS)
        st["n_alerts"] = policy.n_alerts
        out.append(st)
    return out


def render_gif(
    replay: dict[str, Any],
    model: Any,
    path: Path,
    k: int = 20,
    seed: int = 0,
    d_safe: float | None = None,
    fps: int = 5,
    max_frames: int = 100,
    start: int | None = None,
    threshold: float = 0.3,
) -> Path:
    """재생 창을 GIF 로 저장한다. ``start`` 가 없으면 위험이 가장 높은 프레임이 가운데 오도록 자른다."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle, Rectangle

    d_safe = replay["meta"]["d_safe"] if d_safe is None else d_safe
    states = _frame_states(replay, model, k, seed, d_safe, {"threshold": threshold})
    agents = replay["agents"]
    lay = replay["meta"].get("layout") or {"x0": 0.0, "y0": 0.0, "size": 20.0, "aisle_m": 10.0}
    nf = len(states)
    if start is None:
        peak = int(np.argmax([max([p.risk for p in s["pairs"]], default=0.0) for s in states]))
        start = max(0, min(nf - max_frames, peak - max_frames // 2))
    sel = list(range(start, min(nf, start + max_frames)))
    x0, y0, size, aisle = lay["x0"], lay["y0"], lay["size"], lay["aisle_m"]
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=72)
    fig.patch.set_facecolor("#f2f4f0")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.05)
    alert_flash: dict[tuple[str, str], int] = {}

    def render(fi: int) -> None:
        f = sel[fi]
        s = states[f]
        ax.clear()
        ax.set_facecolor("#f7f8f5")
        half = aisle / 2
        a = half
        while a < size:
            ax.add_patch(Rectangle((x0 + a - 1.2, y0), 2.4, size, color=AISLE_C, zorder=0))
            ax.add_patch(Rectangle((x0, y0 + a - 1.2), size, 2.4, color=AISLE_C, zorder=0))
            a += aisle
        for g in np.arange(0, size + 1e-9, 5):
            ax.plot([x0 + g, x0 + g], [y0, y0 + size], color="#cfd5cc", lw=0.6, zorder=0)
            ax.plot([x0, x0 + size], [y0 + g, y0 + g], color="#cfd5cc", lw=0.6, zorder=0)
        pos = s["pos"]
        if s["samples"] is not None:
            for kk in range(min(k, 20)):
                for n, i in enumerate(s["ids"]):
                    tr = s["samples"][kk, n]
                    p0 = pos[i]
                    c = WORKER_C if agents[i]["type"] == 0 else VEHICLE_C
                    ax.plot(
                        [p0[0], *tr[:, 0]],
                        [p0[1], *tr[:, 1]],
                        color=c,
                        lw=0.6,
                        alpha=0.12,
                        zorder=1,
                    )
        if s["mean"] is not None:
            for n, i in enumerate(s["ids"]):
                tr = s["mean"][n]
                p0 = pos[i]
                c = WORKER_C if agents[i]["type"] == 0 else VEHICLE_C
                ax.plot([p0[0], *tr[:, 0]], [p0[1], *tr[:, 1]], color=c, lw=1.4, zorder=2)
        for pr in s["pairs"]:
            if pr.risk < 0.1:
                continue
            pw, pv = pos[pr.worker], pos[pr.vehicle]
            ax.plot(
                [pw[0], pv[0]],
                [pw[1], pv[1]],
                color=RISK_C,
                lw=0.8 + 4 * pr.risk,
                alpha=0.3 + 0.6 * pr.risk,
                zorder=3,
            )
        for i, p in pos.items():
            if agents[i]["type"] == 0:
                ax.add_patch(Circle(p, 0.32, color=WORKER_C, zorder=4))
            else:
                ax.add_patch(
                    Rectangle((p[0] - 0.3, p[1] - 0.3), 0.6, 0.6, color=VEHICLE_C, zorder=4)
                )
        for al in s["alerts"]:
            alert_flash[(al.worker_id, al.vehicle_id)] = f
        id_to_idx = {a["id"]: j for j, a in enumerate(agents)}
        for key, f0 in list(alert_flash.items()):
            if f - f0 > 4 or f < f0:
                del alert_flash[key]
                continue
            for aid in key:
                p = pos.get(id_to_idx[aid])
                if p is not None:
                    ax.add_patch(
                        Circle(
                            p,
                            0.7 + 0.15 * (f - f0),
                            fill=False,
                            color=ALERT_C,
                            lw=2,
                            alpha=1 - (f - f0) / 5,
                            zorder=5,
                        )
                    )
        ax.set_xlim(x0, x0 + size)
        ax.set_ylim(y0, y0 + size)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#d3d9d1")
        top = sorted(s["pairs"], key=lambda p: -p.risk)[:1]
        risk_txt = f"max risk {top[0].risk:.2f}" if top else "no predictable pair"
        ax.set_title(
            f"zone {replay['meta']['zone']}  t = {f * STEP_SECONDS:5.1f} s   alerts {s['n_alerts']}   {risk_txt}",
            fontsize=10,
            loc="left",
            color="#1a1f1c",
        )
        fig.texts.clear()
        fig.text(
            0.02,
            0.012,
            f"teal = worker · orange = vehicle · faint = {k} samples · red link = risk (d_safe {d_safe} m) · ring = alert",
            fontsize=7,
            color="#5f6862",
        )

    anim = FuncAnimation(fig, render, frames=len(sel), interval=1000 / fps)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    log.info("gif: %s (%d frames, %.1f KB)", path, len(sel), path.stat().st_size / 1024)
    return path
