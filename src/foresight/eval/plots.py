"""시각화: 예측 궤적(평균·샘플·정답), 학습 곡선, 재현표 막대그래프.

색은 의미가 고정된 소수 팔레트만 쓴다 — 관측(회색), 정답(검정), 평균 예측(파랑), 샘플(연한 파랑), 차량(주황).
그림은 README/docs 용 PNG 로 저장하며 항상 ``results/figures/`` 아래에 둔다.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_OBS, C_GT, C_MEAN, C_SAMPLE, C_VEH, C_WORKER = (
    "#6b7280",
    "#111827",
    "#2563eb",
    "#93c5fd",
    "#ea580c",
    "#2563eb",
)


def plot_scene(
    obs_abs: np.ndarray,
    gt_abs: np.ndarray | None,
    mean_abs: np.ndarray,
    samples_abs: np.ndarray | None = None,
    agent_type: np.ndarray | None = None,
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """obs (N,8,2), gt (N,12,2), mean (N,12,2), samples (K,N,12,2)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    n = obs_abs.shape[0]
    types = np.zeros(n, dtype=int) if agent_type is None else agent_type
    for i in range(n):
        col = C_VEH if types[i] == 1 else C_MEAN
        if samples_abs is not None:
            for k in range(samples_abs.shape[0]):
                ax.plot(
                    samples_abs[k, i, :, 0],
                    samples_abs[k, i, :, 1],
                    color=C_SAMPLE,
                    alpha=0.25,
                    lw=0.8,
                    zorder=1,
                )
        ax.plot(obs_abs[i, :, 0], obs_abs[i, :, 1], color=C_OBS, lw=1.5, zorder=2)
        ax.scatter(obs_abs[i, -1, 0], obs_abs[i, -1, 1], color=C_OBS, s=14, zorder=3)
        if gt_abs is not None:
            ax.plot(gt_abs[i, :, 0], gt_abs[i, :, 1], color=C_GT, lw=1.5, ls="--", zorder=2)
        ax.plot(mean_abs[i, :, 0], mean_abs[i, :, 1], color=col, lw=2, zorder=4)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.2)
    return ax


def plot_scene_grid(items: list[dict], path: Path, ncols: int = 3) -> None:
    """items: [{obs, gt, mean, samples, types, title}] → 격자 PNG."""
    n = len(items)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows))
    for ax, it in zip(np.array(axes).ravel(), items):
        plot_scene(
            it["obs"],
            it.get("gt"),
            it["mean"],
            it.get("samples"),
            it.get("types"),
            it.get("title", ""),
            ax,
        )
    for ax in np.array(axes).ravel()[n:]:
        ax.axis("off")
    fig.suptitle(
        "grey: observed 3.2 s | black dashed: ground truth 4.8 s | blue: mean prediction | light blue: 20 samples",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_history(history_paths: dict[str, Path], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, hp in history_paths.items():
        h = json.loads(Path(hp).read_text())
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["train_loss"] for r in h], label=name)
        axes[1].plot(ep, [r["val_loss"] for r in h], label=name)
    for ax, t in zip(axes, ("train NLL", "val NLL")):
        ax.set_title(t)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reproduction(results_json: Path, path: Path) -> None:
    """분할별 ADE/FDE: 논문 vs 공식 ckpt 재평가 vs 우리 학습 vs CVM."""
    r = json.loads(Path(results_json).read_text())
    splits = [s for s in ("eth", "hotel", "univ", "zara1", "zara2", "avg") if s in r["ours"]]
    series = {
        "paper": [r["paper"][s] for s in splits],
        "official ckpt (re-eval)": [
            r["official_checkpoint"].get(s, {}).get("best_of_k_per_agent") for s in splits
        ],
        "ours": [r["ours"][s]["best_of_k_per_agent"] for s in splits],
        "CVM (deterministic)": [r["ours"][s]["cvm"] for s in splits],
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    width = 0.2
    x = np.arange(len(splits))
    colors = ["#9ca3af", "#60a5fa", "#2563eb", "#f59e0b"]
    for ax, metric in zip(axes, ("ade", "fde")):
        for j, (name, vals) in enumerate(series.items()):
            ys = [v[metric] if v else np.nan for v in vals]
            ax.bar(x + (j - 1.5) * width, ys, width, label=name, color=colors[j])
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_title(f"{metric.upper()} (m, best-of-20)")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
