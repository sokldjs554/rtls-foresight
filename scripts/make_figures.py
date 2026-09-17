"""results/ 의 산출물에서 README/docs 용 그림을 다시 만든다 (재현 가능한 그림).

python scripts/make_figures.py            # 모든 그림
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("FORESIGHT_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from foresight.data.ethucy import SceneSet  # noqa: E402
from foresight.eval.evaluate import load_model  # noqa: E402
from foresight.eval.plots import plot_history, plot_reproduction, plot_scene_grid  # noqa: E402
from foresight.inference.predictor import TorchPredictor  # noqa: E402

FIG = ROOT / "results" / "figures"


def predictions(ckpt_seed: int = 0) -> None:
    items = []
    for split, ids in [("eth", [3, 20, 40]), ("zara1", [10, 100, 300]), ("univ", [5, 50])]:
        ck = ROOT / "results" / "checkpoints" / split / f"seed{ckpt_seed}" / "best.pth"
        if not ck.exists():
            continue
        ss = SceneSet.load(ROOT / "data" / "processed" / "ethucy" / split / "test.npz")
        p = TorchPredictor(load_model(ck))
        for i in ids:
            pos = ss.scene(i)
            pr = p.predict(pos[:, :8], k=20, seed=0)
            ade = np.linalg.norm(pr.mean_abs - pos[:, 8:], axis=-1).mean()
            items.append(
                {
                    "obs": pos[:, :8],
                    "gt": pos[:, 8:],
                    "mean": pr.mean_abs,
                    "samples": pr.samples_abs,
                    "title": f"{split} test scene {i} | N={pos.shape[0]} | det ADE {ade:.2f} m",
                }
            )
    if items:
        plot_scene_grid(items, FIG / "predictions_ours.png", ncols=4)


def histories() -> None:
    hp = {
        s: ROOT / "results" / "checkpoints" / s / "seed0" / "history.json"
        for s in ("eth", "hotel", "univ", "zara1", "zara2")
    }
    hp = {k: v for k, v in hp.items() if v.exists()}
    if hp:
        plot_history(hp, FIG / "training_curves.png")


def reproduction() -> None:
    rj = ROOT / "results" / "reproduction.json"
    if rj.exists() and json.loads(rj.read_text()).get("ours"):
        plot_reproduction(rj, FIG / "reproduction_bars.png")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    predictions()
    histories()
    reproduction()
    print("figures ->", FIG)
