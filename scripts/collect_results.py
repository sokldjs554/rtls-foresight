"""학습 산출물(results/checkpoints/**/metrics.json, history.json)을 표로 모은다.

    python scripts/collect_results.py
→ results/ablation.json (eth ablation 표), results/train_cost.json (분할별 학습 비용), results/seeds.json (시드 분산)
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "results" / "checkpoints"
SPLITS = ("eth", "hotel", "univ", "zara1", "zara2")


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ablation() -> None:
    base = _load(CK / "eth" / "seed0" / "metrics.json")
    rows = [
        "| 설정 | best-of-20 ADE/FDE | joint | 결정적(μ) | 최적 epoch | 비고 |",
        "|---|---|---|---|---|---|",
    ]
    entries = {}
    if base:
        entries["기본 (공식 코드 동작: view + 속도 커널 + 장면 배치 + 클램프 NLL)"] = base
    labels = {
        "permute": "논문 그림대로 permute 축 교환",
        "poskernel": "논문 본문대로 위치 간 역거리 커널",
        "bucket": "노드 수 같은 장면 32개 배치 (BN 통계 배치 단위)",
        "stableloss": "로그 영역 NLL (클램프 없음)",
    }
    for key, label in labels.items():
        m = _load(CK / f"ablation-eth-{key}" / "metrics.json")
        if m:
            entries[label] = m
    for label, m in entries.items():
        e, mt = m["eval"], m["metrics"]
        rows.append(
            f"| {label} | {e['best_of_k_per_agent']['ade']:.3f}±{e['best_of_k_per_agent_std']['ade']:.3f} / {e['best_of_k_per_agent']['fde']:.3f} | "
            f"{e['best_of_k_joint']['ade']:.2f}/{e['best_of_k_joint']['fde']:.2f} | {e['deterministic']['ade']:.2f}/{e['deterministic']['fde']:.2f} | {mt['best_epoch']} | |"
        )
    (ROOT / "results" / "ablation.json").write_text(
        json.dumps(
            {
                "entries": {
                    k: {"eval": v["eval"], "metrics": v["metrics"]} for k, v in entries.items()
                },
                "table_markdown": "\n".join(rows),
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def train_cost() -> None:
    rows = [
        "| 분할 | 학습 장면 | epoch 평균 (s) | 250 epoch 합계 (min) | 최적 epoch | 최종 검증 NLL |",
        "|---|---|---|---|---|---|",
    ]
    out = {}
    counts = {}
    for s in SPLITS:
        npz = ROOT / "data" / "processed" / "ethucy" / s / "train.npz"
        if npz.exists():
            import numpy as np

            counts[s] = int(np.load(npz)["scene_index"].shape[0])
    for s in SPLITS:
        h = _load(CK / s / "seed0" / "history.json")
        m = _load(CK / s / "seed0" / "metrics.json")
        if not h or not m:
            continue
        sec = [r["sec"] for r in h]
        out[s] = {
            "scenes": counts.get(s),
            "epoch_sec_mean": sum(sec) / len(sec),
            "total_min": sum(sec) / 60,
            "best_epoch": m["best_epoch"],
            "best_val_loss": m["best_val_loss"],
        }
        rows.append(
            f"| {s} | {counts.get(s, '-'):,} | {out[s]['epoch_sec_mean']:.1f} | {out[s]['total_min']:.0f} | {m['best_epoch']} | {m['best_val_loss']:.3f} |"
        )
    (ROOT / "results" / "train_cost.json").write_text(
        json.dumps(
            {"splits": out, "table_markdown": "\n".join(rows)}, indent=1, ensure_ascii=False
        ),
        encoding="utf-8",
    )


def seeds() -> None:
    rows = ["| 분할 | seed 0 | seed 1 | seed 2 | 평균 ± 표준편차 (ADE) |", "|---|---|---|---|---|"]
    out = {}
    for s in SPLITS:
        vals = []
        for seed in (0, 1, 2):
            m = _load(CK / s / f"seed{seed}" / "metrics.json")
            vals.append(
                None if not m else (m["metrics"]["test_ade_bo20"], m["metrics"]["test_fde_bo20"])
            )
        got = [v for v in vals if v]
        if not got:
            continue
        import numpy as np

        ade = np.array([v[0] for v in got])
        std = float(ade.std(ddof=1)) if len(ade) > 1 else 0.0
        out[s] = {"runs": vals, "ade_mean": float(ade.mean()), "ade_std": std}
        cells = [f"{v[0]:.2f}/{v[1]:.2f}" if v else "-" for v in vals]
        rows.append(f"| {s} | {' | '.join(cells)} | {ade.mean():.3f} ± {std:.3f} |")
    (ROOT / "results" / "seeds.json").write_text(
        json.dumps(
            {"splits": out, "table_markdown": "\n".join(rows)}, indent=1, ensure_ascii=False
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    ablation()
    train_cost()
    seeds()
    print("wrote results/ablation.json, train_cost.json, seeds.json")
