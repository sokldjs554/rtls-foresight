"""체크포인트 평가 — 세 프로토콜(ADE/FDE) + CVM 기준선 + 논문 수치 비교표.

재현 실험의 원칙
* 샘플링 시드를 고정하고, 시드 여러 개로 평균±표준편차를 낸다 (best-of-20 은 확률적이다).
* 공식 체크포인트를 같은 평가기로 돌린 열을 함께 둔다 — "우리 학습이 문제인지, 평가/데이터가 문제인지"를 가른다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from foresight.data.dataset import SceneGraphDataset
from foresight.eval import metrics as M
from foresight.models import SocialSTGCNN, constant_velocity

# Social-STGCNN, CVPR 2020, Table 1 (ADE/FDE, m, best-of-20). 평균 행은 논문이 보고한 값 그대로.
PAPER_TABLE1: dict[str, tuple[float, float]] = {
    "eth": (0.64, 1.11),
    "hotel": (0.49, 0.85),
    "univ": (0.44, 0.79),
    "zara1": (0.34, 0.53),
    "zara2": (0.30, 0.48),
    "avg": (0.44, 0.75),
}
# 같은 표의 S-GAN (Gupta et al., CVPR 2018) 행 — 논문이 비교 기준으로 쓴 값.
PAPER_SGAN: dict[str, tuple[float, float]] = {
    "eth": (0.81, 1.52),
    "hotel": (0.72, 1.61),
    "univ": (0.60, 1.26),
    "zara1": (0.34, 0.69),
    "zara2": (0.42, 0.84),
    "avg": (0.58, 1.18),
}


@dataclass
class EvalResult:
    split: str
    n_scenes: int
    n_agents: int
    k: int
    seeds: list[int]
    best_of_k_per_agent: dict[str, float]  # ade/fde 평균 (시드 평균)
    best_of_k_per_agent_std: dict[str, float] = field(default_factory=dict)
    best_of_k_joint: dict[str, float] = field(default_factory=dict)
    deterministic: dict[str, float] = field(default_factory=dict)
    cvm: dict[str, float] = field(default_factory=dict)
    cvm_sampled: dict[str, float] = field(default_factory=dict)


@torch.no_grad()
def evaluate_model(
    model: SocialSTGCNN,
    ds: SceneGraphDataset,
    k: int = 20,
    seeds: tuple[int, ...] = (0, 1, 2),
    split: str = "",
) -> EvalResult:
    model.eval()
    per_seed_pa: list[dict[str, float]] = []
    per_seed_joint: list[dict[str, float]] = []
    det: list[M.DisplacementErrors] = []
    cvm: list[M.DisplacementErrors] = []
    cvm_s: list[M.DisplacementErrors] = []
    # 모델 출력은 시드와 무관하므로 한 번만 계산해 둔다.
    outputs = []
    for b in ds.iter_scenes():
        params = model.predict_params(b.v_obs, b.a_obs)[0]  # (T_pred, N, 5)
        gt_abs = b.pos[0, :, ds.obs_len :].permute(1, 0, 2)  # (T_pred, N, 2)
        last = b.last_obs[0]
        outputs.append((params, gt_abs, last, b.pos[0].numpy()))
        det.append(M.deterministic(params, last, gt_abs))
        pos = b.pos[0].numpy().astype(np.float64)
        cv = (
            torch.from_numpy(constant_velocity(pos[:, : ds.obs_len], ds.scenes.pred_len, k=1))
            .permute(0, 2, 1, 3)
            .float()
        )
        cvm.append(M.best_of_k_per_agent(cv, gt_abs))
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        rng = np.random.default_rng(seed)
        pa: list[M.DisplacementErrors] = []
        jo: list[M.DisplacementErrors] = []
        cvs: list[M.DisplacementErrors] = []
        for params, gt_abs, last, pos in outputs:
            rel = M.sample_relative(params, k, generator=g)
            pred_abs = M.relative_to_absolute(rel, last)
            pa.append(M.best_of_k_per_agent(pred_abs, gt_abs))
            jo.append(M.best_of_k_joint(pred_abs, gt_abs))
            cv_s = constant_velocity(
                pos[:, : ds.obs_len].astype(np.float64),
                ds.scenes.pred_len,
                k=k,
                angle_std_deg=25.0,
                rng=rng,
            )
            cvs.append(
                M.best_of_k_per_agent(torch.from_numpy(cv_s).permute(0, 2, 1, 3).float(), gt_abs)
            )
        per_seed_pa.append(M.summarize(pa))
        per_seed_joint.append(M.summarize(jo))
        cvm_s.append(
            M.DisplacementErrors(
                np.concatenate([c.ade for c in cvs]), np.concatenate([c.fde for c in cvs])
            )
        )
    ade_pa = np.array([s["ade"] for s in per_seed_pa])
    fde_pa = np.array([s["fde"] for s in per_seed_pa])
    return EvalResult(
        split=split,
        n_scenes=len(ds),
        n_agents=int(per_seed_pa[0]["n_agents"]),
        k=k,
        seeds=list(seeds),
        best_of_k_per_agent={"ade": float(ade_pa.mean()), "fde": float(fde_pa.mean())},
        best_of_k_per_agent_std={"ade": float(ade_pa.std()), "fde": float(fde_pa.std())},
        best_of_k_joint={
            "ade": float(np.mean([s["ade"] for s in per_seed_joint])),
            "fde": float(np.mean([s["fde"] for s in per_seed_joint])),
        },
        deterministic=M.summarize(det),
        cvm=M.summarize(cvm),
        cvm_sampled={
            "ade": float(np.mean([c.ade.mean() for c in cvm_s])),
            "fde": float(np.mean([c.fde.mean() for c in cvm_s])),
        },
    )


def load_model(ckpt: Path, **model_kw: object) -> SocialSTGCNN:
    """학습 스크립트가 저장한 체크포인트(``{"state_dict", "config"}``) 또는 공식 state_dict 를 읽는다."""
    from foresight.models import official_key_map

    obj = torch.load(ckpt, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        cfg = dict(obj.get("model", {}))
        cfg.update(model_kw)
        cfg.pop("graph", None)  # 그래프 커널 설정은 데이터셋 쪽 옵션 — 모델 생성자 인자가 아니다
        model = SocialSTGCNN(**cfg)  # type: ignore[arg-type]
        model.load_state_dict(obj["state_dict"])
        return model
    model = SocialSTGCNN(**model_kw)  # type: ignore[arg-type]
    model.load_state_dict(official_key_map(obj))
    return model


def reproduction_table(
    results: dict[str, EvalResult], official: dict[str, EvalResult] | None = None
) -> str:
    """마크다운 재현표. avg 행은 5개 분할의 단순 평균 (논문과 같은 방식)."""
    rows = [
        "| 분할 | 논문 (ADE/FDE) | 공식 ckpt 재평가 | 우리 학습 best-of-20 | joint best-of-20 | 결정적(μ) | CVM | CVM-S(20) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def f(d: dict[str, float]) -> str:
        return f"{d['ade']:.2f}/{d['fde']:.2f}" if d else "-"

    keys = [k for k in ("eth", "hotel", "univ", "zara1", "zara2") if k in results]
    for k in keys:
        r = results[k]
        o = official[k] if official and k in official else None
        rows.append(
            f"| {k} | {PAPER_TABLE1[k][0]:.2f}/{PAPER_TABLE1[k][1]:.2f} | {f(o.best_of_k_per_agent) if o else '-'} | "
            f"{r.best_of_k_per_agent['ade']:.2f}±{r.best_of_k_per_agent_std['ade']:.2f}/{r.best_of_k_per_agent['fde']:.2f}±{r.best_of_k_per_agent_std['fde']:.2f} | "
            f"{f(r.best_of_k_joint)} | {f(r.deterministic)} | {f(r.cvm)} | {f(r.cvm_sampled)} |"
        )
    if len(keys) == 5:

        def avg(get: str) -> str:
            a = np.mean([getattr(results[k], get)["ade"] for k in keys])
            b = np.mean([getattr(results[k], get)["fde"] for k in keys])
            return f"{a:.2f}/{b:.2f}"

        oa = "-"
        if official and all(k in official for k in keys):
            oa = f"{np.mean([official[k].best_of_k_per_agent['ade'] for k in keys]):.2f}/{np.mean([official[k].best_of_k_per_agent['fde'] for k in keys]):.2f}"
        rows.append(
            f"| **avg** | **{PAPER_TABLE1['avg'][0]:.2f}/{PAPER_TABLE1['avg'][1]:.2f}** | {oa} | **{avg('best_of_k_per_agent')}** | {avg('best_of_k_joint')} | {avg('deterministic')} | {avg('cvm')} | {avg('cvm_sampled')} |"
        )
    return "\n".join(rows)


def save_results(
    path: Path, results: dict[str, EvalResult], official: dict[str, EvalResult] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "paper": {k: {"ade": v[0], "fde": v[1]} for k, v in PAPER_TABLE1.items()},
        "sgan_paper": {k: {"ade": v[0], "fde": v[1]} for k, v in PAPER_SGAN.items()},
        "ours": {k: asdict(v) for k, v in results.items()},
        "official_checkpoint": {k: asdict(v) for k, v in (official or {}).items()},
        "table_markdown": reproduction_table(results, official),
    }
    keys = [k for k in ("eth", "hotel", "univ", "zara1", "zara2") if k in results]
    if len(keys) == 5:
        payload["ours"]["avg"] = {
            "best_of_k_per_agent": {
                m: float(np.mean([results[k].best_of_k_per_agent[m] for k in keys]))
                for m in ("ade", "fde")
            },
            "best_of_k_joint": {
                m: float(np.mean([results[k].best_of_k_joint[m] for k in keys]))
                for m in ("ade", "fde")
            },
            "deterministic": {
                m: float(np.mean([results[k].deterministic[m] for k in keys]))
                for m in ("ade", "fde")
            },
            "cvm": {m: float(np.mean([results[k].cvm[m] for k in keys])) for m in ("ade", "fde")},
            "cvm_sampled": {
                m: float(np.mean([results[k].cvm_sampled[m] for k in keys])) for m in ("ade", "fde")
            },
        }
        if official and all(k in official for k in keys):
            payload["official_checkpoint"]["avg"] = {
                "best_of_k_per_agent": {
                    m: float(np.mean([official[k].best_of_k_per_agent[m] for k in keys]))
                    for m in ("ade", "fde")
                }
            }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
