"""데모 재생 데이터·가중치 내보내기.

* ``select_window``: 테스트 장면의 충돌 라벨(``pair_labels``)로 "사전 경보 양성이 가장 많은 구역·60 s 창"을 고른다.
* ``build_replay``: 그 창의 2.5 Hz 프레임(구역 파티션 Parquet)을 읽어 프레임별 위치와, 8 프레임 이력이 있는
  에이전트의 모델 파라미터(12 × 5, PyTorch 로 계산)를 담는다. 페이지의 JS 모델이 같은 값을 다시 계산해
  최대 오차를 표시한다(파이썬 대비 검증). 일부 프레임에는 평균 궤적·결정적 위험도 넣어 두 번째 검증에 쓴다.
* ``export_weights``: ``SocialSTGCNN`` state_dict → JS 가 그대로 읽는 중첩 리스트(BatchNorm 은 eval 모드 파라미터).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from foresight.data.ethucy import SceneSet
from foresight.utils import get_logger

log = get_logger(__name__)
STEP_SECONDS = 0.4


@dataclass
class ReplayWindow:
    zone: int
    bin_lo: int  # 포함 (첫 예측에 필요한 obs_len-1 프레임 이력 포함)
    bin_hi: int  # 미포함
    n_positive: int  # 창 안에서 예측 시점을 갖는 양성 장면 수 (선택 근거)


def _zone_of_meta(meta: tuple[str, float]) -> int | None:
    name = str(meta[0])
    return int(name.split("=", 1)[1]) if name.startswith("zone=") else None


def select_window(
    scenes: SceneSet,
    d_safe: float = 1.0,
    window_bins: int = 150,
    zone: int | None = None,
) -> ReplayWindow:
    """양성(예측 시점에 d_safe 밖 → 4.8 s 안에 안으로) 장면이 가장 많이 몰린 구역·창. ``zone`` 을 주면 그 구역 안에서만."""
    from foresight.eval.collision import pair_labels

    obs_len, pred_len = scenes.obs_len, scenes.pred_len
    pos_by_zone: dict[int, dict[int, int]] = {}
    for r in pair_labels(scenes, d_safe):
        if not r.label:
            continue
        z = _zone_of_meta(scenes.meta[r.scene])
        if z is None or (zone is not None and z != zone):
            continue
        t_pred = int(scenes.meta[r.scene][1]) + obs_len - 1
        d = pos_by_zone.setdefault(z, {})
        d[t_pred] = d.get(t_pred, 0) + 1
    best: tuple[int, int, int] | None = None
    for z, by_bin in pos_by_zone.items():
        bins = np.array(sorted(by_bin))
        vals = np.array([by_bin[b] for b in bins])
        for b0 in bins:
            m = (bins >= b0) & (bins < b0 + window_bins)
            n = int(vals[m].sum())
            if best is None or n > best[0]:
                best = (n, int(z), int(b0))
    if best is None:  # 양성이 없으면(스모크 데이터) 에이전트가 가장 많은 구역의 첫 창
        zones: list[tuple[int, int]] = []
        for m in scenes.meta:
            z_m = _zone_of_meta(m)
            if z_m is not None and (zone is None or z_m == zone):
                zones.append((z_m, int(m[1])))
        if not zones:
            raise ValueError("RTLS 장면(meta 'zone=K')이 없습니다")
        counts: dict[int, int] = {}
        for z, _ in zones:
            counts[z] = counts.get(z, 0) + 1
        z_best = max(counts, key=lambda k: counts[k])
        b0 = min(b for z, b in zones if z == z_best) + obs_len - 1
        best = (0, z_best, b0)
    n, z, b0 = best
    # 창 시작 2 s 전부터 담아 첫 프레임부터 예측이 나오게 하고, 끝에는 정답 미래(pred_len)를 남긴다
    return ReplayWindow(z, b0 - (obs_len - 1), b0 + window_bins + pred_len, n)


def load_window_frames(frames_dir: Path, zone: int, bin_lo: int, bin_hi: int) -> Any:
    """구역 파티션 Parquet 에서 창 안의 프레임을 (bin, tag_id) 순으로 읽는다 (polars DataFrame)."""
    import polars as pl

    pattern = str(Path(frames_dir) / "part=*" / f"zone_id={zone}" / "*.parquet")
    return (
        pl.scan_parquet(pattern, hive_partitioning=False)
        .filter((pl.col("bin") >= bin_lo) & (pl.col("bin") < bin_hi))
        .select("tag_id", "bin", "x", "y", "agent_type")
        .sort("bin", "tag_id")
        .collect()
    )


def _bn(bn: Any) -> dict[str, Any]:
    return {
        "w": bn.weight.detach().tolist(),
        "b": bn.bias.detach().tolist(),
        "m": bn.running_mean.tolist(),
        "v": bn.running_var.tolist(),
        "eps": float(bn.eps),
    }


def export_weights(model: Any, source: str = "") -> dict[str, Any]:
    """``SocialSTGCNN`` → JS 순전파용 가중치 사전. 기본 구조(n_stgcnn=1, view 교환)만 지원한다."""
    if len(model.st_gcns) != 1:
        raise ValueError("데모 JS 는 n_stgcnn=1 만 지원합니다")
    if model.time_channel_swap != "view":
        raise ValueError("데모 JS 는 time_channel_swap='view' 만 지원합니다")
    blk = model.st_gcns[0]
    tcn_bn1, tcn_prelu, tcn_conv, tcn_bn2 = blk.tcn[0], blk.tcn[1], blk.tcn[2], blk.tcn[3]
    res = blk.residual
    st = {
        "gcn_w": blk.gcn.conv.weight.detach()[:, :, 0, 0].tolist(),  # (out, in)
        "gcn_b": blk.gcn.conv.bias.detach().tolist(),
        "bn1": _bn(tcn_bn1),
        "prelu1": float(tcn_prelu.weight.detach().reshape(-1)[0]),
        "tcn_w": tcn_conv.weight.detach()[:, :, :, 0].tolist(),  # (out, in, kt)
        "tcn_b": tcn_conv.bias.detach().tolist(),
        "bn2": _bn(tcn_bn2),
        "prelu_out": float(blk.prelu.weight.detach().reshape(-1)[0]),
    }
    if hasattr(res, "__len__") and len(res) == 2:  # Conv1x1 + BN
        st["res_w"] = res[0].weight.detach()[:, :, 0, 0].tolist()
        st["res_b"] = res[0].bias.detach().tolist()
        st["res_bn"] = _bn(res[1])
        st["res_kind"] = "conv"
    elif res.__class__.__name__ == "Identity":
        st["res_kind"] = "identity"
    else:
        st["res_kind"] = "zero"
    return {
        "meta": {
            "obs_len": model.obs_len,
            "pred_len": model.pred_len,
            "in_channels": int(blk.gcn.conv.weight.shape[1]),
            "out_channels": model.out_channels,
            "n_txpcnn": len(model.tpcnns),
            "t_kernel": int(tcn_conv.weight.shape[2]),
            "time_channel_swap": model.time_channel_swap,
            "param_count": model.num_parameters(),
            "source": source,
            "params": "(mu_x, mu_y, log_sx, log_sy, atanh_rho) — 상대 변위 이변량 가우시안",
        },
        "st_gcn": st,
        "tpcnns": [
            {"w": c.weight.detach().tolist(), "b": c.bias.detach().tolist()} for c in model.tpcnns
        ],
        "prelus": [float(p.weight.detach().reshape(-1)[0]) for p in model.prelus],
        "tpcnn_output": {
            "w": model.tpcnn_output.weight.detach().tolist(),
            "b": model.tpcnn_output.bias.detach().tolist(),
        },
    }


def _r(x: Any, nd: int = 5) -> Any:
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return _r(x.tolist(), nd)
    if isinstance(x, float):
        return round(x, nd)
    return x


def build_replay(
    frames: Any,
    model: Any,
    window: ReplayWindow,
    obs_len: int = 8,
    pred_len: int = 12,
    d_safe: float = 1.0,
    k: int = 20,
    seed: int = 0,
    check_every: int = 25,
    layout: dict[str, Any] | None = None,
    source: str = "",
) -> dict[str, Any]:
    """프레임 DataFrame(tag_id, bin, x, y, agent_type) + 모델 → 데모 재생 사전.

    frames[i] = {"t": 초, "a": [[agent_idx, x, y], ...]}; pred[i] = {"ids": [...], "params": [[12×5] ...]} (8 프레임 이력이 있는
    에이전트만). check[] 는 파이썬 평균 궤적·결정적 위험·샘플 위험(seed 고정)으로, JS 구현의 자체 검증에 쓴다.
    """
    import torch

    from foresight.eval import metrics as M
    from foresight.inference.predictor import TorchPredictor
    from foresight.serving.risk import pairwise_risk, risk_deterministic

    tag_ids = sorted(int(t) for t in frames["tag_id"].unique().to_list())
    idx_of = {t: i for i, t in enumerate(tag_ids)}
    types: dict[int, int] = {}
    for t, a in zip(frames["tag_id"].to_list(), frames["agent_type"].to_list()):
        types.setdefault(int(t), int(a))
    agents = [
        {"id": f"{'W' if types[t] == 0 else 'V'}{t}", "tag": t, "type": types[t]} for t in tag_ids
    ]
    bins = list(range(window.bin_lo, window.bin_hi))
    by_bin: dict[int, list[tuple[int, float, float]]] = {b: [] for b in bins}
    for t, b, x, y in zip(
        frames["tag_id"].to_list(),
        frames["bin"].to_list(),
        frames["x"].to_list(),
        frames["y"].to_list(),
    ):
        if b in by_bin:
            by_bin[int(b)].append((idx_of[int(t)], float(x), float(y)))
    hist: dict[
        int, list[tuple[int, float, float]]
    ] = {}  # agent idx → [(bin, x, y)] 최근 obs_len 개
    predictor = TorchPredictor(model)
    out_frames: list[dict[str, Any]] = []
    out_pred: list[dict[str, Any] | None] = []
    checks: list[dict[str, Any]] = []
    n_pred_frames = 0
    for fi, b in enumerate(bins):
        rows = by_bin[b]
        out_frames.append(
            {
                "t": round(fi * STEP_SECONDS, 1),
                "a": [[i, round(x, 4), round(y, 4)] for i, x, y in rows],
            }
        )
        present = set()
        for i, x, y in rows:
            present.add(i)
            h = hist.setdefault(i, [])
            if h and h[-1][0] != b - 1:
                h.clear()  # 프레임 결손 → 이력 리셋 (파이프라인의 세그먼트 규칙과 같다)
            h.append((b, x, y))
            if len(h) > obs_len:
                del h[0]
        for i in list(hist):
            if i not in present:
                hist[i].clear()
        ready = [i for i in sorted(hist) if len(hist[i]) == obs_len]
        if not ready:
            out_pred.append(None)
            continue
        obs = np.array(
            [[[x, y] for _, x, y in hist[i]] for i in ready], dtype=np.float64
        )  # (N, 8, 2)
        pred = predictor.predict(obs, k=0)
        params = pred.params  # (12, N, 5)
        out_pred.append(
            {"ids": ready, "params": _r(np.transpose(params, (1, 0, 2)), 5)}  # (N, 12, 5)
        )
        n_pred_frames += 1
        if n_pred_frames % check_every == 1:
            atypes = np.array([types[tag_ids[i]] for i in ready], dtype=np.int8)
            g = torch.Generator().manual_seed(seed)
            rel = M.sample_relative(torch.from_numpy(params), k, generator=g)
            last = torch.from_numpy(obs[:, -1].astype(np.float32))
            samples = M.relative_to_absolute(rel, last).permute(0, 2, 1, 3).numpy()  # (K, N, T, 2)
            rm = pairwise_risk(samples, atypes, d_safe)
            rd = risk_deterministic(pred.mean_abs, atypes, d_safe)
            checks.append(
                {
                    "f": fi,
                    "mean": _r(pred.mean_abs, 4),  # (N, 12, 2)
                    "risk_det": _r(
                        [
                            [int(ready[w]), int(ready[v]), float(rd.risk[wi, vi])]
                            for wi, w in enumerate(rd.worker_idx)
                            for vi, v in enumerate(rd.vehicle_idx)
                        ],
                        3,
                    ),
                    "risk": _r(
                        [
                            [int(ready[w]), int(ready[v]), float(rm.risk[wi, vi])]
                            for wi, w in enumerate(rm.worker_idx)
                            for vi, v in enumerate(rm.vehicle_idx)
                        ],
                        3,
                    ),
                }
            )
    return {
        "meta": {
            "zone": window.zone,
            "bin_lo": window.bin_lo,
            "bin_hi": window.bin_hi,
            "n_frames": len(bins),
            "step_s": STEP_SECONDS,
            "obs_len": obs_len,
            "pred_len": pred_len,
            "d_safe": d_safe,
            "k": k,
            "seed": seed,
            "n_pred_frames": n_pred_frames,
            "n_positive_scenes": window.n_positive,
            "source": source,
            "layout": layout or {},
        },
        "agents": agents,
        "frames": out_frames,
        "pred": out_pred,
        "check": checks,
    }


def zone_layout(
    zone: int, n_cols: int = 6, cell_m: float = 20.0, aisle_m: float = 10.0
) -> dict[str, Any]:
    """rtls_sim.PlantLayout 의 행 우선 구역 번호 → 구역 사각형(미터)과 통로 간격."""
    col, row = zone % n_cols, zone // n_cols
    return {
        "x0": col * cell_m,
        "y0": row * cell_m,
        "size": cell_m,
        "aisle_m": aisle_m,
        "n_cols": n_cols,
        "col": col,
        "row": row,
    }


def write_js(path: Path, name: str, obj: dict[str, Any]) -> int:
    """``window.<name> = {...};`` 형태로 저장 (file:// 에서도 <script> 로 읽힌다). 바이트 수를 돌려준다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"window.{name} = " + json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + ";\n"
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))
