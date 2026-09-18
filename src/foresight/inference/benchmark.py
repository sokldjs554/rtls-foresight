"""추론 벤치마크 — eager / torch.compile / ORT fp32 / ORT INT8, 장면 크기별·스레드별 지연과 처리량.

측정 원칙
* 지연에 **전처리(상대 변위 + 정규화 라플라시안)와 후처리(누적합 + K 샘플링)** 를 포함한다. 서비스는
  절대좌표를 받으므로 모델 시간만 재면 실제 지연을 과소평가한다. 세 구간을 따로 기록해 병목을 보인다.
* 실제 테스트 장면(ETH/UNIV)에서 N ∈ {2, 5, 10, 20, 40+} 장면을 골라 쓴다. 난수 입력은 라플라시안이
  실제와 달라 ORT 의 상수 접기/메모리 패턴이 달라질 수 있다.
* 웜업 20 회 뒤 ≥200 회 측정 (quick: 30). 설정당 시간 상한을 두어 병리적으로 느린 조합(4 스레드
  eager 의 OpenMP 스핀)이 전체 실행을 잡아먹지 않게 하고, 실제 반복 횟수를 기록한다.
* 스레드 1 / 4 를 모두 잰다. 7.6K 파라미터 모델은 연산량이 너무 작아 스레드가 늘면 동기화 비용이
  이득을 넘어선다 — 이를 수치로 남기는 것이 이 벤치마크의 목적 중 하나다.
* 측정 당시 부하(load average, 동시 학습 프로세스 수)를 메타데이터에 남긴다 — 공유 머신에서 잰 수치는
  그 맥락 없이는 재현할 수 없다.

스트리밍 워크로드
    "2.5 Hz 프레임 하나에 Z 개 구역": 구역마다 N=10 장면을 k=20 으로 예측하고 작업자–차량 위험을 계산한다.
    프레임당 시간이 400 ms 를 넘으면 소비자가 밀리므로 ``max_zones_at_2p5hz = 400 / per_zone_p95`` 를
    함께 낸다.

전처리 벡터화 검증
    공식 코드(``seq_to_graph``: 파이썬 이중 루프 + networkx ``normalized_laplacian_matrix``)를 그대로 옮긴
    참조 구현과 numpy 브로드캐스트 구현의 시간을 N=20, 57 에서 비교한다.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from foresight.data.ethucy import SceneSet
from foresight.data.graph import (
    inverse_distance_kernel,
    normalized_laplacian,
    relative_displacement,
)
from foresight.inference.export import FP32_NAME, INT8_NAME, load_model, resolve_checkpoint
from foresight.inference.onnx_backend import OnnxPredictor
from foresight.inference.predictor import Predictor, TorchPredictor
from foresight.utils import get_logger, project_root

log = get_logger("foresight.inference.benchmark")

PAPER_CLAIM = {
    "inference_time_s": 0.002,
    "params": 7563,
    "source": "Mohamed et al., CVPR 2020, Table 2 (Speed and parameter comparison)",
    "note": "논문은 측정 하드웨어·배치·전처리 포함 여부를 명시하지 않는다. 우리 수치는 CPU, 배치 1, 전처리·후처리 포함.",
}
BUCKETS: tuple[int, ...] = (2, 5, 10, 20, 40)
STEP_SECONDS = 0.4


# ----------------------------------------------------------------------------- workloads
def pick_workloads(data_dir: Path, buckets: tuple[int, ...] = BUCKETS) -> dict[str, dict[str, Any]]:
    """버킷별로 실제 테스트 장면 하나씩. 마지막 버킷은 '40+' (N ≥ 40 중 가장 큰 장면)."""
    pools: list[tuple[str, SceneSet]] = []
    for split in ("eth", "univ", "zara1", "zara2", "hotel"):
        p = data_dir / split / "test.npz"
        if p.exists():
            pools.append((split, SceneSet.load(p)))
    out: dict[str, dict[str, Any]] = {}
    for i, n in enumerate(buckets):
        last = i == len(buckets) - 1
        label = f"{n}+" if last else str(n)
        found = None
        for split, ss in pools:
            na = ss.num_agents
            idx = np.flatnonzero(na >= n) if last else np.flatnonzero(na == n)
            if len(idx):
                j = int(idx[np.argmax(na[idx])]) if last else int(idx[0])
                found = (split, j, ss.scene(j)[:, : ss.obs_len].copy())
                break
        if found is None:  # 데이터 없는 환경: 합성 장면
            rng = np.random.default_rng(n)
            obs = np.cumsum(rng.normal(0, 0.3, size=(n, 8, 2)), axis=1) + rng.uniform(
                -5, 5, size=(n, 1, 2)
            )
            found = ("synthetic", -1, obs)
        split, j, obs = found
        out[label] = {"split": split, "scene": j, "n_agents": int(obs.shape[0]), "obs": obs}
    return out


# ----------------------------------------------------------------------------- timing
def _percentiles(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs)
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
    }


def time_predictor(
    pred: Predictor,
    obs: np.ndarray,
    k: int,
    warmup: int,
    iters: int,
    budget_s: float,
    seed: int = 0,
) -> dict[str, Any]:
    for _ in range(warmup):
        pred.predict(obs, k=k, seed=seed)
    total: list[float] = []
    comp: dict[str, list[float]] = {"preprocess_ms": [], "model_ms": [], "postprocess_ms": []}
    t_start = time.perf_counter()
    for _ in range(iters):
        t0 = time.perf_counter()
        p = pred.predict(obs, k=k, seed=seed)
        total.append((time.perf_counter() - t0) * 1e3)
        for key in comp:
            comp[key].append(p.timing_ms.get(key, 0.0))
        if time.perf_counter() - t_start > budget_s:
            break
    res = _percentiles(total)
    res.update({f"{key}_p50": float(np.median(v)) for key, v in comp.items()})
    res["throughput_scenes_per_s"] = 1000.0 / res["mean_ms"]
    res["n_iters"] = len(total)
    return res


def _torch_threads(t: int) -> None:
    try:
        torch.set_num_threads(t)
    except RuntimeError:  # 일부 빌드는 병렬 초기화 뒤 변경을 거부한다 — 기록만 남기고 진행
        log.warning("torch.set_num_threads(%d) refused; current=%d", t, torch.get_num_threads())


def build_backends(
    model: torch.nn.Module, onnx_dir: Path, threads: int, include_compile: bool
) -> dict[str, Callable[[], Predictor]]:
    def eager() -> Predictor:
        _torch_threads(threads)
        return TorchPredictor(model, threads=threads)

    def compiled() -> Predictor:
        _torch_threads(threads)
        return TorchPredictor(model, compile=True, threads=threads)

    def ort_fp32() -> Predictor:
        _torch_threads(
            1
        )  # 후처리(샘플링)는 torch 로 돌므로 ORT 스레드 수와 별개로 1 스레드로 고정한다
        p = OnnxPredictor(onnx_dir / FP32_NAME, threads=threads, name="onnx-fp32")
        p.warmup()
        return p

    def ort_int8() -> Predictor:
        _torch_threads(1)
        p = OnnxPredictor(onnx_dir / INT8_NAME, threads=threads, name="onnx-int8")
        p.warmup()
        return p

    out: dict[str, Callable[[], Predictor]] = {"torch-eager": eager}
    if include_compile:
        out["torch-compile"] = compiled
    if (onnx_dir / FP32_NAME).exists():
        out["onnx-fp32"] = ort_fp32
    if (onnx_dir / INT8_NAME).exists():
        out["onnx-int8"] = ort_int8
    return out


def ensure_onnx(onnx_dir: Path, model: torch.nn.Module, data_dir: Path) -> None:
    """ONNX 산출물이 없으면 최소 구성(fp32 + txp-only INT8, 보정 64 장면)으로 만든다 — 벤치마크가 export 에 의존하지 않도록."""
    from foresight.inference.export import export_onnx

    fp32 = onnx_dir / FP32_NAME
    if not fp32.exists():
        export_onnx(model, fp32)  # type: ignore[arg-type]
    int8 = onnx_dir / INT8_NAME
    calib = data_dir / "eth" / "train.npz"
    if not int8.exists() and calib.exists():
        from foresight.inference.quantize import quantize_static_scenes

        quantize_static_scenes(fp32, int8, SceneSet.load(calib), n_calib=64, variant="txp-only")


# ----------------------------------------------------------------------------- streaming workload
def streaming_frame_benchmark(
    pred: Predictor, obs: np.ndarray, zones: int, iters: int, warmup: int = 3
) -> dict[str, Any]:
    """프레임 하나 = Z 구역 × (예측 k=20 + 작업자–차량 위험). 에이전트 타입은 번갈아 배정한다."""
    from foresight.serving.risk import pairwise_risk

    types = (np.arange(obs.shape[0]) % 2).astype(np.int8)

    def frame() -> float:
        t0 = time.perf_counter()
        for _ in range(zones):
            p = pred.predict(obs, k=20)
            assert p.samples_abs is not None
            pairwise_risk(p.samples_abs, types, d_safe=1.0)
        return (time.perf_counter() - t0) * 1e3

    for _ in range(warmup):
        frame()
    ts = [frame() for _ in range(iters)]
    res = _percentiles(ts)
    per_zone_p95 = res["p95_ms"] / zones
    return {
        "zones": zones,
        "n_per_zone": int(obs.shape[0]),
        "frame_p50_ms": res["p50_ms"],
        "frame_p95_ms": res["p95_ms"],
        "per_zone_p95_ms": per_zone_p95,
        "max_zones_at_2p5hz": int((STEP_SECONDS * 1e3) // per_zone_p95),
        "n_iters": len(ts),
    }


# ----------------------------------------------------------------------------- preprocessing reference
def networkx_seq_to_graph(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """공식 Social-STGCNN ``seq_to_graph`` 의 충실한 옮김 (파이썬 이중 루프 + networkx). 벤치마크 참조용."""
    try:
        import networkx as nx
    except ImportError as e:  # pragma: no cover — dev extra
        raise RuntimeError("networkx (dev extra) 가 있어야 공식 전처리와 비교할 수 있다") from e

    n, t = obs.shape[0], obs.shape[1]
    rel = relative_displacement(obs)  # (N, T, 2)
    v = np.zeros((t, n, 2))
    a = np.zeros((t, n, n))
    for s in range(t):
        step = rel[:, s]
        for h in range(n):
            v[s, h] = step[h]
            a[s, h, h] = 1
            for k in range(h + 1, n):
                d = float(np.linalg.norm(step[h] - step[k]))
                w = 1.0 / d if d != 0 else 0.0
                a[s, h, k] = w
                a[s, k, h] = w
        g = nx.from_numpy_array(a[s])
        a[s] = nx.normalized_laplacian_matrix(g).toarray()
    return v.astype(np.float32), a.astype(np.float32)


def numpy_seq_to_graph(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rel = relative_displacement(obs.astype(np.float64)).transpose(1, 0, 2)
    return rel.astype(np.float32), normalized_laplacian(inverse_distance_kernel(rel)).astype(
        np.float32
    )


def preprocess_benchmark(obs: np.ndarray, iters: int) -> dict[str, float]:
    def bench(fn: Callable[[np.ndarray], Any]) -> float:
        fn(obs)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(obs)
            ts.append((time.perf_counter() - t0) * 1e3)
        return float(np.median(ts))

    nx_ms = bench(networkx_seq_to_graph)
    np_ms = bench(numpy_seq_to_graph)
    _, a_nx = networkx_seq_to_graph(obs)
    _, a_np = numpy_seq_to_graph(obs)
    return {
        "n_agents": int(obs.shape[0]),
        "networkx_ms": nx_ms,
        "numpy_ms": np_ms,
        "speedup": nx_ms / np_ms,
        "max_abs_diff": float(np.abs(a_nx - a_np).max()),
    }


# ----------------------------------------------------------------------------- figures
PALETTE = {
    "torch-eager": "#2a78d6",
    "torch-compile": "#eb6834",
    "onnx-fp32": "#1baf7a",
    "onnx-int8": "#eda100",
}


def plot_latency(results: dict[str, Any], out_png: Path, threads: int = 1, k: str = "k20") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results["workloads"].keys())
    backends = [b for b in PALETTE if f"{b}/t{threads}" in results["backends"]]
    if not backends:
        return
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=140)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    width = 0.8 / len(backends)
    x = np.arange(len(labels))
    for i, b in enumerate(backends):
        rows = results["backends"][f"{b}/t{threads}"]
        p50 = [rows[lab][k]["p50_ms"] if lab in rows else np.nan for lab in labels]
        p95 = [rows[lab][k]["p95_ms"] if lab in rows else np.nan for lab in labels]
        xs = x + (i - (len(backends) - 1) / 2) * width
        ax.bar(xs, p50, width * 0.9, color=PALETTE[b], label=b, linewidth=0)
        ax.scatter(xs, p95, s=18, color=PALETTE[b], edgecolors="#fcfcfb", linewidths=1.2, zorder=3)
    # 범위가 20 배를 넘을 때만 로그축 (4 스레드 torch 의 100 배 차이를 한 그림에 담기 위해); 눈금은 평범한 숫자로
    all_p95 = [
        results["backends"][f"{b}/t{threads}"][lab][k]["p95_ms"]
        for b in backends
        for lab in labels
        if lab in results["backends"][f"{b}/t{threads}"]
    ]
    log_scale = max(all_p95) / max(min(all_p95), 1e-6) > 20
    if log_scale:
        from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

        ax.set_yscale("log")
        all_p50 = [
            results["backends"][f"{b}/t{threads}"][lab][k]["p50_ms"]
            for b in backends
            for lab in labels
            if lab in results["backends"][f"{b}/t{threads}"]
        ]
        lo, hi = min(min(all_p50), PAPER_CLAIM["inference_time_s"] * 1e3) * 0.7, max(all_p95) * 1.3
        ticks = [t for t in (0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if lo <= t <= hi]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(lo, hi)
    else:
        ax.set_ylim(0, max(all_p95) * 1.12)
    ax.set_xticks(
        x,
        [
            f"N={results['workloads'][lab]['n_agents']}" + (" (40+)" if lab.endswith("+") else "")
            for lab in labels
        ],
    )
    ax.set_ylabel(f"latency, ms{' (log)' if log_scale else ''} — bar p50, dot p95", color="#52514e")
    ax.set_title(
        f"End-to-end latency per scene ({k}, {threads} thread{'s' if threads > 1 else ''}, preprocess+model+postprocess)",
        color="#0b0b0b",
        fontsize=11,
        loc="left",
    )
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.axhline(PAPER_CLAIM["inference_time_s"] * 1e3, color="#52514e", linewidth=1, linestyle="--")
    ax.text(
        len(labels) - 0.5,
        PAPER_CLAIM["inference_time_s"] * 1e3 * 1.1,
        "paper claim 2 ms",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#52514e",
    )
    ax.legend(frameon=False, fontsize=9, ncol=len(backends), loc="upper left")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_breakdown(
    results: dict[str, Any], out_png: Path, threads: int = 1, label: str = "20"
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    backends = [
        b
        for b in PALETTE
        if f"{b}/t{threads}" in results["backends"]
        and label in results["backends"][f"{b}/t{threads}"]
    ]
    if not backends:
        return
    parts = [
        ("preprocess_ms_p50", "preprocess", "#2a78d6"),
        ("model_ms_p50", "model", "#eb6834"),
        ("postprocess_ms_p50", "postprocess (k=20 sampling)", "#1baf7a"),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=140)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    y = np.arange(len(backends))
    left = np.zeros(len(backends))
    for key, name, color in parts:
        vals = np.array(
            [results["backends"][f"{b}/t{threads}"][label]["k20"][key] for b in backends]
        )
        ax.barh(
            y,
            vals,
            left=left,
            height=0.5,
            color=color,
            label=name,
            linewidth=0,
            edgecolor="#fcfcfb",
        )
        left += vals
    for i in range(len(backends)):
        ax.text(
            left[i] + left.max() * 0.01,
            y[i],
            f"{left[i]:.2f} ms",
            va="center",
            fontsize=8,
            color="#0b0b0b",
        )
    ax.set_yticks(y, backends)
    ax.invert_yaxis()
    ax.set_xlabel("p50 per stage, ms", color="#52514e")
    ax.set_title(
        f"Where the time goes — N={results['workloads'][label]['n_agents']}, k=20, {threads} thread",
        color="#0b0b0b",
        fontsize=11,
        loc="left",
    )
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def _machine_meta() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "cpu": platform.processor() or platform.machine(),
        "n_cpu": os.cpu_count(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "onnxruntime": __import__("onnxruntime").__version__,
        "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY", "(unset)"),
        "load_avg_start": os.getloadavg(),
    }
    try:
        import psutil

        meta["cpu_model"] = platform.uname().processor or ""
        busy = [
            p.info["cmdline"]
            for p in psutil.process_iter(["cmdline"])
            if p.info["cmdline"]
            and "train" in p.info["cmdline"]
            and any("foresight" in c for c in p.info["cmdline"])
        ]
        meta["concurrent_training_processes"] = len(busy)
        meta["mem_total_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:  # psutil 없음 — 메타데이터일 뿐이므로 조용히 생략
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    meta["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return meta


def _bench_worker(args: list[str]) -> None:
    """서브프로세스 모드: OMP_WAIT_POLICY 같은 환경변수는 torch 로드 전에 정해져야 하므로 별도 프로세스에서 eager 만 잰다."""
    cfg = json.loads(args[0])
    model = load_model(Path(cfg["ckpt"]))
    _torch_threads(cfg["threads"])
    pred = TorchPredictor(model, threads=cfg["threads"])
    wl = pick_workloads(Path(cfg["data_dir"]))
    out = {}
    for label, w in wl.items():
        out[label] = {
            "n_agents": w["n_agents"],
            "threads": cfg["threads"],
            "k0": time_predictor(pred, w["obs"], 0, cfg["warmup"], cfg["iters"], cfg["budget_s"]),
            "k20": time_predictor(pred, w["obs"], 20, cfg["warmup"], cfg["iters"], cfg["budget_s"]),
        }
    sys.stdout.write("\n@@RESULT@@" + json.dumps(out))


def run_benchmark(
    out: Path,
    quick: bool = False,
    ckpt: Path | None = None,
    onnx_dir: Path | None = None,
    data_dir: Path | None = None,
    threads_list: tuple[int, ...] = (1, 4),
    zones: int = 8,
    figures_dir: Path | None = None,
) -> dict[str, Any]:
    """CLI ``foresight benchmark`` 진입점. 결과 JSON (backend → N → metrics) 과 PNG 를 쓴다."""
    root = project_root()
    out = Path(out)
    data_dir = data_dir or root / "data" / "processed" / "ethucy"
    onnx_dir = Path(onnx_dir) if onnx_dir else root / "artifacts" / "onnx"
    figures_dir = figures_dir or out.parent / "figures"
    warmup, iters, budget_s = (5, 30, 2.0) if quick else (20, 200, 10.0)
    ckpt_path = resolve_checkpoint(ckpt)
    model = load_model(ckpt_path)
    ensure_onnx(onnx_dir, model, data_dir)
    workloads = pick_workloads(data_dir)
    t_all = time.perf_counter()
    results: dict[str, Any] = {
        "meta": {
            **_machine_meta(),
            "quick": quick,
            "warmup": warmup,
            "iters": iters,
            "budget_s_per_config": budget_s,
            "checkpoint": str(ckpt_path),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "paper_claim": PAPER_CLAIM,
        "model": {"params": int(sum(p.numel() for p in model.parameters()))},
        "model_files": {
            "torch_ckpt_bytes": ckpt_path.stat().st_size,
            **{
                n: (onnx_dir / n).stat().st_size
                for n in (FP32_NAME, INT8_NAME)
                if (onnx_dir / n).exists()
            },
        },
        "workloads": {
            lab: {k: v for k, v in w.items() if k != "obs"} for lab, w in workloads.items()
        },
        "backends": {},
        "errors": {},
        "streaming": {},
        "preprocess": {},
    }
    for threads in threads_list:
        for name, factory in build_backends(
            model, onnx_dir, threads, include_compile=not quick
        ).items():
            key = f"{name}/t{threads}"
            try:
                t0 = time.perf_counter()
                pred = factory()
                if name == "torch-compile":  # 모든 N 을 한 번씩 돌려 재컴파일을 웜업 밖으로 뺀다
                    for w in workloads.values():
                        pred.predict(w["obs"], k=0)
                setup_s = time.perf_counter() - t0
            except Exception as e:  # torch.compile 이 이 빌드에서 실패해도 벤치마크는 계속
                results["errors"][key] = f"{type(e).__name__}: {str(e)[:400]}"
                log.warning("%s unavailable: %s", key, results["errors"][key])
                continue
            rows: dict[str, Any] = {}
            for label, w in workloads.items():
                rows[label] = {
                    "n_agents": w["n_agents"],
                    "threads": threads,
                    "setup_s": round(setup_s, 2),
                    "k0": time_predictor(pred, w["obs"], 0, warmup, iters, budget_s),
                    "k20": time_predictor(pred, w["obs"], 20, warmup, iters, budget_s),
                }
                log.info(
                    "%-22s N=%-3s k=0 p50 %.3f ms | k=20 p50 %.3f ms p95 %.3f (n=%d)",
                    key,
                    label,
                    rows[label]["k0"]["p50_ms"],
                    rows[label]["k20"]["p50_ms"],
                    rows[label]["k20"]["p95_ms"],
                    rows[label]["k20"]["n_iters"],
                )
            results["backends"][key] = rows
            if threads == 1:
                w10 = workloads.get("10") or next(iter(workloads.values()))
                results["streaming"][key] = streaming_frame_benchmark(
                    pred, w10["obs"], zones, iters=10 if quick else 50
                )
                log.info(
                    "%-22s streaming Z=%d frame p95 %.2f ms -> max zones %d",
                    key,
                    zones,
                    results["streaming"][key]["frame_p95_ms"],
                    results["streaming"][key]["max_zones_at_2p5hz"],
                )
            del pred
    # OpenMP 스핀 대기 정책을 바꾼 4 스레드 eager 재측정 (환경변수는 프로세스 시작 전에 정해져야 한다)
    if 4 in threads_list:
        cfg = {
            "ckpt": str(ckpt_path),
            "data_dir": str(data_dir),
            "threads": 4,
            "warmup": warmup,
            "iters": iters,
            "budget_s": budget_s,
        }
        env = {**os.environ, "OMP_WAIT_POLICY": "PASSIVE", "FORESIGHT_ROOT": str(root)}
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "foresight.inference.benchmark",
                    "--worker",
                    json.dumps(cfg),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
            if "@@RESULT@@" in proc.stdout:
                results["backends"]["torch-eager/t4-omp-passive"] = json.loads(
                    proc.stdout.split("@@RESULT@@", 1)[1]
                )
                log.info(
                    "torch-eager/t4-omp-passive N=20 k=20 p50 %.3f ms",
                    results["backends"]["torch-eager/t4-omp-passive"]["20"]["k20"]["p50_ms"],
                )
            else:
                results["errors"]["torch-eager/t4-omp-passive"] = proc.stderr[-400:]
        except Exception as e:
            results["errors"]["torch-eager/t4-omp-passive"] = f"{type(e).__name__}: {str(e)[:300]}"
    # 전처리: networkx 참조 vs numpy
    for label in ("20", "40+"):
        if label in workloads:
            results["preprocess"][str(workloads[label]["n_agents"])] = preprocess_benchmark(
                workloads[label]["obs"], iters=5 if quick else 20
            )
    results["meta"]["load_avg_end"] = os.getloadavg()
    results["meta"]["total_seconds"] = round(time.perf_counter() - t_all, 1)
    figs = []
    for threads in threads_list:
        p = figures_dir / f"benchmark_latency_t{threads}.png"
        plot_latency(results, p, threads=threads)
        if p.exists():
            figs.append(str(p.relative_to(root)) if p.is_relative_to(root) else str(p))
    p = figures_dir / "benchmark_breakdown_n20.png"
    plot_breakdown(results, p, threads=1, label="20")
    if p.exists():
        figs.append(str(p.relative_to(root)) if p.is_relative_to(root) else str(p))
    results["figures"] = figs
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(results, indent=1, ensure_ascii=False, default=float), encoding="utf-8"
    )
    log.info("benchmark -> %s (%.0fs)", out, results["meta"]["total_seconds"])
    return results


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        _bench_worker(sys.argv[2:])
    else:
        run_benchmark(project_root() / "results" / "benchmark.json", quick="--quick" in sys.argv)
