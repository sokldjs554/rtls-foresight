#!/usr/bin/env python
"""RTLS 대규모 처리 벤치마크.

한 프로파일(smoke / small / full)에 대해 네 단계를 **각각 별도 프로세스**로 실행하고 벽시계 시간·처리량·피크 RSS 를 기록한다.

(a) 합성 생성기 ``simulate``            — 파티션 Parquet 생성 처리량
(b) Polars lazy + streaming 파이프라인   — 원시 → 2.5 Hz → SceneSet, 피크 RSS 상한(< 4 GB) 검증
(c) naive pandas 경로 (상한 표본)        — 전부 메모리에 올리는 방식의 시간·메모리를 **표본에서 측정**하고 전체 행 수로 선형 외삽.
                                         외삽값은 ``extrapolated`` 로 명시한다 — 실측이 아니다.
(d) DuckDB 집계 통계                     — ``results/data_pipeline/stats*.json``

왜 프로세스를 나누나: ``ru_maxrss`` 는 프로세스 수명 전체의 최대라 단계별 피크를 정직하게 재려면 프로세스가 달라야 하고,
앞 단계의 할당·페이지 캐시가 다음 단계 수치를 오염시키지 않는다. 자식은 결과 JSON 을 ``@@RESULT@@`` 마커 뒤에 출력한다.

사용:
    python scripts/run_rtls_benchmark.py --profile small
    python scripts/run_rtls_benchmark.py --profile full --pandas-cap-rows 3000000
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MARK = "@@RESULT@@"
STAGES = ("simulate", "polars", "pandas", "duckdb")


def _ru_maxrss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _dir_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


# ---------------------------------------------------------------- 자식 프로세스에서 실행되는 단계들
def stage_simulate(a: argparse.Namespace) -> dict[str, Any]:
    from foresight.data.rtls_sim import simulate

    base = _ru_maxrss_mb()
    r = simulate(a.raw_dir, profile=a.profile, seed=a.seed)
    man = json.loads(r.manifest_path.read_text(encoding="utf-8"))
    return {
        "rows": r.rows,
        "seconds": round(r.seconds, 3),
        "rows_per_s": round(r.rows_per_s, 1),
        "files": r.files,
        "hours": r.config.hours,
        "tags": r.config.tags,
        "near_miss": r.n_events,
        "near_miss_per_vehicle_hour": man["near_miss"]["per_vehicle_hour"],
        "peak_rss_mb": round(_ru_maxrss_mb(), 1),
        "baseline_rss_mb": round(base, 1),
    }


def stage_polars(a: argparse.Namespace) -> dict[str, Any]:
    from foresight.data.rtls_pipeline import PipelineConfig, run_pipeline

    base = _ru_maxrss_mb()
    res = run_pipeline(a.raw_dir, a.out_dir, PipelineConfig())
    return {
        "rows": res.rows_raw,
        "rows_clean": res.rows_clean,
        "rows_frames": res.rows_frames,
        "seconds": round(res.seconds, 3),
        "rows_per_s": round(res.rows_raw / max(res.seconds, 1e-9), 1),
        "timings": {k: round(v, 3) for k, v in res.timings.items()},
        "scenes": res.scenes,
        "agent_windows": res.agents,
        "rss_stage_mb": res.rss_stage_mb,
        "peak_rss_mb": round(max(res.peak_rss_mb, _ru_maxrss_mb()), 1),
        "baseline_rss_mb": round(base, 1),
    }


def stage_pandas(a: argparse.Namespace) -> dict[str, Any]:
    """전부 메모리에 올리는 naive 경로. ``cap_rows`` 이상이 되는 첫 파일들만 읽어 측정하고 전체로 외삽한다."""
    import pandas as pd
    import pyarrow.parquet as pq

    from foresight.data.rtls_sim import layout_from_manifest, load_manifest

    base = _ru_maxrss_mb()
    layout = layout_from_manifest(load_manifest(a.raw_dir))
    files = sorted(Path(a.raw_dir).glob("date=*/hour=*/*.parquet"))
    total_rows = sum(pq.read_metadata(f).num_rows for f in files)
    chosen: list[Path] = []
    n = 0
    for f in files:
        chosen.append(f)
        n += pq.read_metadata(f).num_rows
        if n >= a.pandas_cap_rows:
            break
    t = time.perf_counter()
    df = pd.concat([pd.read_parquet(f) for f in chosen], ignore_index=True)
    df = df[df["quality"] >= 30].drop_duplicates(["tag_id", "ts_ms"])
    df["x"] = df["x"].clip(0.0, layout.width_m)
    df["y"] = df["y"].clip(0.0, layout.height_m)
    df["bin"] = df["ts_ms"] // 400
    frames = (
        df.groupby(["tag_id", "bin"], sort=False)
        .agg(x=("x", "mean"), y=("y", "mean"), n=("x", "size"), agent_type=("agent_type", "first"))
        .reset_index()
    )
    sec = time.perf_counter() - t
    peak = _ru_maxrss_mb()
    ratio = total_rows / n
    return {
        "measured_rows": n,
        "files": len(chosen),
        "seconds": round(sec, 3),
        "rows_per_s": round(n / sec, 1),
        "frames": len(frames),
        "peak_rss_mb": round(peak, 1),
        "baseline_rss_mb": round(base, 1),
        "capped": n < total_rows,
        "extrapolated": {
            "rows": total_rows,
            "seconds": round(sec * ratio, 1),
            "peak_rss_mb": round(base + (peak - base) * ratio, 1),
            "note": "표본 측정값의 선형 외삽 (시간·메모리 모두 행 수에 비례한다고 가정). 실측이 아니다.",
        },
    }


def stage_duckdb(a: argparse.Namespace) -> dict[str, Any]:
    from foresight.data.rtls_pipeline import compute_stats

    base = _ru_maxrss_mb()
    t = time.perf_counter()
    st = compute_stats(a.raw_dir, a.out_dir, a.stats_path)
    sec = time.perf_counter() - t
    return {
        "seconds": round(sec, 3),
        "rows": st["raw"]["rows"],
        "rows_per_s": round(st["raw"]["rows"] / sec, 1),
        "queries": st["queries"],
        "peak_rss_mb": round(_ru_maxrss_mb(), 1),
        "baseline_rss_mb": round(base, 1),
        "near_miss": st["near_miss"],
        "agents_per_zone_frame": st["frames"]["agents_per_zone_frame"],
    }


def run_child(stage: str, a: argparse.Namespace) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        stage,
        "--profile",
        a.profile,
        "--data-dir",
        str(a.data_dir),
        "--seed",
        str(a.seed),
        "--pandas-cap-rows",
        str(a.pandas_cap_rows),
        "--stats-path",
        str(a.stats_path),
    ]
    t = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    wall = time.perf_counter() - t
    sys.stderr.write(proc.stderr[-4000:])
    if proc.returncode != 0:
        raise RuntimeError(
            f"stage {stage} failed (rc={proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
        )
    line = next(ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith(MARK))
    out = json.loads(line[len(MARK) :])
    out["wall_seconds_incl_startup"] = round(wall, 3)
    return out


# ---------------------------------------------------------------- 그림
def _fmt(v: float) -> str:
    return f"{v / 1e6:.2g}M" if v >= 1e6 else (f"{v / 1e3:.0f}K" if v >= 1e3 else f"{v:.0f}")


def plot(bench: dict[str, Any], figure: Path) -> None:
    """프로파일별 처리량(rows/s)과 피크 RSS 막대.

    색은 dataviz 기본 팔레트의 범주 슬롯을 고정 순서로 쓴다(1 blue Polars, 2 orange pandas, 3 aqua 생성기).
    pandas 는 상한 표본에서 실측한 값을 진한 막대로, 전체 행 수로 외삽한 값을 같은 색의 연한 막대로 그려
    "측정" 과 "추정" 을 한 축 위에서 구분한다.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    ink, muted, grid, baseline, surface = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    series = [
        ("polars", "Polars lazy + streaming (measured)", "#2a78d6"),
        ("pandas", "pandas naive (measured on capped subset)", "#eb6834"),
        ("simulate", "Synthetic generator, write path (measured)", "#1baf7a"),
    ]
    profiles = [p for p in ("smoke", "small", "full") if p in bench.get("profiles", {})]
    if not profiles:
        return
    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(1, 2, figsize=(12, 1.6 + 1.2 * len(profiles)), facecolor=surface)
    panels = [
        ("rows_per_s", "Throughput (higher is better)", "rows / s"),
        ("peak_rss_mb", "Peak RSS (lower is better)", "MB"),
    ]
    h = 0.22
    for ax, (metric, title, unit) in zip(axes, panels):
        ax.set_facecolor(surface)
        y = np.arange(len(profiles))
        xmax = 0.0
        for k, (key, _label, color) in enumerate(series):
            pos = y + (k - 1) * (h + 0.04)
            for yy, prof in zip(pos, profiles):
                st = bench["profiles"][prof]["stages"].get(key, {})
                v = float(st.get(metric, 0.0))
                if v <= 0:
                    continue
                ax.barh(yy, v, height=h, color=color, linewidth=0)
                label = "  " + _fmt(v)
                ext = (
                    st.get("extrapolated")
                    if (key == "pandas" and st.get("capped") and metric == "peak_rss_mb")
                    else None
                )
                if ext:  # 외삽값: 같은 색의 연한 막대 + 명시 라벨
                    ev = float(ext["peak_rss_mb"])
                    ax.barh(yy, ev, height=h, color=color, alpha=0.25, linewidth=0, zorder=0)
                    label = f"  {_fmt(v)} measured on {st['measured_rows'] / 1e6:.1f}M rows → ≈{_fmt(ev)} extrapolated"
                    v = ev
                ax.text(v, yy, label, va="center", ha="left", fontsize=8, color=ink)
                xmax = max(xmax, v)
        if metric == "peak_rss_mb":
            ax.axvline(4096, color=muted, linewidth=1)
            ax.text(4096, -0.62, "4 GB budget", color=muted, fontsize=8, ha="center", va="bottom")
            xmax = max(xmax, 4096)
        rows = [bench["profiles"][p]["stages"].get("simulate", {}).get("rows", 0) for p in profiles]
        ax.set_yticks(
            y,
            [
                f"{p}\n{r / 1e6:.1f}M rows" if r >= 1e5 else f"{p}\n{r:,} rows"
                for p, r in zip(profiles, rows)
            ],
            fontsize=9,
            color=ink,
        )
        ax.set_ylim(len(profiles) - 0.5, -0.7)
        ax.set_title(title, loc="left", fontsize=11, color=ink, fontweight="bold")
        ax.set_xlabel(unit, color=muted, fontsize=9)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: _fmt(v) if v else "0"))
        ax.xaxis.grid(True, color=grid, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(baseline)
        ax.tick_params(axis="x", colors=muted, labelsize=8)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, xmax * (1.75 if metric == "peak_rss_mb" else 1.3))
    handles = [Patch(color=c, label=lb) for _k, lb, c in series] + [
        Patch(
            color="#eb6834",
            alpha=0.25,
            label="pandas, linear extrapolation to full row count (not measured)",
        )
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=8,
        frameon=False,
        labelcolor=ink,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        "RTLS data pipeline benchmark — 4 cores / 15 GB, each stage in its own process",
        x=0.01,
        ha="left",
        fontsize=10,
        color=ink,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.97))
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=150, facecolor=surface)
    plt.close(fig)


# ---------------------------------------------------------------- 부모 프로세스
def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--profile", choices=("smoke", "small", "full"), required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "rtls")
    p.add_argument(
        "--out", type=Path, default=ROOT / "results" / "data_pipeline" / "benchmark.json"
    )
    p.add_argument(
        "--figure", type=Path, default=ROOT / "results" / "figures" / "data_pipeline_throughput.png"
    )
    p.add_argument("--stats-path", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pandas-cap-rows", type=int, default=3_000_000)
    p.add_argument(
        "--keep-raw", action="store_true", help="원시 데이터가 3 GB 를 넘어도 지우지 않는다"
    )
    p.add_argument("--raw-delete-threshold-gb", type=float, default=3.0)
    p.add_argument("--stage", choices=STAGES, default=None, help="(내부용) 자식 프로세스 단계")
    p.add_argument(
        "--plot-only", action="store_true", help="benchmark.json 으로 그림만 다시 그린다"
    )
    a = p.parse_args(argv)
    if a.plot_only:
        plot(json.loads(a.out.read_text(encoding="utf-8")), a.figure)
        return {}
    a.raw_dir = a.data_dir / a.profile / "raw"
    a.out_dir = a.data_dir / a.profile / "processed"
    if a.stats_path is None:
        a.stats_path = ROOT / "results" / "data_pipeline" / f"stats_{a.profile}.json"

    if a.stage:  # 자식
        fn = {
            "simulate": stage_simulate,
            "polars": stage_polars,
            "pandas": stage_pandas,
            "duckdb": stage_duckdb,
        }[a.stage]
        print(MARK + json.dumps(fn(a), ensure_ascii=False), flush=True)
        return {}

    import duckdb
    import numpy
    import pandas
    import polars
    import psutil
    import pyarrow

    a.raw_dir.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "profile": a.profile,
        "seed": a.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": {
            "cpu_count": os.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "polars": polars.__version__,
            "pyarrow": pyarrow.__version__,
            "duckdb": duckdb.__version__,
            "pandas": pandas.__version__,
        },
        "stages": {},
    }
    t_all = time.perf_counter()
    for stage in STAGES:
        sys.stderr.write(f"\n=== [{a.profile}] stage {stage} ===\n")
        result["stages"][stage] = run_child(stage, a)
        sys.stderr.write(
            f"=== {stage}: {json.dumps({k: v for k, v in result['stages'][stage].items() if k in ('rows', 'seconds', 'rows_per_s', 'peak_rss_mb')})}\n"
        )
    result["total_wall_seconds"] = round(time.perf_counter() - t_all, 1)
    raw_bytes = _dir_bytes(a.raw_dir)
    result["disk"] = {
        "raw_bytes": raw_bytes,
        "raw_stream_bytes": raw_bytes - _dir_bytes(a.raw_dir / "_truth"),
        "frames_bytes": _dir_bytes(a.out_dir / "frames_2p5hz"),
        "npz_bytes": sum(_dir_bytes(a.out_dir / f"{s}.npz") for s in ("train", "val", "test")),
        "raw_deleted": False,
    }
    if not a.keep_raw and raw_bytes > a.raw_delete_threshold_gb * 2**30:
        shutil.rmtree(a.raw_dir)
        result["disk"]["raw_deleted"] = True
        sys.stderr.write(
            f"raw data ({raw_bytes / 2**30:.2f} GB) deleted; processed npz + stats kept\n"
        )
    # 통계 JSON 의 최신본을 stats.json 으로도 복사 (문서·CI 가 고정 경로를 참조)
    if a.stats_path.exists():
        shutil.copyfile(a.stats_path, a.stats_path.parent / "stats.json")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    bench = json.loads(a.out.read_text(encoding="utf-8")) if a.out.exists() else {"profiles": {}}
    bench["profiles"][a.profile] = result
    a.out.write_text(json.dumps(bench, indent=1, ensure_ascii=False), encoding="utf-8")
    plot(bench, a.figure)
    sys.stderr.write(f"\nwrote {a.out} and {a.figure}\n")
    return result


if __name__ == "__main__":
    main()
