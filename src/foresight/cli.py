"""``foresight`` 명령행 진입점 (typer).

무거운 모듈(torch, mlflow, onnxruntime, fastapi)은 각 명령 안에서 import 한다 — ``foresight --help`` 가
1초 안에 떠야 하고, 서빙 이미지에는 학습 의존성이 없기 때문이다.
"""

from __future__ import annotations

from pathlib import Path

import typer

from foresight.utils import get_logger, project_root

app = typer.Typer(
    add_completion=False, help="RTLS 궤적 예측 · 충돌 사전 경보 파이프라인", no_args_is_help=True
)
log = get_logger("foresight.cli")

ALL_SPLITS = "eth,hotel,univ,zara1,zara2"


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else project_root() / p


@app.command()
def download(
    raw_dir: str = "data/raw/ethucy", splits: str = ALL_SPLITS, force: bool = False
) -> None:
    """ETH/UCY 원본(Social-GAN 포맷)을 받아 sha256 으로 검증한다."""
    from foresight.data.download import download_ethucy

    download_ethucy(_abs(raw_dir), tuple(splits.split(",")), force=force)


@app.command()
def prepare(
    raw_dir: str = "data/raw/ethucy",
    out_dir: str = "data/processed/ethucy",
    splits: str = ALL_SPLITS,
    obs_len: int = 8,
    pred_len: int = 12,
    skip: int = 1,
    min_ped: int = 1,
) -> None:
    """원본 → 장면 npz (분할/서브셋별) + EDA 용 long parquet."""
    from foresight.data.ethucy import build_scenes
    from foresight.utils import Timer

    for split in splits.split(","):
        for subset in ("train", "val", "test"):
            src = _abs(raw_dir) / split / subset
            with Timer() as t:
                scenes = build_scenes(src, obs_len, pred_len, skip, min_ped)
            dst = _abs(out_dir) / split / f"{subset}.npz"
            scenes.save(dst)
            scenes.to_long_frame().write_parquet(dst.with_suffix(".parquet"))
            log.info(
                "%s/%s: %d scenes, %d agents, max N=%d (%.2fs) -> %s",
                split,
                subset,
                len(scenes),
                len(scenes.pos),
                int(scenes.num_agents.max()) if len(scenes) else 0,
                t.elapsed,
                dst,
            )


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train(ctx: typer.Context) -> None:
    """Hydra 오버라이드로 학습: foresight train dataset=eth train=paper seed=0"""
    from foresight.train.train import compose_config, run_training

    cfg = compose_config(list(ctx.args))
    run_training(cfg)


@app.command()
def evaluate(
    ckpt_dir: str = "results/checkpoints",
    data_dir: str = "data/processed/ethucy",
    out: str = "results/reproduction.json",
    splits: str = ALL_SPLITS,
    seeds: str = "0,1,2",
    k: int = 20,
    official: bool = True,
    threads: int = 1,
    ckpt_seed: int = 0,
) -> None:
    """학습된 체크포인트(+공식 체크포인트)를 같은 평가기로 평가해 재현표를 만든다."""
    import torch

    from foresight.data.dataset import SceneGraphDataset
    from foresight.eval.evaluate import (
        EvalResult,
        evaluate_model,
        load_model,
        reproduction_table,
        save_results,
    )
    from foresight.models import SocialSTGCNN, load_official_checkpoint

    torch.set_num_threads(threads)
    seed_list = tuple(int(s) for s in seeds.split(","))
    ours: dict[str, EvalResult] = {}
    offi: dict[str, EvalResult] = {}
    for split in splits.split(","):
        ds = SceneGraphDataset.from_npz(_abs(data_dir) / split / "test.npz")
        ck = _abs(ckpt_dir) / split / f"seed{ckpt_seed}" / "best.pth"
        if ck.exists():
            ours[split] = evaluate_model(load_model(ck), ds, k=k, seeds=seed_list, split=split)
            log.info("%s ours: %s", split, ours[split].best_of_k_per_agent)
        if official:
            m = load_official_checkpoint(
                SocialSTGCNN(),
                str(_abs("assets/official_checkpoints") / f"social-stgcnn-{split}.pth"),
            )
            offi[split] = evaluate_model(m, ds, k=k, seeds=seed_list, split=split)
            log.info("%s official: %s", split, offi[split].best_of_k_per_agent)
    save_results(_abs(out), ours, offi)
    typer.echo(reproduction_table(ours, offi))


@app.command("evaluate-rtls")
def evaluate_rtls(
    ckpts: str = "results/checkpoints/eth/seed0/best.pth",
    names: str = "zero-shot-eth",
    data_dir: str = "data/processed/rtls",
    out: str = "results/rtls_transfer.json",
    collision_out: str = "results/collision_eval.json",
    d_safe: float = 1.0,
    k: int = 20,
    seeds: str = "0,1,2",
    max_scenes: int | None = None,
    every: int = 1,
    threads: int = 1,
    figure: str = "results/figures/collision_pr.png",
) -> None:
    """RTLS 테스트에서 (1) 궤적 ADE/FDE 3 프로토콜 + CVM (2) 충돌 경보 품질(AP/AUROC/F1/선행시간)을 평가한다."""
    import numpy as np
    import torch

    from foresight.data.dataset import SceneGraphDataset
    from foresight.data.ethucy import SceneSet
    from foresight.eval.collision import evaluate_collision, plot_pr_curves, save_collision_eval
    from foresight.eval.evaluate import EvalResult, evaluate_model, load_model
    from foresight.inference.predictor import Predictor, TorchPredictor

    torch.set_num_threads(threads)
    scenes = SceneSet.load(_abs(data_dir) / "test.npz")
    if (
        every > 1
    ):  # 시간 순 장면을 every 개마다 하나씩 (skip=1 로 만든 테스트는 인접 장면이 거의 같다)
        keep = np.arange(0, len(scenes), every)
        counts = scenes.num_agents[keep]
        starts = scenes.scene_index[keep, 0]
        sel = np.concatenate([np.arange(s0, s0 + c) for s0, c in zip(starts, counts)])
        ends = np.cumsum(counts)
        scenes = SceneSet(
            pos=scenes.pos[sel],
            scene_index=np.stack([ends - counts, ends], axis=1),
            meta=[scenes.meta[i] for i in keep],
            obs_len=scenes.obs_len,
            pred_len=scenes.pred_len,
            agent_type=None if scenes.agent_type is None else scenes.agent_type[sel],
        )
    if max_scenes is not None and max_scenes < len(scenes):
        end = int(scenes.scene_index[max_scenes - 1, 1])
        scenes = SceneSet(
            pos=scenes.pos[:end],
            scene_index=scenes.scene_index[:max_scenes],
            meta=scenes.meta[:max_scenes],
            obs_len=scenes.obs_len,
            pred_len=scenes.pred_len,
            agent_type=scenes.agent_type[:end] if scenes.agent_type is not None else None,
        )
    ds = SceneGraphDataset(scenes)
    seed_list = tuple(int(s) for s in seeds.split(","))
    traj: dict[str, dict] = {}
    predictors: dict[str, Predictor | None] = {}
    for name, ck in zip(names.split(","), ckpts.split(",")):
        model = load_model(_abs(ck))
        res: EvalResult = evaluate_model(model, ds, k=k, seeds=seed_list, split="rtls")
        traj[name] = {
            "ckpt": ck,
            "best_of_k_per_agent": res.best_of_k_per_agent,
            "best_of_k_per_agent_std": res.best_of_k_per_agent_std,
            "best_of_k_joint": res.best_of_k_joint,
            "deterministic": res.deterministic,
            "cvm": res.cvm,
            "cvm_sampled": res.cvm_sampled,
            "n_scenes": res.n_scenes,
            "n_agents": res.n_agents,
        }
        predictors[name] = TorchPredictor(model)
        log.info(
            "%s: bo20 %.3f/%.3f det %.3f/%.3f cvm %.3f/%.3f",
            name,
            res.best_of_k_per_agent["ade"],
            res.best_of_k_per_agent["fde"],
            res.deterministic["ade"],
            res.deterministic["fde"],
            res.cvm["ade"],
            res.cvm["fde"],
        )
    rows = [
        "| 모델 | best-of-20 (per-agent) | joint best-of-20 | 결정적(μ) | CVM | CVM-S(20) |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in traj.items():
        rows.append(
            f"| {name} | {r['best_of_k_per_agent']['ade']:.2f}/{r['best_of_k_per_agent']['fde']:.2f} | "
            f"{r['best_of_k_joint']['ade']:.2f}/{r['best_of_k_joint']['fde']:.2f} | "
            f"{r['deterministic']['ade']:.2f}/{r['deterministic']['fde']:.2f} | "
            f"{r['cvm']['ade']:.2f}/{r['cvm']['fde']:.2f} | {r['cvm_sampled']['ade']:.2f}/{r['cvm_sampled']['fde']:.2f} |"
        )
    _abs(out).parent.mkdir(parents=True, exist_ok=True)
    _abs(out).write_text(
        __import__("json").dumps(
            {
                "d_safe": d_safe,
                "n_scenes": len(scenes),
                "results": traj,
                "table_markdown": "\n".join(rows),
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    ev = evaluate_collision(
        scenes, {**predictors, "_baselines": None}, d_safe=d_safe, k=k, sample_fraction=1.0 / every
    )
    save_collision_eval(ev, _abs(collision_out))
    for m, e in ev.methods.items():
        log.info(
            "collision %-16s AP %.3f AUROC %.3f bestF1 %.3f @%s (P %.2f R %.2f lead %.2fs)",
            m,
            e["ap"],
            e["auroc"],
            e["best_f1"]["f1"],
            e["best_f1"]["threshold"],
            e["best_f1"]["precision"],
            e["best_f1"]["recall"],
            e["best_f1"]["lead_time_mean_s"],
        )
    plot_pr_curves(
        scenes,
        {**predictors, "_baselines": None},
        _abs(figure),
        d_safe=d_safe,
        k=k,
    )


@app.command()
def simulate(
    profile: str = "small",
    out_dir: str | None = None,
    hours: float | None = None,
    tags: int | None = None,
    seed: int = 0,
) -> None:
    """합성 공장 RTLS 스트림(10 Hz, 파티션 Parquet) 생성. 기본 출력: data/rtls/<profile>/raw"""
    from foresight.data.rtls_sim import simulate as _simulate

    res = _simulate(
        _abs(out_dir or f"data/rtls/{profile}/raw"),
        hours=hours,
        tags=tags,
        seed=seed,
        profile=profile,
    )
    log.info(
        "simulate: %d rows in %.1fs (%.0f rows/s), %d near-miss events -> %s",
        res.rows,
        res.seconds,
        res.rows_per_s,
        res.n_events,
        res.manifest_path,
    )


@app.command("prepare-rtls")
def prepare_rtls(
    in_dir: str = "data/rtls/small/raw",
    out_dir: str = "data/processed/rtls",
    train_skip: int = 4,
    quality_min: int = 30,
    smoothing: str = "none",
) -> None:
    """RTLS Parquet → 품질 필터 → 2.5 Hz 리샘플 → 구역별 장면 npz (시간 기준 train/val/test) + DuckDB 통계."""
    from foresight.data.rtls_pipeline import PipelineConfig, run_pipeline

    cfg = PipelineConfig(skip_train=train_skip, quality_min=quality_min, smoothing=smoothing)  # type: ignore[arg-type]
    res = run_pipeline(
        _abs(in_dir), _abs(out_dir), cfg=cfg, stats_path=_abs(out_dir) / "stats.json"
    )
    log.info(
        "prepare-rtls: raw %d -> clean %d -> frames %d rows; scenes %s; peak RSS %.0f MB",
        res.rows_raw,
        res.rows_clean,
        res.rows_frames,
        res.scenes,
        res.peak_rss_mb,
    )


@app.command()
def export(
    ckpt: str = "results/checkpoints/eth/seed0/best.pth",
    out: str = "artifacts/onnx",
    int8: bool = True,
    calib: str = "data/processed/ethucy/eth/train.npz",
) -> None:
    """PyTorch 체크포인트 → ONNX (+ 정적 INT8 양자화)."""
    from foresight.inference.export import export_all

    export_all(_abs(ckpt), _abs(out), int8=int8, calib_npz=_abs(calib))


@app.command()
def benchmark(
    out: str = "results/benchmark.json",
    quick: bool = False,
    ckpt: str = "results/checkpoints/eth/seed0/best.pth",
    onnx_dir: str = "artifacts/onnx",
) -> None:
    """추론 벤치마크: eager / torch.compile / ONNX Runtime fp32 / INT8, 장면 크기별 지연·처리량."""
    from foresight.inference.benchmark import run_benchmark

    run_benchmark(_abs(out), quick=quick, ckpt=_abs(ckpt), onnx_dir=_abs(onnx_dir))


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, backend: str = "onnx", workers: int = 1) -> None:
    """FastAPI 추론 서버. workers>1 이면 프로세스 확장 (스레드가 아니라 — docs/inference_optimization.md §3)."""
    import os

    import uvicorn

    os.environ["FORESIGHT_BACKEND"] = backend
    if workers > 1:
        # uvicorn 은 다중 워커일 때 import 문자열(팩토리)만 받는다 — 각 워커가 자기 ONNX 세션을 만든다.
        uvicorn.run(
            "foresight.serving.app:app_from_env",
            host=host,
            port=port,
            workers=workers,
            factory=True,
        )
    else:
        from foresight.serving.app import create_app

        uvicorn.run(create_app(backend=backend), host=host, port=port)


@app.command()
def stream(
    mode: str = "consume",
    source: str = "replay",
    sink: str = "stdout",
    backend: str = "onnx",
    bootstrap: str = "localhost:9092",
    topic: str = "rtls.positions",
    alerts_topic: str = "rtls.alerts",
    replay_file: str | None = None,
    speed: float = 10.0,
    max_seconds: float | None = None,
) -> None:
    """스트리밍. mode=consume: replay(파일)/kafka 입력 → 예측·위험 점수 → stdout/kafka 경보. mode=produce: 재생 파일을 Kafka 위치 토픽으로 발행."""
    from foresight.serving.stream import produce_positions, run_stream

    if mode == "produce":
        if replay_file is None:
            raise typer.BadParameter("--mode produce 에는 --replay-file 이 필요하다")
        produce_positions(
            _abs(replay_file),
            bootstrap=bootstrap,
            topic=topic,
            speed=speed,
            max_seconds=max_seconds,
        )
        return

    run_stream(
        source=source,
        sink=sink,
        backend=backend,
        bootstrap=bootstrap,
        topic=topic,
        alerts_topic=alerts_topic,
        replay_file=None if replay_file is None else _abs(replay_file),
        speed=speed,
        max_seconds=max_seconds,
    )


@app.command("demo")
def demo(
    ckpt: str = "results/checkpoints/rtls-scratch-fast/best.pth",
    data_dir: str = "data/rtls/full/processed",
    out_dir: str = "demo",
    gif: str = "results/figures/demo.gif",
    seconds: float = 60.0,
    zone: int | None = None,
    d_safe: float = 1.0,
    k: int = 20,
    seed: int = 0,
    gif_frames: int = 100,
    no_gif: bool = False,
    threads: int = 1,
) -> None:
    """브라우저 데모 데이터 생성: 양성이 가장 많은 구역·창을 골라 demo/replay.js·demo/model.js 를 쓰고 README 용 GIF 를 그린다."""
    import torch

    from foresight.data.ethucy import SceneSet
    from foresight.demo.replay import (
        build_replay,
        export_weights,
        load_window_frames,
        select_window,
        write_js,
        zone_layout,
    )
    from foresight.eval.evaluate import load_model

    torch.set_num_threads(threads)
    scenes = SceneSet.load(_abs(data_dir) / "test.npz")
    window_bins = round(seconds / 0.4)
    win = select_window(scenes, d_safe=d_safe, window_bins=window_bins, zone=zone)
    log.info(
        "window: zone %d bins [%d, %d) positives %d",
        win.zone,
        win.bin_lo,
        win.bin_hi,
        win.n_positive,
    )
    frames = load_window_frames(_abs(data_dir) / "frames_2p5hz", win.zone, win.bin_lo, win.bin_hi)
    model = load_model(_abs(ckpt))
    replay = build_replay(
        frames,
        model,
        win,
        obs_len=scenes.obs_len,
        pred_len=scenes.pred_len,
        d_safe=d_safe,
        k=k,
        seed=seed,
        layout=zone_layout(win.zone),
        source=f"{Path(data_dir).as_posix()} (test split)",
    )
    out = _abs(out_dir)
    n_r = write_js(out / "replay.js", "FORESIGHT_REPLAY", replay)
    n_m = write_js(
        out / "model.js", "FORESIGHT_MODEL", export_weights(model, source=Path(ckpt).as_posix())
    )
    log.info(
        "wrote %s (%.1f KB, %d frames, %d agents) and %s (%.1f KB)",
        out / "replay.js",
        n_r / 1024,
        replay["meta"]["n_frames"],
        len(replay["agents"]),
        out / "model.js",
        n_m / 1024,
    )
    if not no_gif:
        from foresight.demo.gif import render_gif

        render_gif(replay, model, _abs(gif), k=k, seed=seed, d_safe=d_safe, max_frames=gif_frames)
    typer.echo(
        f"demo ready: {out / 'index.html'} (zone {win.zone}, {replay['meta']['n_frames']} frames)"
    )


if __name__ == "__main__":
    app()
