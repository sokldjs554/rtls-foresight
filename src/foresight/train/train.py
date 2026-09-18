"""학습 루프 — Social-STGCNN 공식 학습 절차의 재현 + MLflow 추적.

공식 절차 (train.py):
    SGD(lr 0.01) · 장면 1개씩 forward · 장면 128개의 손실을 더해 128 로 나눈 뒤 backward/step ·
    StepLR(150, γ=0.2) · 매 epoch 검증 손실 · 검증 손실 최소 epoch 의 가중치 저장.

여기에 추가한 것: 시드 고정, MLflow 로깅(epoch 손실·lr·시간·최종 ADE/FDE), 체크포인트에 설정 동봉,
버킷 배치 모드(ablation), 미세조정(init_from), 스레드 수 제한(분할 병렬 실행).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from mlflow import pytorch as mlflow_pytorch
from omegaconf import DictConfig, OmegaConf

from foresight.data.dataset import SceneBatch, SceneGraphDataset
from foresight.data.ethucy import SceneSet
from foresight.eval.evaluate import EvalResult, evaluate_model, load_model
from foresight.models import SocialSTGCNN, bivariate_nll
from foresight.utils import Timer, get_logger, project_root, seed_everything

log = get_logger(__name__)
# MLflow 3 의 사용 통계 전송·안내 문구를 끈다 (폐쇄망/프록시 환경에서 연결 실패 로그만 남긴다).
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def _load_split(path: str, limit: int | None, kernel: str, normalize: bool) -> SceneGraphDataset:
    scenes = SceneSet.load(_abs(path))
    if limit is not None and limit < len(scenes):
        idx = scenes.scene_index[:limit]
        end = int(idx[-1, 1])
        scenes = SceneSet(
            pos=scenes.pos[:end],
            scene_index=idx,
            meta=scenes.meta[:limit],
            obs_len=scenes.obs_len,
            pred_len=scenes.pred_len,
            agent_type=None if scenes.agent_type is None else scenes.agent_type[:end],
        )
    return SceneGraphDataset(scenes, kernel=kernel, normalize=normalize)  # type: ignore[arg-type]


def build_model(cfg: DictConfig) -> SocialSTGCNN:
    m = cfg.model
    return SocialSTGCNN(
        n_stgcnn=m.n_stgcnn,
        n_txpcnn=m.n_txpcnn,
        in_channels=m.in_channels,
        out_channels=m.out_channels,
        obs_len=m.obs_len,
        pred_len=m.pred_len,
        t_kernel=m.t_kernel,
        time_channel_swap=m.time_channel_swap,
    )


def _batch_loss(model: SocialSTGCNN, b: SceneBatch, exact: bool) -> torch.Tensor:
    out = model(b.v_obs, b.a_obs).permute(0, 2, 3, 1)  # (B, T_pred, N, 5)
    return bivariate_nll(out, b.v_pred, exact=exact)


def run_epoch(
    model: SocialSTGCNN,
    ds: SceneGraphDataset,
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer | None,
    rng: np.random.Generator,
    epoch: int,
) -> float:
    """한 epoch 의 평균 손실. optimizer=None 이면 검증(그래디언트 없음)."""
    train = optimizer is not None
    model.train(train)
    tcfg = cfg.train
    if tcfg.mode == "scene":
        order = rng.permutation(len(ds)) if train else np.arange(len(ds))
        iterator = ds.iter_scenes(order)
        accum = int(tcfg.batch_size)
    else:
        iterator = ds.iter_buckets(int(tcfg.bucket_batch), rng if train else None)
        accum = max(1, int(tcfg.batch_size) // int(tcfg.bucket_batch))
    total, n_batches, acc_loss, acc_n, step = 0.0, 0, None, 0, 0
    with torch.set_grad_enabled(train):
        for b in iterator:
            loss = _batch_loss(model, b, bool(tcfg.loss_exact))
            acc_loss = loss if acc_loss is None else acc_loss + loss
            acc_n += 1
            if acc_n == accum:
                mean_loss = acc_loss / acc_n
                if train:
                    optimizer.zero_grad()  # type: ignore[union-attr]
                    mean_loss.backward()
                    if tcfg.clip_grad:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.clip_grad))
                    optimizer.step()  # type: ignore[union-attr]
                # 학습: 옵티마이저 스텝 평균 (공식 코드와 같은 로그). 검증: 장면 가중 평균 (자투리 배치가 과대 반영되지 않게)
                total += float(mean_loss.item()) * (acc_n if not train else 1)
                n_batches += acc_n if not train else 1
                step += 1
                if train and tcfg.log_every and step % int(tcfg.log_every) == 0:
                    log.info("epoch %d step %d loss %.4f", epoch, step, total / n_batches)
                acc_loss, acc_n = None, 0
        if acc_n > 0 and acc_loss is not None:  # 마지막 자투리 배치
            mean_loss = acc_loss / acc_n
            if train:
                optimizer.zero_grad()  # type: ignore[union-attr]
                mean_loss.backward()
                if tcfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.clip_grad))
                optimizer.step()  # type: ignore[union-attr]
            total += float(mean_loss.item()) * (acc_n if not train else 1)
            n_batches += acc_n if not train else 1
    return total / max(n_batches, 1)


def run_training(cfg: DictConfig) -> dict[str, Any]:
    torch.set_num_threads(int(cfg.threads))
    seed_everything(int(cfg.seed))
    rng = np.random.default_rng(int(cfg.seed))
    run_name = cfg.run_name or f"{cfg.dataset.name}-{cfg.train.get('name', 'train')}-seed{cfg.seed}"
    out_dir = (
        _abs(cfg.out_dir) / cfg.dataset.name / f"seed{cfg.seed}"
        if cfg.run_name is None
        else _abs(cfg.out_dir) / cfg.run_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    kernel, normalize = cfg.model.graph.kernel, bool(cfg.model.graph.normalize)
    with Timer() as t_data:
        ds_train = _load_split(cfg.dataset.train, cfg.dataset.limit_scenes, kernel, normalize)
        ds_val = _load_split(cfg.dataset.val, cfg.dataset.limit_scenes, kernel, normalize)
        ds_test = _load_split(cfg.dataset.test, cfg.dataset.limit_scenes, kernel, normalize)
    log.info(
        "data: train %d / val %d / test %d scenes (%.1fs)",
        len(ds_train),
        len(ds_val),
        len(ds_test),
        t_data.elapsed,
    )

    model = build_model(cfg)
    if cfg.init_from:
        model_kw = {
            k: v for k, v in OmegaConf.to_container(cfg.model, resolve=True).items() if k != "graph"
        }  # type: ignore[union-attr]
        init = load_model(_abs(cfg.init_from), **model_kw)
        model.load_state_dict(init.state_dict())
        log.info("init from %s", cfg.init_from)
    log.info("model params: %d", model.num_parameters())

    if cfg.train.optimizer == "sgd":
        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            model.parameters(), lr=float(cfg.train.lr)
        )
    elif cfg.train.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))
    else:
        raise ValueError(cfg.train.optimizer)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(cfg.train.lr_step), gamma=float(cfg.train.lr_gamma)
    )

    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    mlflow.set_experiment(str(cfg.mlflow.experiment))
    flat = {
        k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
        for k, v in _flatten(OmegaConf.to_container(cfg, resolve=True)).items()
    }  # type: ignore[arg-type]
    history: list[dict[str, float]] = []
    best_val, best_epoch = float("inf"), -1
    best_path = out_dir / "best.pth"
    last_path = out_dir / "last.pth"
    start_epoch, resume_run_id = 0, None
    # 설정 지문: 같은 out_dir 에 다른 설정(paper vs fast, 다른 데이터)의 last.pth 가 남아 있으면 이어 붙지 않는다.
    cfg_fingerprint = hashlib.sha256(
        json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    if bool(cfg.get("resume", True)) and last_path.exists():
        # 재시작 안전성: 컨테이너가 죽어도 마지막 epoch 부터 같은 MLflow run 에 이어 붙는다
        # (모델·옵티마이저·스케줄러·RNG·기록을 복원하므로 처음부터 돌린 것과 같은 궤적을 따른다).
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        if ck.get("config_fingerprint") != cfg_fingerprint:
            log.warning(
                "last.pth was written by a different config (%s != %s); starting fresh",
                ck.get("config_fingerprint"),
                cfg_fingerprint,
            )
            ck = None
        elif ck.get("mlflow_run_id"):
            try:  # 새 클론·다른 tracking 스토어에는 run 이 없다 → 새 run 으로 이어 붙인다
                mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
                mlflow.get_run(ck["mlflow_run_id"])
            except Exception:
                log.warning(
                    "MLflow run %s not found in %s; continuing in a new run",
                    ck["mlflow_run_id"],
                    cfg.mlflow.tracking_uri,
                )
                ck["mlflow_run_id"] = None
    else:
        ck = None
    if ck is not None:
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        rng.bit_generator.state = ck["numpy_rng"]
        torch.set_rng_state(ck["torch_rng"])
        history, best_val, best_epoch = ck["history"], ck["best_val"], ck["best_epoch"]
        start_epoch, resume_run_id = int(ck["epoch"]) + 1, ck.get("mlflow_run_id")
        if start_epoch > int(cfg.train.epochs):
            raise RuntimeError(
                f"{last_path} already holds {start_epoch} epochs >= train.epochs={cfg.train.epochs}; delete it or raise epochs"
            )
        log.info(
            "resume from %s: epoch %d, best val %.4f @%d",
            last_path,
            start_epoch,
            best_val,
            best_epoch,
        )
    with mlflow.start_run(
        run_id=resume_run_id, run_name=None if resume_run_id else run_name
    ) as run:
        if not resume_run_id:
            mlflow.log_params({k: str(v)[:250] for k, v in flat.items()})
            mlflow.set_tags(
                {
                    "dataset": cfg.dataset.name,
                    "mode": cfg.train.mode,
                    "params": model.num_parameters(),
                }
            )
        for epoch in range(start_epoch, int(cfg.train.epochs)):
            with Timer() as t_ep:
                tr = run_epoch(model, ds_train, cfg, optimizer, rng, epoch)
                va = (
                    run_epoch(model, ds_val, cfg, None, rng, epoch)
                    if epoch % int(cfg.train.val_every) == 0
                    else float("nan")
                )
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            history.append(
                {"epoch": epoch, "train_loss": tr, "val_loss": va, "lr": lr, "sec": t_ep.elapsed}
            )
            mlflow.log_metrics(
                {"train_loss": tr, "val_loss": va, "lr": lr, "epoch_sec": t_ep.elapsed}, step=epoch
            )
            improved = va < best_val
            if improved:
                best_val, best_epoch = va, epoch
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "model": OmegaConf.to_container(cfg.model, resolve=True),
                        "epoch": epoch,
                        "val_loss": va,
                        "dataset": cfg.dataset.name,
                        "seed": int(cfg.seed),
                    },
                    best_path,
                )
            log.info(
                "epoch %3d train %.4f val %.4f lr %.4g (%.1fs)%s",
                epoch,
                tr,
                va,
                lr,
                t_ep.elapsed,
                " *" if improved else "",
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "history": history,
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "epoch": epoch,
                    "mlflow_run_id": run.info.run_id,
                    "config_fingerprint": cfg_fingerprint,
                },
                last_path.with_suffix(".tmp"),
            )
            os.replace(
                last_path.with_suffix(".tmp"), last_path
            )  # 원자적 교체: 저장 중 죽어도 이전 last.pth 가 남는다
        (out_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
        mlflow.log_artifact(str(out_dir / "history.json"))

        # 최종 평가: 검증 손실 최소 체크포인트 (공식 절차) 로 테스트 ADE/FDE
        best = load_model(best_path)
        res: EvalResult = evaluate_model(
            best,
            ds_test,
            k=int(cfg.eval.k),
            seeds=tuple(int(s) for s in cfg.eval.seeds),
            split=cfg.dataset.name,
        )
        metrics = {
            "test_ade_bo20": res.best_of_k_per_agent["ade"],
            "test_fde_bo20": res.best_of_k_per_agent["fde"],
            "test_ade_bo20_std": res.best_of_k_per_agent_std["ade"],
            "test_fde_bo20_std": res.best_of_k_per_agent_std["fde"],
            "test_ade_joint": res.best_of_k_joint["ade"],
            "test_fde_joint": res.best_of_k_joint["fde"],
            "test_ade_det": res.deterministic["ade"],
            "test_fde_det": res.deterministic["fde"],
            "test_ade_cvm": res.cvm["ade"],
            "test_fde_cvm": res.cvm["fde"],
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
        }
        mlflow.log_metrics(metrics)
        if bool(cfg.mlflow.register) and not str(cfg.mlflow.tracking_uri).startswith("file:"):
            # MLflow 3 레지스트리는 "로그된 모델"만 등록할 수 있다. 상태 사전 파일이 아니라 pytorch flavor 로 남긴다.
            info = mlflow_pytorch.log_model(
                best,
                name="model",
                pip_requirements=["torch", "numpy"],
                serialization_format="pickle",
                registered_model_name=f"social-stgcnn-{cfg.dataset.name}",
            )
            summary_model_uri = info.model_uri
        else:
            summary_model_uri = None
        summary = {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "dataset": cfg.dataset.name,
            "seed": int(cfg.seed),
            "model_uri": summary_model_uri,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "eval": asdict(res),
            "metrics": metrics,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        (out_dir / "metrics.json").write_text(
            json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        mlflow.log_artifact(str(out_dir / "metrics.json"))
        mlflow.log_artifact(str(best_path))
    log.info(
        "done: %s  ADE/FDE(bo20) %.3f/%.3f  det %.3f/%.3f  cvm %.3f/%.3f",
        cfg.dataset.name,
        metrics["test_ade_bo20"],
        metrics["test_fde_bo20"],
        metrics["test_ade_det"],
        metrics["test_fde_det"],
        metrics["test_ade_cvm"],
        metrics["test_fde_cvm"],
    )
    return summary


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def compose_config(overrides: list[str] | None = None) -> DictConfig:
    """Hydra compose API — CLI 와 테스트에서 @hydra.main 없이 설정을 만든다."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(project_root() / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=list(overrides or []))
    # train/<name>.yaml 의 이름을 run_name 에 쓰기 위해 보관
    for ov in overrides or []:
        if ov.startswith("train="):
            OmegaConf.set_struct(cfg, False)
            cfg.train.name = ov.split("=", 1)[1]
    if "name" not in cfg.train:
        OmegaConf.set_struct(cfg, False)
        cfg.train.name = "paper"
    return cfg


if __name__ == "__main__":  # python -m foresight.train.train dataset=eth train=paper -m seed=0,1,2
    import hydra

    @hydra.main(
        config_path=str(project_root() / "configs"), config_name="config", version_base=None
    )
    def _main(cfg: DictConfig) -> None:
        OmegaConf.set_struct(cfg, False)
        cfg.train.setdefault(
            "name", hydra.core.hydra_config.HydraConfig.get().runtime.choices.get("train", "paper")
        )
        run_training(cfg)

    _main()
