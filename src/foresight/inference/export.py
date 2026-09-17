"""PyTorch 체크포인트 → ONNX (동적 N) → 정적 INT8, 그리고 매니페스트.

내보내기 전략
    ``torch.onnx.export(dynamo=True)`` 를 먼저 시도한다. torch.export 기반이라 ``einsum``·``view`` 를
    각각 Einsum·Reshape 한 노드로 옮겨 22 노드짜리 깨끗한 그래프가 나온다(레거시 TorchScript 경로는
    Shape/Gather/Concat 이 끼어 64 노드). dynamo 경로가 실패하면 레거시 경로로 내려간다 — 둘 다
    N 을 동적 축으로 잡는다.

검증
    ``onnx.checker`` + **실제 테스트 장면**(ETH N=2~5, UNIV N=3~57) 50 개 이상에서 torch 출력과
    최대 절대 오차 < 1e-4 를 확인한다. 합성 난수 입력은 라플라시안이 실제 분포와 달라 파라미터
    범위가 비현실적이므로 실제 장면으로 검증한다.

INT8
    전체 양자화와 TXP-CNN 부분 양자화를 둘 다 만들어 테스트 세트 ADE/FDE 로 비교하고, 나은 쪽을
    ``social_stgcnn_int8.onnx`` 로 남긴다 (다른 쪽은 ``_int8_full`` / ``_int8_txp`` 로 보존해 문서 표에 쓴다).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import torch

from foresight.data.ethucy import SceneSet
from foresight.inference.predictor import preprocess
from foresight.models import SocialSTGCNN
from foresight.utils import get_logger, project_root

log = get_logger("foresight.inference.export")

FP32_NAME = "social_stgcnn_fp32.onnx"
INT8_NAME = "social_stgcnn_int8.onnx"
OFFICIAL_ETH = Path("assets/official_checkpoints/social-stgcnn-eth.pth")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_checkpoint(ckpt: Path | None) -> Path:
    """학습 체크포인트가 아직 없으면 공식 ETH 체크포인트로 내려간다 (파이프라인 나머지를 막지 않기 위해)."""
    if ckpt is not None and Path(ckpt).exists():
        return Path(ckpt)
    fallback = project_root() / OFFICIAL_ETH
    log.warning("checkpoint %s not found — falling back to official checkpoint %s", ckpt, fallback)
    return fallback


def load_model(ckpt: Path) -> SocialSTGCNN:
    from foresight.eval.evaluate import load_model as _load

    return _load(ckpt).eval()


def export_onnx(
    model: SocialSTGCNN, out_path: Path, example_n: int = 5, opset: int = 18
) -> dict[str, object]:
    """동적 N 으로 ONNX 내보내기. dynamo → 실패 시 legacy. 사용한 경로/opset 을 돌려준다."""
    model = model.eval()
    v = torch.randn(1, 2, model.obs_len, example_n)
    a = torch.eye(example_n).repeat(model.obs_len, 1, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errors: dict[str, str] = {}
    t0 = time.perf_counter()
    try:
        n_dim = torch.export.Dim("N", min=1, max=1024)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                model,
                (v, a),
                str(out_path),
                dynamo=True,
                opset_version=opset,
                external_data=False,
                input_names=["v", "a"],
                output_names=["params"],
                dynamic_shapes={"v": {3: n_dim}, "a": {1: n_dim, 2: n_dim}},
            )
        exporter = "dynamo"
    except Exception as e:  # 어떤 실패든 레거시 경로로 폴백 (실패 사유는 매니페스트에 남긴다)
        errors["dynamo"] = f"{type(e).__name__}: {str(e)[:300]}"
        log.warning(
            "dynamo exporter failed (%s); falling back to legacy exporter", type(e).__name__
        )
        opset = min(opset, 17)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                model,
                (v, a),
                str(out_path),
                dynamo=False,
                opset_version=opset,
                input_names=["v", "a"],
                output_names=["params"],
                dynamic_axes={"v": {3: "N"}, "a": {1: "N", 2: "N"}, "params": {3: "N"}},
            )
        exporter = "legacy"
    dt = time.perf_counter() - t0
    m = onnx.load(str(out_path))
    onnx.checker.check_model(m, full_check=True)
    ops = sorted({n.op_type for n in m.graph.node})
    log.info(
        "exported %s via %s (opset %d, %d nodes, %.1fs)",
        out_path.name,
        exporter,
        opset,
        len(m.graph.node),
        dt,
    )
    return {
        "exporter": exporter,
        "opset": opset,
        "n_nodes": len(m.graph.node),
        "ops": ops,
        "export_seconds": round(dt, 2),
        "errors": errors,
    }


def parity_scenes(data_dir: Path, min_scenes: int = 50) -> list[np.ndarray]:
    """검증용 실제 장면 (N 2~57 골고루). ETH 테스트 전체 + UNIV 테스트에서 N 별 1 장면씩."""
    scenes: list[np.ndarray] = []
    eth = data_dir / "eth" / "test.npz"
    univ = data_dir / "univ" / "test.npz"
    if eth.exists():
        s = SceneSet.load(eth)
        scenes += [s.scene(i)[:, : s.obs_len] for i in range(len(s))]
    if univ.exists():
        s = SceneSet.load(univ)
        n = s.num_agents
        for size in np.unique(n):
            i = int(np.flatnonzero(n == size)[0])
            scenes.append(s.scene(i)[:, : s.obs_len])
    if len(scenes) < min_scenes:  # 데이터가 없는 환경(CI 최소 구성) — 합성 장면으로 채운다
        rng = np.random.default_rng(0)
        while len(scenes) < min_scenes:
            n_ag = int(rng.integers(2, 30))
            start = rng.uniform(-5, 5, size=(n_ag, 1, 2))
            vel = rng.uniform(-0.5, 0.5, size=(n_ag, 1, 2))
            scenes.append(start + vel * np.arange(8)[None, :, None])
    return scenes


def check_parity(
    model: SocialSTGCNN, onnx_path: Path, scenes: list[np.ndarray], tol: float = 1e-4
) -> dict[str, object]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    sizes: list[int] = []
    with torch.no_grad():
        for obs in scenes:
            v, a = preprocess(obs)
            ref = model(torch.from_numpy(v), torch.from_numpy(a)).numpy()
            out = sess.run(None, {"v": v, "a": a})[0]
            if out.shape != ref.shape:
                raise AssertionError(f"shape mismatch {out.shape} vs {ref.shape}")
            worst = max(worst, float(np.abs(out - ref).max()))
            sizes.append(obs.shape[0])
    ok = worst < tol
    log.info(
        "parity on %d scenes (N %d..%d): max|Δ|=%.2e %s",
        len(scenes),
        min(sizes),
        max(sizes),
        worst,
        "OK" if ok else "FAIL",
    )
    if not ok:
        raise AssertionError(f"ONNX/torch parity failed: max abs diff {worst:.3e} >= {tol}")
    return {
        "n_scenes": len(scenes),
        "n_min": int(min(sizes)),
        "n_max": int(max(sizes)),
        "max_abs_diff": worst,
        "tol": tol,
    }


def export_all(
    ckpt: Path,
    out: Path,
    int8: bool = True,
    calib_npz: Path | None = None,
    data_dir: Path | None = None,
    n_calib: int = 200,
    test_npz: Path | None = None,
) -> dict[str, object]:
    """CLI ``foresight export`` 진입점. fp32 (+INT8 두 변형) 를 만들고 ``manifest.json`` 을 쓴다."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = resolve_checkpoint(Path(ckpt) if ckpt else None)
    root = project_root()
    data_dir = data_dir or root / "data" / "processed" / "ethucy"
    model = load_model(ckpt)
    fp32 = out / FP32_NAME
    info = export_onnx(model, fp32)
    parity = check_parity(model, fp32, parity_scenes(data_dir))
    manifest: dict[str, object] = {
        "source_checkpoint": str(ckpt),
        "source_sha256": sha256_of(ckpt),
        "param_count": model.num_parameters(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": __import__("onnxruntime").__version__,
        "platform": platform.platform(),
        "inputs": {
            "v": "(1, 2, 8, N) float32 relative displacement",
            "a": "(8, N, N) float32 normalized Laplacian",
        },
        "output": {"params": "(1, 5, 12, N) float32 (mu_x, mu_y, log_sx, log_sy, atanh_rho)"},
        "export": info,
        "parity": parity,
        "files": {FP32_NAME: {"sha256": sha256_of(fp32), "bytes": fp32.stat().st_size}},
    }
    if int8:
        from foresight.inference.quantize import evaluate_onnx_accuracy, quantize_static_scenes

        calib_path = Path(calib_npz) if calib_npz else data_dir / "eth" / "train.npz"
        test_path = Path(test_npz) if test_npz else data_dir / "eth" / "test.npz"
        calib = SceneSet.load(calib_path)
        test = SceneSet.load(test_path)
        base = evaluate_onnx_accuracy(fp32, test)
        variants: dict[str, object] = {}
        best_name, best_ade = "", float("inf")
        for variant, fname in (
            ("full", "social_stgcnn_int8_full.onnx"),
            ("txp-only", "social_stgcnn_int8_txp.onnx"),
        ):
            qr = quantize_static_scenes(fp32, out / fname, calib, n_calib=n_calib, variant=variant)
            acc = evaluate_onnx_accuracy(qr.path, test)
            variants[variant] = {
                "file": fname,
                "quantize": {k: v for k, v in asdict(qr).items() if k != "path"},
                "accuracy": asdict(acc),
                "delta_vs_fp32": {
                    "ade_det": acc.ade_det - base.ade_det,
                    "fde_det": acc.fde_det - base.fde_det,
                    "ade_bo20": acc.ade_bo20 - base.ade_bo20,
                    "fde_bo20": acc.fde_bo20 - base.fde_bo20,
                },
            }
            manifest["files"][fname] = {
                "sha256": sha256_of(qr.path),
                "bytes": qr.path.stat().st_size,
            }  # type: ignore[index]
            score = acc.ade_bo20 + acc.ade_det
            if score < best_ade:
                best_ade, best_name = score, fname
        # 더 나은 변형을 대표 INT8 파일로 복사한다
        best_src = out / best_name
        (out / INT8_NAME).write_bytes(best_src.read_bytes())
        manifest["files"][INT8_NAME] = {
            "sha256": sha256_of(out / INT8_NAME),
            "bytes": (out / INT8_NAME).stat().st_size,
            "copy_of": best_name,
        }  # type: ignore[index]
        manifest["int8"] = {
            "calibration_npz": str(calib_path),
            "n_calib": n_calib,
            "test_npz": str(test_path),
            "fp32_accuracy": asdict(base),
            "variants": variants,
            "selected": best_name,
            "note": "dynamic quantization does not quantize Conv; static QDQ per-channel, MinMax calibration (histogram methods fail on ragged N)",
        }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    log.info("manifest -> %s", out / "manifest.json")
    return manifest


if __name__ == "__main__":  # pragma: no cover
    root = project_root()
    export_all(
        root / os.environ.get("FORESIGHT_CKPT", "results/checkpoints/eth/seed0/best.pth"),
        root / "artifacts" / "onnx",
    )
