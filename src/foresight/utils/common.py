from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def project_root() -> Path:
    """저장소 루트. 환경변수 FORESIGHT_ROOT 가 있으면 우선한다 (Docker/CI 에서 사용)."""
    env = os.environ.get("FORESIGHT_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3]


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """python / numpy / torch 시드를 한 번에 고정한다.

    재현 실험의 약속: 같은 시드 + 같은 코드 = 같은 수치. CPU 학습이라 cudnn 설정은 무의미하지만
    torch.use_deterministic_algorithms 는 einsum/scatter 계열의 비결정성을 막아 준다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_logger(name: str = "foresight") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("FORESIGHT_LOGLEVEL", "INFO"))
        logger.propagate = False
    return logger


class Timer:
    """`with Timer() as t: ...; t.elapsed` 형태의 벽시계 타이머."""

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self.start
