from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("FORESIGHT_ROOT", str(ROOT))
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@pytest.fixture(scope="session")
def golden() -> dict[str, np.ndarray]:
    z = np.load(ROOT / "tests" / "fixtures" / "golden_eth_test.npz")
    return {k: z[k] for k in z.files}


@pytest.fixture(scope="session")
def official_eth_ckpt() -> Path:
    return ROOT / "assets" / "official_checkpoints" / "social-stgcnn-eth.pth"


@pytest.fixture(scope="session")
def eth_test_npz() -> Path:
    p = ROOT / "data" / "processed" / "ethucy" / "eth" / "test.npz"
    if not p.exists():
        pytest.skip(
            "processed ETH data not present (run `foresight download && foresight prepare`)"
        )
    return p
