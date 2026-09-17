"""그래프 구성: 공식 코드(networkx) 와의 동일성, float32 커널 엣지 케이스, 골든 값."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from foresight.data.graph import (
    inverse_distance_kernel,
    normalized_laplacian,
    relative_displacement,
    scene_to_graph,
)


def _reference_laplacian(a: np.ndarray) -> np.ndarray:
    """공식 코드: nx.from_numpy_matrix → normalized_laplacian_matrix (networkx 3 에서는 from_numpy_array)."""
    out = np.zeros_like(a)
    for t in range(a.shape[0]):
        g = nx.from_numpy_array(a[t])
        out[t] = nx.normalized_laplacian_matrix(g).toarray()
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_normalized_laplacian_matches_networkx(seed: int) -> None:
    rng = np.random.default_rng(seed)
    feat = rng.normal(size=(8, 7, 2))
    a = inverse_distance_kernel(feat)
    np.testing.assert_allclose(normalized_laplacian(a), _reference_laplacian(a), atol=1e-9)


def test_kernel_zero_velocity_gives_zero_laplacian() -> None:
    """t=0 에서 상대 변위가 전부 0 이면 A=I, L=0 — 공식 코드와 같은 동작."""
    v, a, _, _ = scene_to_graph(np.cumsum(np.ones((4, 20, 2)), axis=1), obs_len=8)
    assert np.all(v[0] == 0)
    assert np.allclose(a[0], 0)


def test_identical_velocities_in_float32_have_zero_weight() -> None:
    """float64 로는 1e-17 차이, float32 로는 0 인 변위 쌍: 공식 코드는 float32 → 커널 0 이어야 한다."""
    pos = np.zeros((2, 20, 2))
    pos[0, :, 0] = 8.46 + 0.11 * np.arange(20)
    pos[1, :, 0] = 3.59 + 0.11 * np.arange(20)
    pos[1, :, 1] = 1.0
    a = inverse_distance_kernel(relative_displacement(pos).transpose(1, 0, 2))
    assert a[5, 0, 1] == 0.0


def test_scene_to_graph_matches_golden(golden: dict[str, np.ndarray]) -> None:
    for i in golden["scene_ids"]:
        vo, ao, vp, ap = scene_to_graph(golden[f"pos_{i}"], obs_len=8)
        np.testing.assert_array_equal(vo, golden[f"V_obs_{i}"])
        np.testing.assert_array_equal(ao, golden[f"A_obs_{i}"])
        np.testing.assert_array_equal(vp, golden[f"V_pred_{i}"])
        np.testing.assert_array_equal(ap, golden[f"A_pred_{i}"])
