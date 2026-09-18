"""모델: 파라미터 수 7,563, 공식 체크포인트 로드, 골든 출력 일치(기계 간 1e-6), 배치 경로 동일성, permute 모드 차이."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from foresight.data.graph import scene_to_graph
from foresight.models import SocialSTGCNN, bivariate_nll, load_official_checkpoint, official_key_map


def test_parameter_count_matches_paper() -> None:
    assert SocialSTGCNN().num_parameters() == 7563  # 논문: 7.6K


def test_official_checkpoint_loads_strictly(official_eth_ckpt: Path) -> None:
    sd = torch.load(official_eth_ckpt, map_location="cpu")
    assert "tpcnn_ouput.weight" in sd  # 공식 코드의 오타 키
    model = load_official_checkpoint(SocialSTGCNN(), str(official_eth_ckpt))
    assert set(model.state_dict()) == set(official_key_map(sd))


def test_output_matches_reference(golden: dict[str, np.ndarray], official_eth_ckpt: Path) -> None:
    """공식 체크포인트 + 골든 입력 → 골든 출력.

    골든 값을 만든 기계에서는 비트 단위로 같지만, CPU 가 다르면 float32 conv/einsum 커널(SIMD 경로)이 달라
    ~1e-6 차이가 난다(GitHub 러너에서 최대 9.5e-7 관측). 그래서 기계 간 비교는 1e-5 상대·2e-6 절대 허용오차로 둔다 —
    구현 차이(예: permute 축 교환)는 1e-1 수준이라 이 허용오차로도 충분히 잡힌다.
    """
    model = load_official_checkpoint(SocialSTGCNN(), str(official_eth_ckpt)).eval()
    for i in golden["scene_ids"]:
        vo, ao, _, _ = scene_to_graph(golden[f"pos_{i}"], obs_len=8)
        out = model.predict_params(
            torch.from_numpy(vo).permute(2, 0, 1).unsqueeze(0), torch.from_numpy(ao)
        )[0].numpy()
        np.testing.assert_allclose(out, golden[f"out_{i}"], rtol=1e-5, atol=2e-6)


def test_batched_adjacency_path_equals_shared(
    golden: dict[str, np.ndarray], official_eth_ckpt: Path
) -> None:
    model = load_official_checkpoint(SocialSTGCNN(), str(official_eth_ckpt)).eval()
    vo, ao, _, _ = scene_to_graph(golden["pos_0"], obs_len=8)
    v = torch.from_numpy(vo).permute(2, 0, 1).unsqueeze(0)
    a = torch.from_numpy(ao)
    out1 = model.predict_params(v, a)
    out2 = model.predict_params(torch.cat([v, v]), torch.stack([a, a]))
    torch.testing.assert_close(out2[0], out1[0])
    torch.testing.assert_close(out2[1], out1[0])


def test_permute_mode_differs_from_view(
    golden: dict[str, np.ndarray], official_eth_ckpt: Path
) -> None:
    """공식 구현의 view 와 논문 그림대로의 permute 는 다른 함수다 — 재현 문서의 근거."""
    vo, ao, _, _ = scene_to_graph(golden["pos_0"], obs_len=8)
    v = torch.from_numpy(vo).permute(2, 0, 1).unsqueeze(0)
    a = torch.from_numpy(ao)
    m_view = load_official_checkpoint(
        SocialSTGCNN(time_channel_swap="view"), str(official_eth_ckpt)
    ).eval()
    m_perm = load_official_checkpoint(
        SocialSTGCNN(time_channel_swap="permute"), str(official_eth_ckpt)
    ).eval()
    assert not torch.allclose(m_view.predict_params(v, a), m_perm.predict_params(v, a))


def test_bivariate_nll_exact_is_capped_and_stable_matches_when_moderate() -> None:
    torch.manual_seed(0)
    pred = torch.randn(12, 3, 5) * 0.3
    target = torch.randn(12, 3, 2) * 0.3
    exact = bivariate_nll(pred, target, exact=True)
    stable = bivariate_nll(pred, target, exact=False)
    assert torch.isfinite(exact) and torch.isfinite(stable)
    torch.testing.assert_close(exact, stable, atol=1e-4, rtol=1e-4)
    # 극단값: exact 경로는 pdf 클램프(1e-20) 때문에 항당 -log(1e-20)=46.05 를 넘지 못한다
    far = target + 100.0
    assert bivariate_nll(pred, far, exact=True) <= 46.06
    assert bivariate_nll(pred, far, exact=False) > 100.0


def test_training_step_reduces_loss() -> None:
    torch.manual_seed(0)
    model = SocialSTGCNN()
    v = torch.randn(1, 2, 8, 6) * 0.2
    a = torch.rand(8, 6, 6)
    target = torch.randn(12, 6, 2) * 0.2
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = bivariate_nll(model(v, a).permute(0, 2, 3, 1)[0], target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
