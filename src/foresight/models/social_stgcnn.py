"""Social-STGCNN (Mohamed et al., CVPR 2020) — 처음부터 구현.

구조 (논문 Fig. 2, 공식 코드 ``model.py`` 와 파라미터 수 7,563 이 같도록 맞춤):

    입력 V (B, 2, T_obs, N), A (T_obs, N, N)
    ┌ ST-GCNN ×n_stgcnn ─────────────────────────────────────────────┐
    │ GraphConv: 1x1 Conv(2→5) → einsum('nctv,tvw->nctw') 로 A_t 곱  │
    │ TCN: BN → PReLU → Conv(k=3, 시간축) → BN → Dropout, + 잔차(1x1)  │
    └────────────────────────────────────────────────────────────────┘
    ┌ TXP-CNN ×n_txpcnn ─ 시간축을 채널로 보고 Conv2d(T_obs→T_pred, 3x3) ┐
    │ 이후 층은 (T_pred→T_pred) + 잔차, 마지막 출력 Conv                    │
    └──────────────────────────────────────────────────────────────────┘
    출력 (B, 5, T_pred, N): 각 (t, 노드) 의 상대 변위 이변량 가우시안 (μx, μy, log σx, log σy, atanh ρ)

(C,T) 축 교환 — 공식 구현의 ``view`` 문제
    공식 코드는 ST-GCNN 출력 ``(B, C, T, N)`` 을 TXP-CNN 입력 ``(B, T, C, N)`` 으로 바꿀 때 ``permute`` 가
    아니라 ``view`` 를 쓴다. ``view`` 는 메모리를 재해석할 뿐이라 시간·채널 축이 뒤섞이지만, 공개된
    체크포인트와 논문 수치는 그 동작으로 만들어졌다. 그래서 ``time_channel_swap="view"`` (기본, 재현용) 와
    ``"permute"`` (논문 그림대로) 를 모두 지원하고, 둘의 차이는 ablation 으로 측정한다.

배치 처리
    공식 코드는 장면 1개 = 배치 1 로만 돈다(BatchNorm 통계도 장면 단위). 이 구현은 같은 N 을 갖는
    장면들을 ``(B, ·, ·, N)`` 으로 묶어 돌릴 수 있고, ``A`` 를 ``(B, T, N, N)`` 로도 받는다.
    학습 재현은 B=1 을 쓰고, 추론 벤치마크에서 배치 효과를 측정한다.
"""

from __future__ import annotations

import torch
from torch import nn


class GraphConv(nn.Module):
    """채널 1x1 conv 후 프레임별 인접행렬을 곱하는 공간 그래프 합성곱."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if a.dim() == 3:  # (T, N, N) — 배치 공유
            return torch.einsum("nctv,tvw->nctw", x, a).contiguous()
        return torch.einsum("nctv,ntvw->nctw", x, a).contiguous()  # (B, T, N, N)


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_kernel: int = 3,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        assert t_kernel % 2 == 1
        pad = ((t_kernel - 1) // 2, 0)
        self.gcn = GraphConv(in_channels, out_channels)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, (t_kernel, 1), (1, 1), pad),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        self.residual: nn.Module
        if not residual:
            self.residual = _Zero()
        elif in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels)
            )
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x, a)
        x = self.tcn(x) + res
        return self.prelu(x)


class _Zero(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x[:, :0])


class SocialSTGCNN(nn.Module):
    """논문 기본 설정: n_stgcnn=1, n_txpcnn=5, 입력 2, 출력 5, obs 8, pred 12, 시간 커널 3."""

    def __init__(
        self,
        n_stgcnn: int = 1,
        n_txpcnn: int = 5,
        in_channels: int = 2,
        out_channels: int = 5,
        obs_len: int = 8,
        pred_len: int = 12,
        t_kernel: int = 3,
        time_channel_swap: str = "view",
    ) -> None:
        super().__init__()
        if time_channel_swap not in ("view", "permute"):
            raise ValueError("time_channel_swap must be 'view' or 'permute'")
        self.obs_len, self.pred_len, self.out_channels = obs_len, pred_len, out_channels
        self.time_channel_swap = time_channel_swap
        blocks = [STGCNBlock(in_channels, out_channels, t_kernel)]
        blocks += [STGCNBlock(out_channels, out_channels, t_kernel) for _ in range(n_stgcnn - 1)]
        self.st_gcns = nn.ModuleList(blocks)
        tp = [nn.Conv2d(obs_len, pred_len, 3, padding=1)]
        tp += [nn.Conv2d(pred_len, pred_len, 3, padding=1) for _ in range(n_txpcnn - 1)]
        self.tpcnns = nn.ModuleList(tp)
        self.tpcnn_output = nn.Conv2d(pred_len, pred_len, 3, padding=1)
        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(n_txpcnn)])

    def forward(self, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """v: (B, C_in, T_obs, N), a: (T_obs, N, N) 또는 (B, T_obs, N, N) → (B, 5, T_pred, N)."""
        for blk in self.st_gcns:
            v = blk(v, a)
        # (B, C, T, N) → (B, T, C, N): 시간축을 채널로 보고 T_obs→T_pred 외삽
        v = self._swap(v)
        v = self.prelus[0](self.tpcnns[0](v))
        for k in range(1, len(self.tpcnns) - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v
        v = self.tpcnn_output(v)
        return self._swap(v)

    def _swap(self, v: torch.Tensor) -> torch.Tensor:
        b, c, t, n = v.shape
        if self.time_channel_swap == "view":
            return v.contiguous().view(b, t, c, n)
        return v.permute(0, 2, 1, 3).contiguous()

    @torch.no_grad()
    def predict_params(self, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """추론 편의: (B, 5, T_pred, N) → (B, T_pred, N, 5)."""
        return self.forward(v, a).permute(0, 2, 3, 1).contiguous()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def official_key_map(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """공식 체크포인트(``st_gcns.0.gcn.conv.*``, ``tpcnn_ouput.*`` 오타 포함)의 키를 이 구현의 키로 바꾼다.

    공식 코드의 ``residual`` 은 ``nn.Sequential(Conv2d, BatchNorm2d)`` 로 이름이 같고, ``tcn`` 순서도 같다.
    다른 것은 ``tpcnn_ouput`` → ``tpcnn_output`` 뿐이다.
    """
    out = {}
    for k, val in state_dict.items():
        k2 = k.replace("tpcnn_ouput.", "tpcnn_output.")
        out[k2] = val
    return out


def load_official_checkpoint(model: SocialSTGCNN, path: str) -> SocialSTGCNN:
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(official_key_map(sd), strict=True)
    return model
