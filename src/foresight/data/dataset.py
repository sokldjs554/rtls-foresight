"""장면 데이터셋: SceneSet → (V_obs, A_obs, V_pred, A_pred) 텐서.

그래프는 장면당 한 번만 만들어 메모리에 캐시한다 (ETH 학습 분할 2,785 장면, 수십 MB).
공식 코드는 배치 크기 1(장면 1개)만 지원한다. 여기서는 두 모드를 제공한다.

* ``scene`` 모드: ``batch_size=1`` 장면 단위 — 재현 학습에 사용.
* ``bucket`` 모드: 노드 수 N 이 같은 장면끼리 묶어 ``(B, ·, ·, N)`` 배치를 만든다. 패딩이 없으니 마스크가
  필요 없고 BatchNorm 통계만 배치 단위로 바뀐다. 추론 벤치마크와 "빠른 학습" ablation 용.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from foresight.data.ethucy import SceneSet
from foresight.data.graph import Kernel, scene_to_graph


@dataclass
class SceneBatch:
    v_obs: torch.Tensor  # (B, 2, T_obs, N)
    a_obs: torch.Tensor  # (B, T_obs, N, N)
    v_pred: torch.Tensor  # (B, T_pred, N, 2)  — 손실 계산용 (모델 출력 permute 와 같은 배치)
    a_pred: torch.Tensor  # (B, T_pred, N, N)
    pos: torch.Tensor  # (B, N, T, 2) 절대좌표 float32
    agent_type: torch.Tensor  # (B, N) int8
    indices: list[int]

    @property
    def last_obs(self) -> torch.Tensor:
        """(B, N, 2) 마지막 관측 위치 — 상대 변위 → 절대좌표 복원 기준점."""
        return self.pos[:, :, self.v_obs.shape[2] - 1]


class SceneGraphDataset(Dataset):
    def __init__(
        self, scenes: SceneSet, kernel: Kernel = "velocity", normalize: bool = True
    ) -> None:
        self.scenes = scenes
        self.obs_len = scenes.obs_len
        self.graphs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = [
            scene_to_graph(scenes.scene(i), self.obs_len, kernel, normalize)
            for i in range(len(scenes))
        ]

    @classmethod
    def from_npz(cls, path: Path, **kw: object) -> SceneGraphDataset:
        return cls(SceneSet.load(Path(path)), **kw)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, i: int) -> SceneBatch:
        vo, ao, vp, ap = self.graphs[i]
        pos = self.scenes.scene(i).astype(np.float32)
        return SceneBatch(
            v_obs=torch.from_numpy(vo).permute(2, 0, 1).unsqueeze(0),
            a_obs=torch.from_numpy(ao).unsqueeze(0),
            v_pred=torch.from_numpy(vp).unsqueeze(0),
            a_pred=torch.from_numpy(ap).unsqueeze(0),
            pos=torch.from_numpy(pos).unsqueeze(0),
            agent_type=torch.from_numpy(self.scenes.types(i)).unsqueeze(0),
            indices=[i],
        )

    def num_agents(self, i: int) -> int:
        return int(self.scenes.num_agents[i])

    def iter_scenes(self, order: np.ndarray | None = None) -> Iterator[SceneBatch]:
        idx = np.arange(len(self)) if order is None else order
        for i in idx:
            yield self[int(i)]

    def iter_buckets(
        self, batch_size: int, rng: np.random.Generator | None = None
    ) -> Iterator[SceneBatch]:
        """노드 수가 같은 장면을 최대 ``batch_size`` 개씩 묶어 낸다 (패딩 없음)."""
        buckets: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self)):
            buckets[self.num_agents(i)].append(i)
        batches: list[list[int]] = []
        for ids in buckets.values():
            ids_arr = np.array(ids)
            if rng is not None:
                rng.shuffle(ids_arr)
            batches += [
                ids_arr[j : j + batch_size].tolist() for j in range(0, len(ids_arr), batch_size)
            ]
        if rng is not None:
            rng.shuffle(batches)  # type: ignore[arg-type]
        for b in batches:
            items = [self[i] for i in b]
            yield SceneBatch(
                v_obs=torch.cat([it.v_obs for it in items]),
                a_obs=torch.cat([it.a_obs for it in items]),
                v_pred=torch.cat([it.v_pred for it in items]),
                a_pred=torch.cat([it.a_pred for it in items]),
                pos=torch.cat([it.pos for it in items]),
                agent_type=torch.cat([it.agent_type for it in items]),
                indices=b,
            )
