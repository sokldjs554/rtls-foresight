"""ETH/UCY 궤적 파일 → 장면(scene) 윈도우.

공식 Social-STGCNN / Social-GAN 데이터로더의 규칙을 그대로 따른다(재현 실험의 전제).

* 파일 포맷: ``frame_id  ped_id  x  y`` (탭 구분, 미터, 2.5 fps).
* 길이 ``seq_len = obs_len + pred_len`` (8 + 12) 의 슬라이딩 윈도우, stride ``skip``.
* 윈도우의 20 프레임 전부에 연속으로 등장하는 보행자만 노드로 채택한다.
* 채택된 보행자가 ``min_ped`` 보다 **많은** 장면만 남긴다 (공식 코드: ``> min_ped``, 기본 1 → 2명 이상).
* 좌표는 소수 4자리로 반올림한다 (공식 코드 ``np.around(..., 4)``).

공식 구현은 파이썬 이중 루프 + networkx 로 장면당 수십 ms 가 걸려 ETH 학습 분할(2,785 장면)
캐시 생성에 3~4 분이 걸린다. 여기서는 Polars 로 프레임을 정렬·그룹화하고 numpy 로 윈도우를
잘라 같은 장면 집합을 수 초 만에 만든다. 동일성은 ``tests/test_ethucy_reference.py`` 에서 검증한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

RAW_SCHEMA = {"frame": pl.Float64, "ped": pl.Float64, "x": pl.Float64, "y": pl.Float64}


@dataclass
class SceneSet:
    """한 분할(subset)의 모든 장면.

    Attributes:
        pos: ``(total_agents, seq_len, 2)`` 절대 좌표(미터). 장면별 에이전트가 이어 붙어 있다.
        scene_index: ``(num_scenes, 2)`` 각 장면의 [start, end) 에이전트 인덱스.
        meta: 장면별 (파일명, 시작 frame_id) — 오류 분석·시각화용.
    """

    pos: np.ndarray
    scene_index: np.ndarray
    meta: list[tuple[str, float]]
    obs_len: int
    pred_len: int
    agent_type: np.ndarray | None = (
        None  # (total_agents,) int8: 0=pedestrian/worker, 1=vehicle(지게차·AGV). ETH/UCY 는 None
    )

    def __len__(self) -> int:
        return len(self.scene_index)

    def scene(self, i: int) -> np.ndarray:
        s, e = self.scene_index[i]
        return self.pos[s:e]

    @property
    def num_agents(self) -> np.ndarray:
        return self.scene_index[:, 1] - self.scene_index[:, 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        files = np.array([m[0] for m in self.meta])
        starts = np.array([m[1] for m in self.meta], dtype=np.float64)
        agent_type = (
            self.agent_type
            if self.agent_type is not None
            else np.zeros(len(self.pos), dtype=np.int8)
        )
        np.savez_compressed(
            path,
            pos=self.pos,
            scene_index=self.scene_index,
            files=files,
            starts=starts,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            agent_type=agent_type,
        )

    @classmethod
    def load(cls, path: Path) -> SceneSet:
        z = np.load(path, allow_pickle=False)
        meta = list(zip(z["files"].tolist(), z["starts"].tolist()))
        agent_type = z["agent_type"] if "agent_type" in z.files else None
        return cls(
            pos=z["pos"],
            scene_index=z["scene_index"],
            meta=meta,
            obs_len=int(z["obs_len"]),
            pred_len=int(z["pred_len"]),
            agent_type=agent_type,
        )

    def types(self, i: int) -> np.ndarray:
        """장면 i 의 에이전트 타입 (없으면 전부 0)."""
        s, e = self.scene_index[i]
        if self.agent_type is None:
            return np.zeros(e - s, dtype=np.int8)
        return self.agent_type[s:e]

    def to_long_frame(self) -> pl.DataFrame:
        """EDA/DuckDB 용 long 포맷 (scene, agent, t, x, y)."""
        n_scene = len(self)
        counts = self.num_agents
        scene_col = np.repeat(np.arange(n_scene), counts * (self.obs_len + self.pred_len))
        agent_col = np.repeat(np.arange(len(self.pos)), self.obs_len + self.pred_len)
        t_col = np.tile(np.arange(self.obs_len + self.pred_len), len(self.pos))
        flat = self.pos.reshape(-1, 2)
        return pl.DataFrame(
            {"scene": scene_col, "agent": agent_col, "t": t_col, "x": flat[:, 0], "y": flat[:, 1]}
        )


def read_raw_file(path: Path, delim: str = "\t") -> pl.DataFrame:
    """공식 ``read_file`` 과 같은 결과(4열 float)를 Polars 로 읽는다."""
    df = pl.read_csv(
        path,
        separator=delim,
        has_header=False,
        new_columns=list(RAW_SCHEMA),
        schema_overrides=RAW_SCHEMA,
    )
    return df


def _windows_from_frame(
    data: np.ndarray, seq_len: int, skip: int, min_ped: int
) -> tuple[list[np.ndarray], list[float]]:
    """한 파일의 ``(rows, 4)`` 배열에서 장면 배열 목록을 만든다.

    반환되는 각 장면은 ``(N, seq_len, 2)`` 이며 보행자 순서는 공식 코드와 같이 ``np.unique(ped_id)``
    (오름차순) 순서다 — 그래프 노드 순서가 같아야 골든 값이 일치한다.
    """
    frames = np.unique(data[:, 0])
    frame_pos = {f: i for i, f in enumerate(frames)}
    # 보행자별 (frame index, x, y) — frame 순 정렬
    order = np.lexsort((data[:, 0], data[:, 1]))
    data = data[order]
    ped_ids, starts_idx = np.unique(data[:, 1], return_index=True)
    ends_idx = np.append(starts_idx[1:], len(data))
    ped_frames: dict[float, np.ndarray] = {}
    ped_xy: dict[float, np.ndarray] = {}
    for pid, s, e in zip(ped_ids, starts_idx, ends_idx):
        rows = data[s:e]
        ped_frames[pid] = np.array([frame_pos[f] for f in rows[:, 0]], dtype=np.int64)
        ped_xy[pid] = np.around(rows[:, 2:4], decimals=4)
    # frame index -> 그 프레임에 등장하는 보행자
    peds_at_frame: list[list[float]] = [[] for _ in frames]
    for pid, fidx in ped_frames.items():
        for fi in fidx:
            peds_at_frame[fi].append(pid)

    scenes: list[np.ndarray] = []
    starts: list[float] = []
    num_windows = len(frames) - seq_len + 1
    for idx in range(0, max(num_windows, 0), skip):
        cand = set()
        for fi in range(idx, idx + seq_len):
            cand.update(peds_at_frame[fi])
        kept: list[np.ndarray] = []
        for pid in sorted(cand):
            fidx = ped_frames[pid]
            # 윈도우 안에 있는 이 보행자의 프레임들
            m = (fidx >= idx) & (fidx < idx + seq_len)
            if m.sum() == 0:
                continue
            first, last = fidx[m][0], fidx[m][-1]
            # 공식 코드: pad_end - pad_front == seq_len 이어야 채택 (처음~끝이 윈도우 전체를 덮음)
            if last - first + 1 != seq_len:
                continue
            if m.sum() != seq_len:  # 중간 결손 — 공식 코드에서는 shape 오류가 나는 비정상 케이스
                continue
            kept.append(ped_xy[pid][m])
        if len(kept) > min_ped:
            scenes.append(np.stack(kept, axis=0))
            starts.append(float(frames[idx]))
    return scenes, starts


def build_scenes(
    raw_dir: Path,
    obs_len: int = 8,
    pred_len: int = 12,
    skip: int = 1,
    min_ped: int = 1,
    delim: str = "\t",
) -> SceneSet:
    """디렉터리의 모든 파일에서 장면을 만든다 (파일명 오름차순 = 공식 ``os.listdir`` 과 무관하게 결정적)."""
    seq_len = obs_len + pred_len
    all_scenes: list[np.ndarray] = []
    meta: list[tuple[str, float]] = []
    for path in sorted(p for p in Path(raw_dir).iterdir() if p.is_file()):
        data = read_raw_file(path, delim).to_numpy()
        scenes, starts = _windows_from_frame(data, seq_len, skip, min_ped)
        all_scenes.extend(scenes)
        meta.extend((path.name, s) for s in starts)
    counts = np.array([s.shape[0] for s in all_scenes], dtype=np.int64)
    ends = np.cumsum(counts)
    starts_i = ends - counts
    pos = (
        np.concatenate(all_scenes, axis=0).astype(np.float64)
        if all_scenes
        else np.zeros((0, seq_len, 2))
    )
    return SceneSet(
        pos=pos,
        scene_index=np.stack([starts_i, ends], axis=1),
        meta=meta,
        obs_len=obs_len,
        pred_len=pred_len,
    )
