"""API 요청/응답 스키마 (pydantic v2).

검증 규칙은 모델 계약에서 온다: 관측은 **정확히 8 점**(obs_len), 에이전트 1~200 명(N=200 이면
라플라시안 (8,200,200) ≈ 1.3 MB, 위험 행렬 (20,100,100,12) ≈ 20 MB — 한 요청의 상한으로 적당하다),
타입은 0(작업자)/1(차량). 잘못된 입력은 422 로 즉시 거절해 모델까지 가지 않게 한다.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

OBS_LEN = 8
PRED_LEN = 12
MAX_AGENTS = 200
MAX_K = 100

Point = Annotated[list[float], Field(min_length=2, max_length=2)]


class AgentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64, description="태그/에이전트 식별자")
    type: Literal[0, 1] = Field(default=0, description="0=작업자(보행자), 1=차량(지게차·AGV)")
    obs: list[Point] = Field(
        min_length=OBS_LEN,
        max_length=OBS_LEN,
        description=f"최근 {OBS_LEN} 프레임(0.4 s 간격) 절대좌표 [x, y] (m)",
    )

    @field_validator("obs")
    @classmethod
    def _finite(cls, v: list[list[float]]) -> list[list[float]]:
        for p in v:
            if any(x != x or abs(x) == float("inf") for x in p):
                raise ValueError("obs contains NaN/inf")
        return v


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agents: list[AgentIn] = Field(min_length=1, max_length=MAX_AGENTS)
    k: int = Field(default=20, ge=0, le=MAX_K, description="샘플 수 (0 이면 평균 궤적만)")
    seed: int | None = Field(default=None, description="샘플링 시드 (재현용)")
    include_samples: bool = Field(default=False, description="응답에 K 개 샘플 궤적을 포함할지")

    @field_validator("agents")
    @classmethod
    def _unique_ids(cls, v: list[AgentIn]) -> list[AgentIn]:
        ids = [a.id for a in v]
        if len(set(ids)) != len(ids):
            raise ValueError("agent ids must be unique")
        return v


class PredictionOut(BaseModel):
    id: str
    type: int
    mean: list[Point] = Field(description=f"{PRED_LEN} 스텝 평균 궤적 (절대좌표)")
    sigma: list[Point] = Field(description="스텝별 (σx, σy) — 상대 변위 분포의 표준편차 (m)")
    rho: list[float] = Field(description="스텝별 상관계수 ρ")
    samples: list[list[Point]] | None = Field(
        default=None, description="(K, 12, 2) 샘플 궤적 (include_samples=true)"
    )


class PredictResponse(BaseModel):
    predictions: list[PredictionOut]
    k: int
    backend: str
    timing_ms: dict[str, float]


class RiskRequest(PredictRequest):
    d_safe: float = Field(default=1.0, gt=0, le=50, description="안전 거리 (m)")
    threshold: float = Field(default=0.3, ge=0, le=1, description="경보 임계 (risk ≥ threshold)")
    deterministic: bool = Field(default=False, description="평균 궤적만으로 판정 (샘플링 없음)")


class RiskPairOut(BaseModel):
    worker_id: str
    vehicle_id: str
    risk: float
    ttc_s: float | None = Field(
        description="충돌 조건부 기대 최초 접근 시각(초); 충돌 샘플 없으면 null"
    )
    min_dist_mean: float


class RiskResponse(BaseModel):
    pairs: list[RiskPairOut]
    alerts: list[RiskPairOut] = Field(
        description="risk ≥ threshold 인 쌍 (HTTP API 는 무상태 — 쿨다운·연속 프레임 정책은 스트리밍 소비자에서 적용)"
    )
    n_workers: int
    n_vehicles: int
    k: int
    d_safe: float
    threshold: float
    backend: str
    timing_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    backend: str
    model_path: str
    model_sha256: str
    warm: bool
    params: int | None = None
    threads: int
