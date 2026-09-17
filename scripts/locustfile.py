"""Locust 부하 테스트 — ``POST /risk`` 에 5~20 명 에이전트(작업자·차량 혼합) 요청.

실행 (서버를 먼저 띄운 뒤):
    foresight serve --backend onnx --port 8000
    locust -f scripts/locustfile.py --headless -u 20 -r 20 -t 30s --host http://127.0.0.1:8000

궤적은 직선 + 약간의 잡음으로 만든다. 일부 쌍은 일부러 접근하게 해 경보 경로(alerts 가 비어 있지 않은
응답 직렬화)까지 부하에 포함한다.
"""

from __future__ import annotations

import math
import random

from locust import HttpUser, between, task

OBS_LEN = 8
DT = 0.4


def _trajectory(
    x0: float, y0: float, vx: float, vy: float, noise: float = 0.05
) -> list[list[float]]:
    return [
        [
            round(x0 + vx * DT * t + random.gauss(0, noise), 3),
            round(y0 + vy * DT * t + random.gauss(0, noise), 3),
        ]
        for t in range(OBS_LEN)
    ]


def make_payload(n_min: int = 5, n_max: int = 20, k: int = 20) -> dict[str, object]:
    n = random.randint(n_min, n_max)
    agents = []
    for i in range(n):
        vehicle = i % 3 == 0  # 1/3 은 차량
        speed = random.uniform(1.5, 3.0) if vehicle else random.uniform(0.8, 1.6)
        ang = random.uniform(0, 2 * math.pi)
        vx, vy = speed * math.cos(ang), speed * math.sin(ang)
        x0, y0 = random.uniform(-8, 8), random.uniform(-4, 4)
        agents.append(
            {
                "id": f"{'veh' if vehicle else 'wkr'}-{i}",
                "type": 1 if vehicle else 0,
                "obs": _trajectory(x0, y0, vx, vy),
            }
        )
    # 한 쌍은 정면 접근시켜 경보를 만든다
    if n >= 2:
        agents[0]["obs"] = _trajectory(-3.0, 0.0, 2.0, 0.0)
        agents[1]["obs"] = _trajectory(3.0, 0.2, -1.2, 0.0)
        agents[0]["type"], agents[1]["type"] = 1, 0
    return {"agents": agents, "k": k, "d_safe": 1.0, "threshold": 0.3}


class RiskUser(HttpUser):
    wait_time = between(0.005, 0.02)

    @task(8)
    def risk(self) -> None:
        self.client.post("/risk", json=make_payload(), name="/risk")

    @task(1)
    def predict(self) -> None:
        p = make_payload()
        self.client.post("/predict", json={"agents": p["agents"], "k": 20}, name="/predict")

    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="/health")
