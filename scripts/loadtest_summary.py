"""Locust CSV(--csv results/loadtest/run) → results/loadtest.json (+ 마크다운 표).

python scripts/loadtest_summary.py --users 10 --seconds 30 --note "학습 프로세스 2개와 동시 실행"
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-prefix", default="results/loadtest/run")
    ap.add_argument("--out", default="results/loadtest.json")
    ap.add_argument("--users", type=int, required=True)
    ap.add_argument("--seconds", type=int, required=True)
    ap.add_argument("--backend", default="onnx-fp32")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(ROOT / f"{a.csv_prefix}_stats.csv", encoding="utf-8")))
    out_rows = []
    for r in rows:
        name = r["Name"] if r["Type"] else "Aggregated"
        out_rows.append(
            {
                "name": name,
                "requests": int(r["Request Count"]),
                "failures": int(r["Failure Count"]),
                "rps": float(r["Requests/s"]),
                "p50_ms": float(r["50%"]),
                "p95_ms": float(r["95%"]),
                "p99_ms": float(r["99%"]),
                "max_ms": float(r["Max Response Time"]),
                "avg_ms": float(r["Average Response Time"]),
            }
        )
    table = [
        "| 엔드포인트 | 요청 | 실패 | RPS | p50 (ms) | p95 (ms) | p99 (ms) | 최대 (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in out_rows:
        table.append(
            f"| {r['name']} | {r['requests']:,} | {r['failures']} | {r['rps']:.1f} | {r['p50_ms']:.0f} | {r['p95_ms']:.0f} | {r['p99_ms']:.0f} | {r['max_ms']:.0f} |"
        )
    table.append(
        f"\n동시 사용자 {a.users} · {a.seconds} s · 백엔드 {a.backend} · 워커 1 · 요청당 에이전트 5–20, K=20"
        + (f" · {a.note}" if a.note else "")
    )
    payload = {
        "users": a.users,
        "seconds": a.seconds,
        "backend": a.backend,
        "note": a.note,
        "platform": platform.platform(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": out_rows,
        "table_markdown": "\n".join(table),
    }
    (ROOT / a.out).write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(table))


if __name__ == "__main__":
    main()
