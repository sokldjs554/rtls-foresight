"""README/docs 의 수치를 results/*.json 과 동기화하거나(--check 없이) 최신인지 검사한다(--check).

마커 문법 (이전 프로젝트들과 같은 관례):
    <!-- num:reproduction.ours.avg.best_of_k_per_agent.ade:.2f -->0.50<!-- /num -->

    key  = "<results 파일 stem>.<JSON 안의 점 경로>" (배열 인덱스는 숫자)
    포맷 = 선택, 파이썬 format spec (기본 ".2f"; 정수는 ",d" 등)

    python tools/check_readme_numbers.py            # 갱신
    python tools/check_readme_numbers.py --check    # CI: 오래된 값이 있으면 1
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TARGETS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("**/*.md"))]
MARK = re.compile(r"<!-- num:([A-Za-z0-9_.\-/+\[\]]+?)(?::([^ >]+))? -->(.*?)<!-- /num -->", re.S)

_cache: dict[str, dict] = {}


def _load(stem: str) -> dict | None:
    if stem not in _cache:
        p = RESULTS / f"{stem}.json"
        _cache[stem] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None  # type: ignore[assignment]
    return _cache[stem]


def resolve(key: str) -> object | None:
    stem, _, path = key.partition(".")
    data = _load(stem)
    if data is None:
        return None
    cur: object = data
    for part in path.split(".") if path else []:
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


def render(value: object, spec: str | None) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return format(value, spec or (".2f" if isinstance(value, float) else "d"))
    return str(value)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:  # 테스트의 임시 디렉터리 등 저장소 밖 경로
        return str(path)


def _substitute(text: str, label: str, stale: list[str], missing: list[str]) -> str:
    def repl(m: re.Match[str]) -> str:
        key, spec, old = m.group(1), m.group(2), m.group(3)
        val = resolve(key)
        if val is None:
            missing.append(f"{label}: {key}")
            return m.group(0)
        new = render(val, spec)
        if new != old:
            stale.append(f"{label}: {key} {old!r} -> {new!r}")
        head = f"<!-- num:{key}{':' + spec if spec else ''} -->"
        return f"{head}{new}<!-- /num -->"

    return MARK.sub(repl, text)


def process(check: bool) -> int:
    stale: list[str] = []
    missing: list[str] = []
    for target in TARGETS:
        if not target.exists():
            continue
        text = target.read_text(encoding="utf-8")
        new_text = _substitute(text, _rel(target), stale, missing)
        if new_text != text and not check:
            target.write_text(new_text, encoding="utf-8")
    for m_ in missing:
        print(f"[missing] {m_}")
    for s_ in stale:
        print(f"[{'stale' if check else 'updated'}] {s_}")
    if check and (stale or missing):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(process(check="--check" in sys.argv))
