"""results/*.json 의 마크다운 표를 README/docs 의 마커 블록에 주입한다 (숫자 마커는 tools/check_readme_numbers.py).

python scripts/sync_tables.py            # 주입
python scripts/sync_tables.py --check    # 최신인지 검사 (CI)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BLOCKS: dict[str, tuple[Path, str]] = {
    # 마커 이름: (JSON 파일, JSON 키)
    "REPRODUCTION_TABLE": (ROOT / "results" / "reproduction.json", "table_markdown"),
    "ABLATION_TABLE": (ROOT / "results" / "ablation.json", "table_markdown"),
    "RTLS_TRANSFER_TABLE": (ROOT / "results" / "rtls_transfer.json", "table_markdown"),
    "COLLISION_TABLE": (ROOT / "results" / "collision_eval.json", "table_markdown"),
    "TRAIN_COST": (ROOT / "results" / "train_cost.json", "table_markdown"),
    "SEEDS_TABLE": (ROOT / "results" / "seeds.json", "table_markdown"),
}
TARGETS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def render(name: str) -> str | None:
    path, key = BLOCKS[name]
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get(key)


def sync(check: bool) -> int:
    stale = 0
    for target in TARGETS:
        text = target.read_text(encoding="utf-8")
        new = text
        for name in BLOCKS:
            pat = re.compile(rf"(<!-- {name}:START -->\n)(.*?)(\n<!-- {name}:END -->)", re.S)
            table = render(name)
            if table is None:
                continue
            new = pat.sub(lambda m, t=table: f"{m.group(1)}{t}{m.group(3)}", new)
        if new != text:
            stale += 1
            if check:
                print(f"stale table block in {target.relative_to(ROOT)}")
            else:
                target.write_text(new, encoding="utf-8")
                print(f"updated {target.relative_to(ROOT)}")
    return 1 if (check and stale) else 0


if __name__ == "__main__":
    sys.exit(sync(check="--check" in sys.argv))
