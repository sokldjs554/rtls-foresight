"""mkdocs 훅: results/figures 와 demo/ 를 사이트에 복사한다.

문서는 그림을 `../results/figures/*.png` 로 참조한다 — GitHub 에서 바로 보이게 하려면 그림이 results/ 에 있어야 하고,
mkdocs 는 docs_dir 밖의 파일을 모르므로 빌드 후 같은 상대 경로가 되도록 site/results/figures 로 복사한다.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def on_post_build(config, **kwargs):
    root = Path(config["docs_dir"]).parent
    site = Path(config["site_dir"])
    src = root / "results" / "figures"
    if src.exists():
        shutil.copytree(src, site / "results" / "figures", dirs_exist_ok=True)
    demo = root / "demo"  # 브라우저 데모(정적 파일) → <site>/demo/
    if demo.exists():
        shutil.copytree(demo, site / "demo", dirs_exist_ok=True)
