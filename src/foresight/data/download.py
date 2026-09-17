"""ETH/UCY 원본 파일 다운로드.

Social-STGCNN 공식 저장소가 vendoring 한 Social-GAN 포맷(`frame ped x y`) 파일을
raw.githubusercontent.com 에서 받아 sha256 으로 검증한다. 파일 목록과 해시는
`manifests/ethucy.json` 에 고정돼 있어 "무엇을 받았는가"가 코드와 함께 버전 관리된다.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from importlib import resources
from pathlib import Path

from foresight.utils import get_logger

log = get_logger(__name__)

SPLITS = ("eth", "hotel", "univ", "zara1", "zara2")
SUBSETS = ("train", "val", "test")


def load_manifest() -> dict:
    with (
        resources.files("foresight.data.manifests")
        .joinpath("ethucy.json")
        .open("r", encoding="utf-8") as f
    ):
        return json.load(f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_with_retry(url: str, timeout: int, attempts: int = 5) -> bytes:
    """전송 중단(IncompleteRead)·일시적 오류에 지수 백오프로 재시도한다. 프록시 뒤에서는 드물지 않다."""
    import http.client
    import time

    last: Exception | None = None
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except (http.client.IncompleteRead, OSError) as e:  # URLError 는 OSError 의 하위 클래스
            last = e
            wait = 2**i
            log.warning("retry %d/%d in %ds: %s (%s)", i + 1, attempts, wait, url, e)
            time.sleep(wait)
    raise RuntimeError(f"download failed after {attempts} attempts: {url}") from last


def download_ethucy(
    raw_dir: Path, splits: tuple[str, ...] = SPLITS, force: bool = False, timeout: int = 60
) -> list[Path]:
    """매니페스트의 모든 파일을 `raw_dir/<split>/<subset>/<file>` 로 내려받고 해시를 검증한다.

    이미 존재하고 해시가 맞는 파일은 건너뛴다(멱등). 해시 불일치는 즉시 예외 — 조용히 다른 데이터로
    학습하는 것이 재현 실험에서 가장 나쁜 실패이기 때문이다.
    """
    man = load_manifest()
    out: list[Path] = []
    for entry in man["files"]:
        if entry["split"] not in splits:
            continue
        dst = raw_dir / entry["split"] / entry["subset"] / entry["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not force and _sha256(dst) == entry["sha256"]:
            out.append(dst)
            continue
        url = man["base_url"].format(**entry)
        log.info("download %s", url)
        data = _fetch_with_retry(url, timeout)
        got = hashlib.sha256(data).hexdigest()
        if got != entry["sha256"]:
            raise RuntimeError(
                f"sha256 mismatch for {url}: expected {entry['sha256'][:12]}, got {got[:12]}"
            )
        dst.write_bytes(data)
        out.append(dst)
    log.info("ETH/UCY ready: %d files under %s", len(out), raw_dir)
    return out
