"""브라우저 데모 생성 — 모델 가중치·재생 데이터를 JS 파일로 내보내고 README 용 GIF 를 그린다.

데모 페이지(``demo/index.html``)는 서버 없이 동작한다: ``demo/foresight.js`` 가 전처리·순전파·샘플링·위험 점수·
경보 정책을 순수 JS 로 다시 구현하고, ``demo/model.js`` (가중치)와 ``demo/replay.js`` (합성 RTLS 구역 60 s 재생)를
``<script>`` 로 읽는다. JS 구현은 ``tests/test_demo_js.py`` 가 Node 로 PyTorch 와 수치 비교한다.
"""

from foresight.demo.replay import (
    ReplayWindow,
    build_replay,
    export_weights,
    select_window,
    write_js,
)

__all__ = ["ReplayWindow", "build_replay", "export_weights", "select_window", "write_js"]
