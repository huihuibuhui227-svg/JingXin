### Task 2: 连接词密度(纯函数 + 标记表)

**Files:**
- Create: `voice_interaction/asr/connective_density.py`, `voice_interaction/asr/connective_markers.json`
- Test: `tests/test_connective_density.py`

**Interfaces:**
- Consumes: `funasr_engine.count_cjk_chars`
- Produces: `load_markers(path=None) -> dict`;`connective_density(text: str, markers=None, min_chars=None, counter=None) -> float | None`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_connective_density.py
from voice_interaction.asr.connective_density import connective_density, load_markers


def test_counts_markers_per_hundred_chars():
    text = "然后" + "字" * 98                       # 100 字,命中 1
    assert connective_density(text, min_chars=10) == 1.0


def test_punctuation_and_space_do_not_count_as_chars():
    a = connective_density("然后" + "字" * 98 + "。。。,,,   ", min_chars=10)
    assert a == 1.0


def test_below_min_chars_returns_none_not_zero():
    assert connective_density("然后好", min_chars=10) is None


def test_empty_text_returns_none():
    assert connective_density("", min_chars=10) is None


def test_no_marker_returns_zero_not_none():
    assert connective_density("字" * 50, min_chars=10) == 0.0


def test_marker_table_has_version_and_list():
    m = load_markers()
    assert m["version"]
    assert len(m["markers"]) >= 10
    assert "然后" in m["markers"] and "但是" in m["markers"]
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_connective_density.py -q`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现标记表与模块**

```json
// voice_interaction/asr/connective_markers.json
{
  "version": "1.0.0",
  "_provisional": true,
  "basis": "汉语话语连接标记的封闭清单。M1 首版:可辩护即可,不是标定产物。改表必须 bump version 并在报告里注明(spec §9.5)。",
  "markers": ["然后", "所以", "但是", "因为", "而且", "如果", "虽然", "不过",
              "因此", "另外", "其实", "首先", "其次", "最后", "总之", "比如", "例如"]
}
```

```python
# voice_interaction/asr/connective_density.py
"""连接词密度 = 连接词数 ÷ 字数 × 100(每百字)。

采集时计算(voice 模块有文本),数字进语音日志;原句不进仓库(spec D8)。
分母刻意不含时长 —— 时长类信息属于 M3 的时长线,把时长混进文本层指标
正是本项目栽过的"1410 维被时长污染"那条(spec D4)。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .funasr_engine import _config, count_cjk_chars

_MARKERS_PATH = Path(__file__).with_name("connective_markers.json")


@lru_cache(maxsize=1)
def load_markers(path: str | None = None) -> dict[str, Any]:
    return json.loads(Path(path or _MARKERS_PATH).read_text(encoding="utf-8"))


def connective_density(text: str, markers: list[str] | None = None,
                       min_chars: int | None = None,
                       counter: Callable[[str], int] | None = None) -> float | None:
    """返回每百字连接词数;文本过短或为空时返回 None(不出值,而不是写 0)。"""
    cfg = _config()["min_chars_for_density"]
    floor = min_chars if min_chars is not None else int(cfg["value"])
    chars = (counter or count_cjk_chars)(text or "")
    if chars < floor:
        return None
    table = markers if markers is not None else load_markers()["markers"]
    hits = sum(1 for m in table if m in (text or ""))
    return round(hits / chars * 100, 4)
```

- [ ] **Step 4: 运行,确认通过;提交**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_connective_density.py -q`
Expected: 6 passed

```bash
git add voice_interaction/asr/connective_density.py voice_interaction/asr/connective_markers.json tests/test_connective_density.py
git -c user.name="huihuibuhui227" -c user.email="huihuibuhui227@gmail.com" commit -m "feat(m1): 连接词密度(每百字)+ 版本化标记表"
```

---

