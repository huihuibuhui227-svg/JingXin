"""连接词密度 = 连接词数 ÷ 字数 × 100(每百字)。

采集时计算(voice 模块有文本),数字进语音日志;原句不进仓库(spec D8)。
分母刻意不含时长 —— 时长类信息属于 M3 的时长线,把时长混进文本层指标
正是本项目栽过的"1410 维被时长污染"那条(spec D4)。比率按字算,不是速率。

每命中一个标记只计一次(判"这个标记出现过"),不数出现次数:重复说同一个
连接词是冗词问题,不是结构丰富度,两者不该混在同一个数字里。

密度下限从 asr_config.json 的 min_chars_for_density 读,不在代码里写死。
低于下限或空文本返回 None,而不是 0.0 —— "没测出值"和"测出来是 0"是两件
不同的事,报告层对二者的渲染也不同。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .funasr_engine import count_cjk_chars, load_config

_MARKERS_PATH = Path(__file__).with_name("connective_markers.json")


@lru_cache(maxsize=1)
def load_markers(path: str | None = None) -> dict[str, Any]:
    """读版本化标记表(version / _provisional / basis / markers)。"""
    return json.loads(Path(path or _MARKERS_PATH).read_text(encoding="utf-8"))


def connective_density(text: str, markers: list[str] | None = None,
                       min_chars: int | None = None,
                       counter: Callable[[str], int] | None = None) -> float | None:
    """返回每百字连接词数;文本过短或为空时返回 None(不出值,而不是写 0)。"""
    cfg = load_config()["min_chars_for_density"]
    floor = min_chars if min_chars is not None else int(cfg["value"])
    chars = (counter or count_cjk_chars)(text or "")
    if chars < floor:
        return None
    table = markers if markers is not None else load_markers()["markers"]
    hits = sum(1 for m in table if m in (text or ""))
    return round(hits / chars * 100, 4)
