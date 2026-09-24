"""连接词密度 = 连接词数 ÷ 字数 × 100(每百字)。

采集时计算(voice 模块有文本),数字进语音日志;原句不进仓库(spec D8)。
分母刻意不含时长 —— 时长类信息属于 M3 的时长线,把时长混进文本层指标
正是本项目栽过的"1410 维被时长污染"那条(spec D4)。比率按字算,不是速率。

每命中一个标记只计一次(判"这个标记出现过"),不数出现次数:重复说同一个
连接词是冗词问题,不是结构丰富度,两者不该混在同一个数字里。匹配是子串匹配,
所以标记表内部不能有包含关系(否则一个词被两个标记各计一次);这条不变量由
tests/test_connective_density.py 里的表守卫测试压着,改表时它会红。

式中的 100 是"每百字"这个**单位**,不是可调阈值 —— 单位不写进数据文件,
免得"这个数字可以调"变成一种错觉。

密度下限从 asr_config.json 的 min_chars_for_density 读,不在代码里写死。
低于下限或空文本返回 None,而不是 0.0 —— "没测出值"和"测出来是 0"是两件
不同的事,报告层对二者的渲染也不同。
"""
from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .funasr_engine import count_cjk_chars, load_config

_MARKERS_PATH = Path(__file__).with_name("connective_markers.json")


@lru_cache(maxsize=1)
def _load_markers_cached(path: str | None = None) -> dict[str, Any]:
    """解析并缓存标记表本体。**私有**:外流的是副本,见 `load_markers`。"""
    return json.loads(Path(path or _MARKERS_PATH).read_text(encoding="utf-8"))


def load_markers(path: str | None = None) -> dict[str, Any]:
    """读版本化标记表(version / _provisional / basis / markers)。

    **交出副本,不是缓存本体。** 这个返回值会被报告路径消费(密度是要进报告的数字),
    而缓存是进程级的:任何调用方一次就地修改(`table["markers"].append(...)`、
    `table["version"] = ...`)都会留在缓存里,此后整个进程算出的密度都按被改过的表来
    —— 值就不再由 `connective_markers.json` 决定,而是由"谁先改过缓存"决定。
    深拷贝是最省事又最彻底的一层(顶层 dict 与嵌套 list 一并隔离);
    缓存仍然有效,省掉的是每次读盘+解析,不是这次拷贝。
    """
    return copy.deepcopy(_load_markers_cached(path))


def connective_density(text: str, markers: list[str] | None = None,
                       min_chars: int | None = None,
                       counter: Callable[[str], int] | None = None) -> float | None:
    """返回每百字连接词数;文本过短或为空时返回 None(不出值,而不是写 0)。"""
    cfg = load_config()["min_chars_for_density"]
    floor = min_chars if min_chars is not None else int(cfg["value"])
    chars = (counter or count_cjk_chars)(text or "")
    # `not chars` 单独写:空文本恒返 None,不依赖"下限恰好大于 0",也免去除零。
    if not chars or chars < floor:
        return None
    table = markers if markers is not None else load_markers()["markers"]
    hits = sum(1 for m in table if m in (text or ""))
    return round(hits / chars * 100, 4)
