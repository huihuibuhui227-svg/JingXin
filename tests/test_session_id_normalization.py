# tests/test_session_id_normalization.py
"""D3(最终全分支审查,triage 标「验收前」):「什么算没给 id」必须有一处定义、一处测试。

**失效形态(修复前,已复现):** 端点里写的是裸 `if not session_id:` 与
`session_id or NONE_SESSION`,于是同一层里三种说法并存:

| 客户端给的 | 修复前的判定 | 落在哪 |
|---|---|---|
| 不给参数 | 没给 | `NONE` |
| `session_id=""` | 没给(`not ""` 为真) | `NONE`(静默) |
| `session_id="  "` | **非法**(`not "  "` 为假,原样进正则 → 不匹配) | **400** |
| 直接调 `validate_session_id("")` | **非法** | 抛 `ValueError` |

同样是「客户端给了个空的」,一个静默归 `NONE`、一个报 400;而空串在"没给"与"非法"
两层里各执一词。三份 `_resolve_session_id`(voice/face/gesture)逐字相同,所以这不是
模块间分叉,是**同一层内**的口径分叉。

**方向:取「响亮失败」那一侧**,理由是这两个文件自己写下的原则 ——
`face_expression/api/app.py` 守卫的文档:「为什么不'顺手改成 NONE':坏输入被吞掉之后
客户端拿到的是 200 和一份看着正常的响应,问题只在报告里以'数据对不上'的形式浮出来。
本项目一贯要求坏输入响亮失败。」

所以规则定为:

* **参数没给**(`None`)→ `NONE`。这是无会话客户端的正常路径,不是坏输入。
* **给了但内容是空的**(`""` / `"  "`)→ **非法 → 400**。空串与"没给"在客户端那里
  是两件事:后者是没接会话,前者是**参数拼错了**(如 `?session_id=` 而变量为空)。
  把它静默归进 `NONE` 就是上面引文说的那种"问题只在报告里浮出来"。

修法:把这条规则提成具名函数 `normalize_session_id`,与判非法的 `validate_session_id`
成对定义、成对测试。三份副本由本文件末尾的源码级契约测试压着。
"""

import re
from pathlib import Path

import pytest

from voice_interaction.asr.session import NONE_SESSION
from voice_interaction.asr.transcript_store import (normalize_session_id,
                                                    validate_session_id)

_MINTED = "20260924_141139_ab31"


def test_absent_means_no_id():
    """参数不存在 = 没给 → `NONE`。

    红在(修复前):没有 `normalize_session_id`,本测试 ImportError。
    """
    assert normalize_session_id(None) == NONE_SESSION


def test_a_present_but_empty_value_is_not_absent():
    """给了参数但内容是空的 → 原样交出,由守卫判非法 —— **不得**静默归 `NONE`。

    红法:把 `normalize_session_id` 写成旧的裸写法(`return raw or NONE_SESSION`)。
    那时 `normalize_session_id("")` 会返回 `"NONE"` 而不是 `""`,第一条断言先红;
    就算把第一条改宽,第二个循环也会红 —— 吞掉空串之后守卫再也不会拒绝它。
    """
    assert normalize_session_id("") == "", "空串被静默归成了 NONE"
    assert normalize_session_id("   ") == "   ", "仅空白被静默归成了 NONE"

    for raw in ["", "   "]:
        with pytest.raises(ValueError):
            validate_session_id(normalize_session_id(raw))


def test_a_real_id_passes_through_untouched():
    """真的 id 必须原样通过(铸造形状与 `NONE` 自身都不能被改坏)。

    红法:把 `normalize_session_id` 写成总是返回 `NONE`。
    """
    assert normalize_session_id(_MINTED) == _MINTED
    assert normalize_session_id(NONE_SESSION) == NONE_SESSION


def test_the_absent_path_is_always_acceptable_to_the_guard():
    """「没给」这条路产出的值必须**必然**过守卫 —— 否则无会话客户端会拿到 400。

    这是两个函数的**复合**关系,单看任何一个都看不出来:
    `normalize_session_id` 负责"没给→NONE",`validate_session_id` 负责"NONE 合法"。
    红法:把 `NONE_SESSION` 改成不含法字符的常量(如 `"N/A"`),或让守卫拒绝 `NONE`。
    """
    validate_session_id(normalize_session_id(None))


# ── 三份副本的契约 ──────────────────────────────────────────────────────────
# 三个模块各自持有守卫的副本是**有意为之**(Ruling M1-2:face/gesture 不依赖 voice 包,
# 否则会把 TTS/ASR 整条 import 链拉起来)。代价是改一处必须改三处,所以再加一条源码级
# 契约测试压着 —— 与 test_session_id_contract.py 钉 `NONE_SESSION` 四处副本同一手法。

_REPO = Path(__file__).resolve().parents[1]
_COPIES = {
    "voice": "voice_interaction/asr/transcript_store.py",
    "face": "face_expression/api/app.py",
    "gesture": "gesture_analysis/api/app.py",
}


def _normalizer_source(rel_path: str) -> str:
    """抽出该文件里 `normalize_session_id` 的函数源码(从 def 行到下一个顶层语句)。"""
    text = (_REPO / rel_path).read_text(encoding="utf-8")
    m = re.search(r"^def normalize_session_id\(.*?(?=^\S|\Z)", text,
                  re.MULTILINE | re.DOTALL)
    assert m, f"{rel_path} 里没有 normalize_session_id —— 三份副本必须齐全"
    return m.group(0).strip()


def test_all_three_normalizer_copies_are_identical():
    """三份 `normalize_session_id` 必须逐字相同。

    红在(修复前):三份都还不存在 → `_normalizer_source` 的断言先红。
    红法:只改其中一份的实现(例如只给 voice 那处把空串判非法),另两份留着旧口径
    —— 那正是"三模块按 session 对上号"会静默失效的形态:同一个客户端请求在三个模块里
    得到不同判定。
    """
    sources = {name: _normalizer_source(p) for name, p in _COPIES.items()}
    distinct = set(sources.values())
    assert len(distinct) == 1, (
        "三份 normalize_session_id 已经分叉:"
        + "; ".join(f"{n}={p}" for n, p in _COPIES.items())
        + " —— 同一个客户端请求会在三个模块里得到不同判定"
    )
