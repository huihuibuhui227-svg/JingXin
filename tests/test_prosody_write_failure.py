# tests/test_prosody_write_failure.py
"""D2(最终全分支审查,triage 标「验收前」):一行没写进去**必须响亮失败**,不许吞掉。

**失效形态(修复前,已复现):** `VoiceLogger.log_prosody` 的整个写入体包在
`except Exception` 里 —— 失败时 `print` 一句 + `traceback.print_exc()` 然后 `return False`。
而它唯一的调用点(`voice_interaction/api/app.py` 的 `/interview/answer_audio`)**丢弃返回值**,
于是异常在两层里各被挡一次:

* 客户端拿到 **200** 和一份看着正常的响应(里面那个 `connective_density` 还是算出来的值);
* 报告侧那一行**凭空消失** —— 不出错、不留痕,只是有效样本量悄悄少一个。

这正是 M1 要杀的诚实轴,而且发生在 M1 自己造的那个槽上(连接词密度)。用户裁定:**报错**。

**这条测试只钉住被裁定改动的那一层**(logger 抛)。端点侧由
`voice_interaction/api/app.py` 的 `except Exception → HTTPException(500)` 接住,但
`voice_interaction.api.app` 在本环境**没有任何测试 import 它**(有意为之,见
`tests/test_voice_transcribe_helper.py` 的说明),所以"端点回 500"这半段**没有自动化覆盖**
—— 与审查者指出的结构性盲区同一条,别把本文件读成"端点已验"。
"""

from pathlib import Path

import pytest

from voice_interaction.utils.logger import VoiceLogger

_SID = "20260924_141139_ab31"


def _logger_with_unwritable_target(tmp_path: Path) -> VoiceLogger:
    """构造一个正常的 logger,再把落盘目标指到**父目录不存在**的路径。

    为什么不在构造时就给坏目录:`__init__` 会 `mkdir(parents=True)`,
    它会把坏目录直接建出来 —— 那样测的就不是写入失败了。构造后再改 `csv_file`,
    改的正是 `log_prosody` 真正拿去 `open(..., 'a')` 的那个属性。
    """
    lg = VoiceLogger(log_type="interview", log_dir=str(tmp_path), session_id=_SID)
    lg.csv_file = tmp_path / "no_such_dir" / "x.csv"
    assert not lg.csv_file.parent.exists(), "前提不成立:父目录居然存在"
    return lg


def test_log_prosody_raises_instead_of_returning_false(tmp_path):
    """写不进去 → **抛**,不是 `return False`。

    红在(修复前实测):`except Exception` 捕获后 `return False` → `pytest.raises` 不成立。
    调用点丢弃返回值,所以 `return False` 等于"什么都没发生"。
    """
    lg = _logger_with_unwritable_target(tmp_path)

    with pytest.raises(OSError):
        lg.log_prosody({}, question_index=0, emotion="", feedback="",
                       connective_density=3.0, connective_density_std=0.0, n_rows=1)


def test_a_successful_write_still_reports_success(tmp_path):
    """能写的时候照旧返回 `True`,并且行真的落了盘。

    这是修复的**另一侧**:把吞异常删掉时顺手把成功路径也弄坏(例如把 `return True`
    一起删了,或让 `with open` 那几行被包进别的分支),本测试变红。
    断言落在**文件内容**上而不是返回值上 —— 只看返回值的话,一个什么都不写的
    `return True` 也能过。
    """
    lg = VoiceLogger(log_type="interview", log_dir=str(tmp_path), session_id=_SID)

    assert lg.log_prosody({}, question_index=0, emotion="", feedback="",
                          connective_density=3.0, connective_density_std=0.0,
                          n_rows=1) is True

    body = lg.csv_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(body) == 2, f"表头 + 1 行,实际 {len(body)} 行"
    assert _SID in body[1], "首列应当是会话 id"
    assert "3.0" in body[1], "算出来的密度没写进去"
