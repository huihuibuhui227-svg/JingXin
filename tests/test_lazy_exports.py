"""M1-1:voice_interaction/__init__.py 必须是惰性导出(PEP 562)。

父包总会在任何子模块之前被执行,所以 eager 的 __init__ 会让
`import voice_interaction.asr.session` 顺带构造 TTS 与音频设备/librosa 管线 ——
慢,而且会因为与被测代码毫无关系的原因失败。
(原先这里还有 vosk;它已随 2 GB 模型删除,故不再进 `HEAVY`。)

本文件里只有 test_lightweight_import_pulls_no_heavy_pipeline 是"改前红、改后绿"
的驱动测试;另外两个是该改动的回归护栏(名字解析、AttributeError 语义)。
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# 只有重管线会拖进来的顶层包(vendored funasr_client 只要 numpy,故不含 numpy)
#
# ⚠️ 这张表**只许放本环境真装着的包**。`"vosk"` 曾在这里,而 vosk 已随 2 GB 模型
# 从 requirements 删除 ⟹ `"vosk" in sys.modules` 恒假,那条参数化用例对它
# **零约束力**(永远绿,且永远不会告诉任何人它已经失效)。
# 这正是「一个测试被当成既有缺陷之后就不再被追问」的同一形态 —— 见
# `test_heavy_names_are_actually_installed` 那条守卫。
HEAVY = ("sounddevice", "librosa")


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                          capture_output=True, text=True)


def test_heavy_names_are_actually_installed():
    """`HEAVY` 里的每一项必须真的能导入 —— 否则它守不住任何东西。

    红法:往 `HEAVY` 里加回 `"vosk"`(或任何未安装的包)⟹ 本条红。
    判据不是"包名眼熟",是 `find_spec` 能不能找到。
    """
    missing = [m for m in HEAVY if importlib.util.find_spec(m) is None]
    assert missing == [], (
        f"HEAVY 里这些包在本环境不存在:{missing} —— "
        f"它们对应的断言恒绿(零约束力)。要么从 HEAVY 删掉,要么先装上。"
    )


@pytest.mark.parametrize("stmt", [
    "import voice_interaction",
    "import voice_interaction.asr.session as s; assert s.NONE_SESSION == 'NONE'",
])
def test_lightweight_import_pulls_no_heavy_pipeline(stmt):
    proc = _run(f"{stmt}; import sys; print(','.join(m for m in {HEAVY!r} if m in sys.modules))")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", f"轻量导入被拖进了:{proc.stdout.strip()}"


def test_every_name_in_all_resolves():
    """惰性映射表必须覆盖 __all__ 的每一项(映射写错这里就红)。"""
    import voice_interaction

    assert voice_interaction.__all__, "__all__ 不能为空"
    missing = [n for n in voice_interaction.__all__ if getattr(voice_interaction, n, None) is None]
    assert missing == []


def test_from_import_forms_still_work():
    """老的 `from voice_interaction import X` 写法必须一字不改地继续可用。"""
    from voice_interaction import (SpeechRecognitionResult, ProsodyFeatureExtractor,
                                   TTSPipeline, InterviewAssessmentPipeline)

    assert ProsodyFeatureExtractor is not None
    assert SpeechRecognitionResult is not None
    assert TTSPipeline is not None
    assert InterviewAssessmentPipeline is not None


def test_unknown_attribute_raises_attributeerror():
    """__getattr__ 不能把未知名字悄悄变成 None。"""
    import voice_interaction

    with pytest.raises(AttributeError):
        voice_interaction.NoSuchThingInThisPackage
