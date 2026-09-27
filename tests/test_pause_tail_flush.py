# tests/test_pause_tail_flush.py
"""停顿三条(C 档)的口径修复:① 循环加收尾、② 分母改成 `speech_duration_sec`。

三条 L0 行(`pause_duration_mean` / `pause_duration_max` / `pause_frequency`)的
`acceptance` 逐条抄在下面 —— 本文件只守**能做到**的那些子项:

  · `pause_duration_mean` ①:「循环加收尾(音频以静音结尾时那段停顿进 `pause_intervals`)」
  · `pause_duration_mean` ②:「合成信号单测:造「语音 + 尾部 4 秒静音」的输入,断言
    `pause_duration_max ≥ 4`(§1.2:55 的实测症状原文:「尾部 4s 静音报 0 次停顿」)」
  · `pause_frequency` ①:「分母由整段时长改成 `speech_duration_sec`」
  · `pause_frequency` ②:「加收尾」

**③「门限改由标定集给出」不在本文件内** —— 本仓没有标定集,M3.1 已被裁定跳过
(见 `docs/下一步.md` §0.12);那一类子项**原样留着**,由行的 `acceptance` 写明。

为什么不写"新旧口径都能过"的判据:本文件里每一条都先要求值**非 0**(旧口径下
尾部那段停顿被丢掉,这些量恒 0),再要求一个**只有新口径才成立**的关系:
  · 尾部越长,量到的停顿越长(旧口径两者都是 0);
  · 尾部 **4 秒 vs 8 秒**,`pause_frequency` **不变**(分母是语音秒;旧口径下会差一倍);
  · 总时长**相同**、语音秒**减半**时,`pause_frequency` 约**翻倍**(旧口径下两者相等)。

入口一律走活路径 `voice_interaction.api.app.prosody_features_from_pcm`(16k/16bit/单声道
PCM ⟹ 特征字典),不在测试里重写一遍公式 —— 但**合成信号**本身是本文件造的,
这是 `acceptance` 明文要的那条单测(真素材实测另见 `experiments/replay_retained.py`
的 `--modality voice` 腿)。
"""
import importlib

import numpy as np
import pytest

voice_app = importlib.import_module("voice_interaction.api.app")

SR = 16000


def _pcm(x: np.ndarray) -> bytes:
    """float 波形 → 16k/16bit 小端 PCM(活路径 `prosody_features_from_pcm` 的入口口径)。"""
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _speech_then_silence(tail_sec: float, speech_sec: float = 2.0,
                         lead_sec: float = 0.0) -> bytes:
    """`[前导静音] + 语音 + 尾部静音` 的 16k 单声道信号(180 Hz 正弦当真语音)。

    静音段是**逐样本 0** —— 能量门(`mean(rms) × 0.3`)必然判它为非语音。
    """
    n = int((lead_sec + speech_sec + tail_sec) * SR)
    t = np.arange(n) / SR
    x = np.zeros(n, dtype=np.float32)
    m = (t >= lead_sec) & (t < lead_sec + speech_sec)
    x[m] = 0.3 * np.sin(2 * np.pi * 180 * t[m])
    return _pcm(x)


def _features(pcm_bytes: bytes) -> dict:
    return voice_app.prosody_features_from_pcm(pcm_bytes)


def _all_silence(seconds: float) -> bytes:
    return _pcm(np.zeros(int(seconds * SR), dtype=np.float32))


# ---------------------------------------------------------------- ① 循环加收尾
def test_trailing_silence_is_counted_as_a_pause():
    """音频以静音结尾 ⟹ 那一段**必须**进 `pause_intervals`。

    红法(改前就是红的):`extract_pause_features` 的 for 循环**后面没有收尾** ——
    `in_pause` 在循环结束时仍为真,那段停顿整个丢掉。实测改前:尾部 4 秒静音,
    `pause_duration_max` = **0.0**。

    ⚠️ 为什么断言是 `≥ 3.9` 而不是 acceptance 原文的 `≥ 4`:停顿长度按"整窗安静"的
    帧数估,`librosa.feature.rms` 窗长 2048(0.128 s @16k)、跳步 512(0.032 s),
    所以每一段停顿的**两端各被吃掉不到一个窗** —— 4 秒的尾部量到 3.9x 秒。
    acceptance 的 `≥ 4` 与这条机制差一个窗宽,差值已写进行里(见报告),判据按机制取。
    """
    f = _features(_speech_then_silence(tail_sec=4.0))

    assert f["pause_duration_max"] > 0, (
        f"尾部 4 秒静音一段停顿都没量到(max={f['pause_duration_max']!r})—— 收尾没加")
    assert f["pause_duration_max"] >= 3.9, f
    assert f["pause_duration_mean"] >= 3.9, (
        f"整段只有这一处停顿,均值也该是它:{f['pause_duration_mean']!r}")


def test_trailing_pause_length_follows_the_tail_length():
    """尾部 4 秒 → 8 秒,量到的停顿必须**跟着长**(差 ≈ 4 秒)。

    红法:不收尾 ⟹ 两次都是 0.0,差值 0 —— 这条把"写死一个常数"也一并挡住了。
    """
    short = _features(_speech_then_silence(tail_sec=4.0))["pause_duration_max"]
    long_ = _features(_speech_then_silence(tail_sec=8.0))["pause_duration_max"]

    assert abs((long_ - short) - 4.0) < 0.3, (
        f"尾部长了 4 秒,量到的停顿只长了 {long_ - short:.2f} 秒({short!r} → {long_!r})")


def test_the_all_silence_clip_is_one_pause_of_the_whole_clip():
    """全静音 = 一整段"连续低于能量门"的帧 ⟹ 一处停顿,长度就是整段。

    这是收尾的**直接推论**,不是新口径:改前这一支 `pause_intervals` 是空的、
    写 0.0(「没测到」写成「量到 0」)。分母(语音秒)为 0 ⟹ `pause_frequency` 无定义、留空。
    """
    f = _features(_all_silence(6.0))

    assert f["pause_duration_max"] == 6.0, f
    assert f["speech_duration_sec"] == 0.0, f
    assert f["pause_frequency"] is None, (
        f"语音时长为 0 时「每分钟停顿数」无定义,不许写 0:{f['pause_frequency']!r}")


# ------------------------------------------- ② 分母改成 speech_duration_sec
def test_pause_frequency_denominator_is_speech_time_not_total_time():
    """**总时长相同**、语音秒不同 ⟹ 次数/分钟必须跟着变。

    两个信号都是 8 秒:`语音 4 秒 + 静音 4 秒` 与 `语音 2 秒 + 静音 6 秒`。
    停顿都只有尾部那 1 处。
      · 新口径(÷ 语音秒):后者的分母减半 ⟹ 频率约**翻倍**;
      · 旧口径(÷ 整段时长):两者都是 `1 ÷ 8 × 60 = 7.5`,**相等**。
    ⟹ 这条在旧口径下不可能过(改前两者都恒 0.0,连"相等"都不成立)。
    """
    longer_speech = _features(_speech_then_silence(tail_sec=4.0, speech_sec=4.0))
    shorter_speech = _features(_speech_then_silence(tail_sec=6.0, speech_sec=2.0))

    assert longer_speech["duration_sec"] == shorter_speech["duration_sec"] == 8.0, (
        longer_speech["duration_sec"], shorter_speech["duration_sec"])
    a, b = longer_speech["pause_frequency"], shorter_speech["pause_frequency"]
    assert a and b, f"尾部那处停顿没被算进频率:{a!r} / {b!r}"
    assert b == pytest.approx(2 * a, rel=0.2), (
        f"分母若真是语音秒,语音减半 ⟹ 频率翻倍;实测 {a!r} → {b!r}")


def test_growing_the_tail_does_not_change_pause_frequency():
    """同样的语音 + 更长的尾巴 ⟹ **频率不变**(分母是语音秒,不是整段时长)。

    红法:分母退回 `duration` ⟹ 8 秒尾巴那次的分母是 6 秒那次的 2 倍左右,
    频率当场掉一半。改前两次都是 0.0(尾部停顿没被算进分子)。
    """
    f4 = _features(_speech_then_silence(tail_sec=4.0))
    f8 = _features(_speech_then_silence(tail_sec=8.0))

    assert f4["pause_frequency"], f"尾部停顿没进分子:{f4['pause_frequency']!r}"
    assert f8["pause_frequency"] == pytest.approx(f4["pause_frequency"], rel=0.2), (
        f"尾巴从 4 秒拉到 8 秒,频率却从 {f4['pause_frequency']!r} 变到 "
        f"{f8['pause_frequency']!r} —— 分母还是整段时长?")


def test_zero_frequency_and_no_definition_are_different_values():
    """`0.0` 与 `None` 是两件事,按**分母**分:

      · 全是语音(分母 > 0、分子 0)⟹ `pause_frequency` 是**真值 0.0**;
      · 全是静音(分母 = 0)⟹ 无定义,留**空**。

    红法(改前两句话都是红的):改前两支都写 0.0 —— 而"分母为 0"那一支
    在收尾加上之后还多一种坏法:`0 秒 ÷ 0 秒` 会抛 `ZeroDivisionError`。
    两个方向一起钉:谁把这两个值统一了,这条就红。
    """
    all_speech = _features(_speech_then_silence(tail_sec=0.0, speech_sec=3.0))
    assert all_speech["speech_duration_sec"] > 0, all_speech
    assert all_speech["pause_frequency"] == 0.0, all_speech

    all_silence = _features(_all_silence(6.0))
    assert all_silence["speech_duration_sec"] == 0.0, all_silence
    assert all_silence["pause_frequency"] is None, all_silence
