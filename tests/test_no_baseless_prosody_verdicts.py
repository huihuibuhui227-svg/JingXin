# tests/test_no_baseless_prosody_verdicts.py
"""后端那套**无出处硬编码阈值**的语音判语,必须不能出声(§3 第 28 条)。

`assessment_pipeline._analyze_prosody()` 原先按
`pitch_variation > 40` / `speech_ratio > 0.6` / `energy_mean` 0.5–0.8 这三条
**没有出处的阈值**下判语:「语调起伏大,富有表现力」「表达流畅」「音量适中」
「整体语音表达良好,继续保持!」。这三个数的依据都站不住:

- `speech_ratio` 是**自指阈值**,实测真数据里 8 段有 7 段恰为 1.0
  ⟹ 判语**恒为**「表达流畅」,不是量出来的;
- `energy_mean` 真值约 0.02–0.06,而阈值是 0.5–0.8 ⟹ 恒判「声音偏轻」;
- `pitch_variation` 实测可达 221 Hz(见 N1 账本:pyin 的 f0 轨迹不干净),
  阈值 40 一撞就出「富有表现力」。

**这个文件守两件事:**

1. **活路径**(`add_answer`)喂不出 prosody,所以这段**本来就不展开** ——
   这条要一直绿,否则说明有人悄悄把 prosody 接进了评测;
2. **真给它 prosody,它也不许下判语** —— 这条是防"哪天有人接上了"的雷。
   `add_qa_pair()` **接受** prosody 参数,所以这雷是**随时能引爆**的:
   实测喂一个 `pitch_variation=45 / speech_ratio=0.8 / energy_mean=0.5` 的假对象,
   它立刻吐「语调起伏大,富有表现力;表达流畅;音量适中」。没有依据的结论
   不许由系统说出 —— 这正是本项目要杀的形态。
"""
import importlib
from types import SimpleNamespace

import pytest

ap = importlib.import_module("voice_interaction.pipeline.assessment_pipeline")

EMPTY = "未获取到语音特征数据，无法进行语调分析。"

# 那套硬编码阈值会吐出来的判语。真给它 prosody 也不许出现其中任何一句。
BASELESS = ["富有表现力", "语调平缓", "不够自信", "表达流畅", "表达较连贯",
            "停顿较多", "略显犹豫", "声音洪亮", "声音偏轻", "音量适中",
            "整体语音表达良好", "继续保持", "增强感染力", "提升表达流畅度"]


def _fake_prosody(**kw):
    """一个"韵律分析结果"替身。默认值**故意挑在阈值的高档**:
    若那套阈值还在,它会一路吐出「富有表现力 / 表达流畅 / 音量适中」。"""
    fields = dict(pitch_variation=45.0, speech_ratio=0.8, energy_mean=0.65, is_valid=True)
    fields.update(kw)
    return SimpleNamespace(**fields)


def _pipeline(cls):
    p = cls()
    p.add_answer("我先讲一下我的项目经历")          # 与 /interview/answer_audio 同一调法
    return p


# ---------------------------------------------------------------- 第 1 件:活路径
@pytest.mark.parametrize("cls_name", ["InterviewAssessmentPipeline", "ResearchAssessmentPipeline"])
def test_live_path_says_it_has_no_prosody(cls_name):
    """活路径**喂不出** prosody,所以它只该说出"没拿到数据"这句实话。

    红法:把 `add_answer` 改成会把 prosody 塞进 `qa_pairs` —— 本测试仍会红在
    下面那句判语上,因为那时它就开始下结论了。
    """
    p = _pipeline(getattr(ap, cls_name))
    assert p.qa_pairs[-1].prosody_analysis is None, "活路径居然有 prosody 了?先看这是不是有意的"
    assert p._analyze_prosody() == EMPTY


# ------------------------------------------- 第 2 件:真给它 prosody,也不许下判语
@pytest.mark.parametrize("cls_name", ["InterviewAssessmentPipeline", "ResearchAssessmentPipeline"])
def test_supplied_prosody_does_not_yield_threshold_verdicts(cls_name):
    """`add_qa_pair()` 接受 prosody —— 这雷随时能引爆,引爆了也不许下判语。

    红法(改动前就是红的):让 `_analyze_prosody` 按老那套阈值判。
    """
    p = getattr(ap, cls_name)()
    p.add_qa_pair("请自我介绍", "我叫……", prosody_analysis=_fake_prosody())

    out = p._analyze_prosody()
    hit = [w for w in BASELESS if w in out]
    assert not hit, f"又按无出处的阈值下判语了:{hit}\n完整输出:{out!r}"


@pytest.mark.parametrize("cls_name", ["InterviewAssessmentPipeline", "ResearchAssessmentPipeline"])
def test_supplied_prosody_still_says_how_much_it_measured(cls_name):
    """删判语 ≠ 装看不见:量到了几段**要说**(那是事实,不是判断)。

    断言刻意写成 `"2 段"` 而不是 `"2"` —— 改动前那句输出里
    `【回答 2】` 本来就含一个 "2",拿 `"2" in out` 会**假绿**。
    """
    p = getattr(ap, cls_name)()
    p.add_qa_pair("请自我介绍", "我叫……", prosody_analysis=_fake_prosody())
    p.add_qa_pair("为什么选这个岗", "因为……", prosody_analysis=_fake_prosody())

    out = p._analyze_prosody()
    assert "2 段" in out, f"没说量到了几段:{out!r}"
