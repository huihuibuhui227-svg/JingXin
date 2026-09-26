# tests/test_evidence_gaps_tell_the_truth.py
"""报告里的「证据缺口」必须说**真的原因**,而不是一句万能的「未采集到对应数据」。

背景(2026-09-26 实测,场 `20260926_124439_0f5b`):报告的 20 个槽里 6 个写着
「未采集到对应数据」,而那 6 个的真实原因**各不相同、且多数不是"没采到"**:

| 槽 | 真实原因 |
|---|---|
| 困惑微表情(au4_freq) | 该指标**已封停**(封停名单里写着"从 micro_exp 字符串解析出的垃圾匹配") |
| 眼部挤压(au7_freq) | 同上,已封停 |
| 眨眼频率(blink_rate) | 已封停;而**数据这次真的采到了**(`blink_rate_per_min` 229/275 行有值),只是被 `feature_engine` 的关键词表丢了 |
| 语音流畅度(fluency_score) | 已封停("生产分支为死代码") |
| 回答详尽度 / 反应延迟 | **全系统没有产出方**(全仓只有 mapping_rules 里那两行) |

一个封停是**指标自身的属性**,与这次采没采到数据无关 —— 所以它该在"找数据"之前
就定论;找不到数据时把封停说成"未采集到",读者会去调摄像头,而实际上调了也没用。
「没有产出方」又是第三件事:它不是采集问题,是**缺一个生产者**。

修法只改**真假话**,不让任何指标凭空通过:封停的照样封停(只是说法变准),
没有产出方的依旧不出分(只是点名"缺产出方"而不是"缺数据")。
"""
from pathlib import Path

import pandas as pd

from report_frontend.research_mapper import ResearchCapabilityMapper

ROOT = Path(__file__).resolve().parent.parent


def _gaps(dim_key: str, feats: dict) -> list[str]:
    dim = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"][dim_key]
    return dim["evidence_gaps"]


def _empty_face_feats() -> dict:
    """一份"面部分析跑起来了、但没有任何 AU 数据"的特征集。

    真正的空集不行:`_n_rows` 得在,否则 G3 那条路本身就测不到。
    """
    return {"face": {"focus_score_mean": 0.3, "focus_score_std": 0.02, "_n_rows": 100.0}}


def test_quarantined_indicator_reports_the_quarantine_not_missing_data():
    """已封停的指标,缺数据时也必须说「该指标本轮停用」。

    红在:改成先查封停之前,mapper 是先 `_fuzzy_match`,找不到就写
    「未采集到对应数据」—— 把真正的拦截原因(封停)盖住了。
    """
    gaps = _gaps("logical_thinking", _empty_face_feats())
    mine = [g for g in gaps if g.startswith("困惑微表情")]
    assert mine, f"该槽不在缺口里,测不到:{gaps}"
    assert "该指标本轮停用" in mine[0], f"封停被说成了别的原因:{mine[0]}"
    assert "未采集到对应数据" not in mine[0], f"仍是那句假话:{mine[0]}"


def test_indicator_without_a_producer_says_so():
    """没有产出方的指标要点名"缺产出方",不是"没采到"。

    红在:`text_avg_length` / `reaction_time` 全仓没有任何地方产出
    (`_fuzzy_match` 空手而归)⟹ 原先只会得到那句「未采集到对应数据」。
    """
    gaps = _gaps("cognitive_efficiency", _empty_face_feats())
    for name in ("回答详尽度", "反应延迟"):
        mine = [g for g in gaps if g.startswith(name)]
        assert mine, f"{name} 不在缺口里:{gaps}"
        assert "尚无产出方" in mine[0], f"{name} 没点名缺产出方:{mine[0]}"
        assert "未采集到对应数据" not in mine[0], f"{name} 仍是那句假话:{mine[0]}"


def _face_df(**extra) -> pd.DataFrame:
    """真日志的最小复刻:**必须带一个命中关键词表的列**。

    ⚠️ 少了 `focus_score` 这个 df 会走进 `_extract_face_features` 末尾的兜底分支
    (`if len(features) == 0: 全量数值扫描`)—— 于是**列全都进来了、测试假通过**,
    而真日志里 `focus_score`/`symmetry_score`/`tension_score` 都在,兜底从不触发。
    (2026-09-26 实测踩过:第一版测试没带关键词列,两条断言都是假通过的。)
    """
    return pd.DataFrame({"focus_score": [0.3, 0.3, 0.3], **extra})


def test_quarantine_does_not_swallow_the_session_specific_verdict():
    """有数据时,会话自身的结论(样本不足/无变化)不许被"封停"语盖掉。

    封停是"这个指标本轮不用",而样本不足是"这一场只答了 3 段"—— 后者才是使用者
    能据以行动的信息(多答几题)。把后者换成前者,报告就从"可改进"变成了"无话可说"。

    红在:把封停判定提到匹配**之前**(2026-09-26 我先就是这么写的)—— 那样
    `pitch_variation` 即便拿到了数据、样本量确实不够,也只会得到「该指标本轮停用」,
    而 §0.7 判过「有效样本不足」是对的。
    """
    feats = {"voice_interview": {"pitch_variation_mean": 220.0,
                                 "pitch_variation_std": 5.0,
                                 "pitch_variation_sample_size": 3.0,
                                 "_n_rows": 3.0}}
    gaps = _gaps("communication_fluency", feats)
    mine = [g for g in gaps if g.startswith("语调变化")]
    assert mine, f"该槽不在缺口里:{gaps}"
    assert "有效样本不足" in mine[0], f"会话自身的信息被吞了:{mine[0]}"


def test_blink_column_is_not_dropped_by_the_keyword_table():
    """`blink_rate_per_min` 是真采到的列(该场 229/275 行有值),不许被引擎丢掉。

    红在:`target_keywords` 里没有 blink,列名也不含 score/tension/... ⟹
    `_extract_face_features` 的数值分支把它跳过,于是报告说"未采集到"。
    """
    from report_frontend.feature_engine import PsychologicalFeatureEngine

    df = _face_df(blink_rate_per_min=[0.0, 12.0, 24.0], is_blink=[0, 1, 0])
    feats = PsychologicalFeatureEngine({"face": df}).extract_all_features()
    keys = set(feats.get("face", {}).keys())
    # 非空前提:兜底分支没被触发(否则本测试测的是另一条路)
    assert "face_focus_score_mean" in keys, f"没走到关键词那条路,测试退化了:{sorted(keys)}"
    assert "face_blink_rate_per_min_mean" in keys, f"眨眼列被丢了:{sorted(keys)}"


def test_numeric_au_columns_are_not_dropped_by_the_keyword_table():
    """26 个数值 AU 列同样被丢(`'au_' in 'au4_frown'` → False)。

    红在:表里只有 `'au_'` 与 `'AU'`,而日志列名形如 `au4_frown`(au 后没有下划线)。
    """
    from report_frontend.feature_engine import PsychologicalFeatureEngine

    df = _face_df(au4_frown=[0.0, 0.30, 0.55], au7_eye_squeeze=[0.10, 0.20, 0.30])
    feats = PsychologicalFeatureEngine({"face": df}).extract_all_features()
    keys = set(feats.get("face", {}).keys())
    assert "face_focus_score_mean" in keys, f"没走到关键词那条路,测试退化了:{sorted(keys)}"
    assert "face_au4_frown_mean" in keys, f"AU 数值列被丢了:{sorted(keys)}"
    assert "face_au7_eye_squeeze_mean" in keys, f"AU 数值列被丢了:{sorted(keys)}"
