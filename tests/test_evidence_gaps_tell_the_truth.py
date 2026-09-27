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

    ⚠️ 2026-09-27 换过取样槽:原样本是「困惑微表情」(关键词 `au4_freq`),而那条封停
    于本日**已解封**(原文「从 micro_exp 字符串解析出的垃圾匹配」被实测推翻:该槽的输入列
    `micro_exp_au_name` 在三场正式素材里 non-null = 0)⟹ 它现在说的是「未采集到对应数据」,
    而这**正是真话**(微表情确实一帧都没采到),再拿它当"封停被说成缺数据"的样本就反了。
    改用同一维度里**仍在封停**的「面部专注度」(`focus_score`,理由「3 个硬编码取值
    {0.3,0.5,0.8}」仍成立)—— 被钉的性质一字未改,只是换了一个仍然成立的样本。
    """
    gaps = _gaps("logical_thinking", _empty_face_feats())
    mine = [g for g in gaps if g.startswith("面部专注度")]
    assert mine, f"该槽不在缺口里,测不到:{gaps}"
    assert "该指标本轮停用" in mine[0], f"封停被说成了别的原因:{mine[0]}"
    assert "未采集到对应数据" not in mine[0], f"仍是那句假话:{mine[0]}"


def test_indicator_without_a_producer_says_so():
    """★ 2026-09-26 起,两个"没有产出方"的槽**都有了**。

    · 「回答详尽度」← 语音端点写 `n_chars`(字数,与连接词密度同一个计数器);
    · 「反应延迟」  ← 语音端点写 `reaction_time`(首次开口墙钟 − 该题 ask_end)。
    所以它们没数据时该说的是「这次没采到」(真话),不再是「没有产出方」。
    (机制本身由下一条 `test_the_no_producer_mechanism_still_works` 钉着。)
    """
    gaps = _gaps("cognitive_efficiency", _empty_face_feats())
    #    所以它们没数据时该说的是"这次没采到"(真话),不再是"没有产出方"。
    from report_frontend.research_mapper import NO_PRODUCER
    # ⚠️ 2026-09-27:此处的 `NO_PRODUCER == {}` 已换成**逐槽点名**。那条断言是
    #    「此刻没有任何槽缺产出方」的代理,而封停名单对账那一轮往名单里补了三个
    #    **真的**缺产出方的槽(`fluency_score` / `hand_score` / `shoulder_score`,
    #    见 `tests/test_report_layer.py` 的两条 NO_PRODUCER 钉子)—— 名单不该再是空的,
    #    但本测试要守的性质一字未改:这两个**有**产出方的槽不许出现在名单里。
    for slot in ("n_chars", "reaction_time"):
        assert slot not in NO_PRODUCER, (
            f"{slot} 已有产出方,却被登记成「没有产出方」—— 那会把"
            f"「这次没采到」说成「缺一个生产者」,读者会去改错的地方")
    for name in ("回答详尽度", "反应延迟"):
        mine = [g for g in gaps if g.startswith(name)]
        assert mine, f"{name} 不在缺口里:{gaps}"
        assert "尚无产出方" not in mine[0], f"{name} 已有产出方,不该再说缺产出方:{mine[0]}"
        assert "未采集到对应数据" in mine[0], f"{name} 该说「这次没采到」:{mine[0]}"


def test_the_no_producer_mechanism_still_works(monkeypatch):
    """名单空了,但**机制本身**要留着并被钉住 —— 将来再出现"映射表里有、全系统没人算"
    的槽,往名单里加一行就该立刻说真话。

    红法:把 `NO_PRODUCER.get(keyword)` 那一支删掉 ⟹ 加进去的条目不再生效。
    """
    from report_frontend import research_mapper
    monkeypatch.setitem(research_mapper.NO_PRODUCER, "reaction_time", "测试探针")
    gaps = _gaps("cognitive_efficiency", _empty_face_feats())
    mine = [g for g in gaps if g.startswith("反应延迟")]
    assert mine and "尚无产出方 —— 测试探针" in mine[0], f"机制失效:{mine}"


def test_reaction_time_slot_is_reachable_now():
    """★ 接线之后「反应延迟」要能**真的过门**(此前全系统没有生产者)。

    红法:把语音日志里那三列或映射关键词任一改掉 ⟹ 查不到,槽退回"没采到"。
    """
    feats = {"voice_interview": {"reaction_time_mean": 4.5, "reaction_time_std": 2.0,
                                 "reaction_time_sample_size": 12.0, "_n_rows": 12.0}}
    dim = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]["cognitive_efficiency"]
    names = [ev.get("human_name") for ev in dim["evidence_chain"]]
    assert "反应延迟" in names, f"该槽仍进不了证据链:{dim['evidence_gaps']}"
    ev = next(e for e in dim["evidence_chain"] if e["human_name"] == "反应延迟")
    assert ev["raw_value"] == 4.5 and ev["n_valid"] == 12


def test_answer_length_slot_is_reachable_now():
    """★ 接线之后,「回答详尽度」要能**真的过门** —— 不是只改了个说法。

    红法:把映射关键词改回 `text_avg_length`(而日志列叫 `n_chars`)⟹ 查不到,
    这个槽退回到"没有产出方"。这正是 §4.10 那条:上游产出名与下游消费名必须同名。
    """
    feats = {"voice_interview": {"n_chars_mean": 24.0, "n_chars_std": 6.0,
                                 "n_chars_sample_size": 12.0, "_n_rows": 12.0}}
    dim = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]["cognitive_efficiency"]
    names = [ev.get("human_name") for ev in dim["evidence_chain"]]
    assert "回答详尽度" in names, f"该槽仍进不了证据链:{dim['evidence_gaps']}"
    ev = next(e for e in dim["evidence_chain"] if e["human_name"] == "回答详尽度")
    assert ev["raw_value"] == 24.0 and ev["n_valid"] == 12


def test_length_scale_entry_follows_the_column_name():
    """量程表的键必须跟着列名走 —— 否则 30 字满量程那条登记**静默失效**、回落到默认 10。

    红法:把 `scale_factors` 里的键改回 `length` ⟹ 下面取到的是 `_default_scale_factor`。
    """
    from report_frontend.evidence_gate import scale_factor_for
    sf = scale_factor_for("voice_interview_n_chars_mean")
    assert sf["full_scale"] == 30.0, f"没命中原登记,取到的是兜底:{sf}"


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
