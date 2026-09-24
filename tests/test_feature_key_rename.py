# tests/test_feature_key_rename.py
"""连接词密度在报告侧只留一处定义,且量程随新定义走。

背景:该指标曾有两份定义 —— 语音模块按「每百字连接词数」写入日志列 `connective_density`
(Task 3),报告层 `feature_engine` 又从文本列按「命中数÷字符数」硬算一份同名指标。
两份定义同名不同义,报告里那个名字到底指哪个量,不可回答。Task 6 删掉报告层那份。
"""

from pathlib import Path

from report_frontend.research_mapper import ResearchCapabilityMapper

ROOT = Path(__file__).resolve().parent.parent


def test_mapper_indicator_is_renamed_and_matches_new_key():
    """映射元组必须改认新键、显示新名。

    红在:改名前关键字是 `logic_keyword_density`、显示名是「逻辑关键词密度」,
    `"connective_density" in keywords` 与 `"连接词密度" in names` 两条都取不到。
    """
    rule = ResearchCapabilityMapper().mapping_rules["logical_thinking"]
    keywords = [ind[0] for ind in rule["indicators"]]
    names = [ind[3] for ind in rule["indicators"]]
    assert "connective_density" in keywords
    assert "logic_keyword_density" not in keywords
    assert "连接词密度" in names
    # 权重/方向/重要性不随改名变 —— 元组形状 (keyword, weight, is_positive, human_name, importance)
    renamed = next(ind for ind in rule["indicators"] if ind[0] == "connective_density")
    assert renamed == ("connective_density", 0.4, True, "连接词密度", "core")


def test_feature_engine_has_no_second_definition_of_the_metric():
    """同名指标只能有一处定义 —— feature_engine 里的两张关键词表必须删净。

    红在:改名前 `logic_keywords = ['因为', '所以', ...]` 两张表都在文件里,
    两次 `"logic_keyword" not in text` 都不成立。

    静态扫描只钉得住**旧名字**;把表改名留着它抓不到,所以再喂一个文本列做行为断言。
    注释里出现 connective_density 是允许的(本仓对 .py 只取字符串字面量,注释豁免,
    见 test_report_layer._string_literals);不许出现的是**引擎自己算出来的键**。
    """
    text = (ROOT / "report_frontend/feature_engine.py").read_text(encoding="utf-8")
    assert "logic_keyword" not in text
    assert "logic_keywords" not in text

    import pandas as pd

    from report_frontend.feature_engine import PsychologicalFeatureEngine

    df = pd.DataFrame({"answer": ["因为所以但是", "然后我就说了很多话" * 5],
                       "duration_sec": [1.0, 2.0]})
    feats = PsychologicalFeatureEngine({"voice_research": df}).extract_all_features()
    keys = " ".join(feats.get("voice_research", {}).keys())
    # 非空前提:引擎确实跑了这个 df(否则下一条断言对任何实现都通过)
    assert "duration_sec" in keys, f"引擎没在这个 df 上跑起来,本测试退化为空断言:{keys!r}"
    assert "density" not in keys, f"报告层仍在自造密度特征:{keys}"


def test_new_key_can_pass_the_gate_and_score():
    """新列必须能真的走到打分路径,而不是只改了个名字。

    红在:改名前 mapper 找的是 `logic_keyword_density`,喂新键时 `_fuzzy_match` 空手而归
    → evidence_chain 为空。

    ⚠️ `_n_rows` 必须 ≥ 阈值表的默认样本量(10):该键不含阈值表里的任何词,样本量不足
    会被 G3 拦下 —— 那样本测试也会红,但红错了理由(「样本量不足」而不是「键没被映射到」)。
    """
    feats = {"voice_research": {"connective_density_mean": 3.5,
                                "connective_density_std": 0.4,
                                "_n_rows": 100.0}}
    dim = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]["logical_thinking"]
    assert dim["evidence_chain"], "证据链为空 —— 新键没被映射到"
    assert any(ev.get("human_name") == "连接词密度" for ev in dim["evidence_chain"]), (
        f"过门的槽不是连接词密度:{[ev.get('human_name') for ev in dim['evidence_chain']]}"
    )
    # 过门槽要带得走会话内变异(否则区间与 G2 都落空)
    ev = next(ev for ev in dim["evidence_chain"] if ev["human_name"] == "连接词密度")
    assert ev["std"] == 0.4 and ev["n_valid"] == 100


def test_real_voice_log_column_reaches_the_report():
    """端到端:按真实日志列名造的最小帧,必须一路走到打分路径。

    这条**不喂手写键名** —— 它钉的是那条管路本身:语音模块写下的列
    `connective_density` / `connective_density_std` 经 feature_engine 的通用数值路径
    产出带模态前缀的 `_mean`/`_std` 键,mapper 的 substring 匹配必须认出来。

    红在两处:
    - 改名前 mapper 认的是 `logic_keyword_density`,真实列产出的键一个都匹配不上
      → `ev is None`,报「日志列没进证据链」;
    - 改名后若量程没跟着定义走(仍 0.02),均值 3.54(每百字)会饱和到 1.0
      → `normalized_score` 断言变红。
    """
    import pandas as pd

    from report_frontend.feature_engine import PsychologicalFeatureEngine

    # 12 个回答行:模态行数 12 ≥ 阈值表默认的 10,连接词密度槽才过得了 G3
    values = [3.0, 4.0, 2.0, 5.0, 3.5, 4.5, 3.0, 2.5, 4.0, 3.5, 4.5, 3.0]
    df = pd.DataFrame({"session_id": ["20260924_120000"] * len(values),
                       "timestamp": range(len(values)),
                       "connective_density": values,
                       "connective_density_std": [0.5] * len(values),
                       "n_rows": [1] * len(values)})
    feats = PsychologicalFeatureEngine({"voice_research": df}).extract_all_features()
    engine_keys = sorted(feats.get("voice_research", {}))
    assert any(k.endswith("connective_density_mean") for k in engine_keys), (
        f"引擎没从日志列产出连接词密度的均值键:{engine_keys}"
    )

    dim = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]["logical_thinking"]
    ev = next((e for e in dim["evidence_chain"] if e["human_name"] == "连接词密度"), None)
    assert ev is not None, f"日志列没进证据链:{dim['evidence_gaps']}"
    # 均值 42.5/12 = 3.5417 → 会话内均值;满量程每百字 10 个 → 0.3542 → 四舍五入 0.35
    assert ev["raw_value"] == round(sum(values) / len(values), 4)
    assert ev["normalized_score"] == 0.35, (
        f"折算量程不对(旧量程 0.02 会饱和到 1.0):{ev['normalized_score']}"
    )


def test_density_scale_matches_the_per_hundred_definition():
    """量程必须跟着定义走:新定义是**每百字连接词数**(典型 0–10),不是旧的 0–1 比率。

    红在:改名前 density 族登记的是 `full_scale: 0.02`(旧公式「命中数÷字符数」的量程),
    于是 5.0(每百字 5 个)会被当成 0.02 满量程而**饱和**到 1.0,取不到 0.5;
    旧量程下这个指标任何真实取值都恒等于 1.0,等于没有量尺。
    """
    from report_frontend.evidence_gate import scale_factor_for

    spec = scale_factor_for("connective_density_mean")
    assert spec["kind"] == "full_scale", f"density 族不再是满量程型:{spec['kind']}"
    assert spec["full_scale"] == 10.0, (
        f"density 满量程未随定义更新:{spec['full_scale']}(旧公式的量程是 0.02)"
    )

    mapper = ResearchCapabilityMapper()
    assert mapper._dynamic_normalize(5.0, "connective_density_mean") == 0.5, \
        "每百字 5 个应落在量程中点"
    assert mapper._dynamic_normalize(10.0, "connective_density_mean") == 1.0
