# tests/test_brand_and_redline_compliance.py
"""全后端面向用户出口的品牌统一与红线合规守卫。

覆盖：
1. `templates/dashboard.html`（无嵌套 DOCTYPE、无旧名、无违规临床/情绪词）；
2. `exports.py`（转写导出头采用正式品牌、图表列不含已停用的 `focus_score`/`gaze_deviation`）；
3. `face_expression` 与 `gesture_analysis` 的面向用户反馈文案不含违规情绪/临床词；
4. `voice_interaction` 问答评估摘要与根路由文档不含“抗压/科研潜质评估”等违规词；
5. `demo_run.py` 输出标题已切换为「衡才智析」。
"""
from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

from report_frontend import brand

REPO = Path(__file__).resolve().parents[1]


def test_dashboard_template_is_valid_html_and_redline_clean():
    html = (REPO / "templates" / "dashboard.html").read_text(encoding="utf-8")
    assert html.count("<!DOCTYPE html>") == 1, "dashboard.html 存在重复嵌套 DOCTYPE"
    assert brand.BRAND_NAME in html
    hits = brand.find_redline_violations(html)
    assert hits == [], f"dashboard.html 命中红线词: {hits}"


def test_exports_header_and_face_other_columns_are_clean(tmp_path, monkeypatch):
    import exports
    import media_retention

    sid = "20261123_120000_abcd"
    sdir = tmp_path / sid
    sdir.mkdir(parents=True)
    (sdir / "transcript.json").write_text(
        '{"recorded_at":"2026-11-23","merged":{"n_chars":12,"text":"测试回答文本内容"},"segments":[{"index":0,"text":"测试回答文本内容"}]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    media_retention._reset_for_tests()

    txt = exports.transcript_txt(sid)
    assert txt.startswith(brand.TRANSCRIPT_EXPORT_HEADER)
    assert brand.find_redline_violations(txt) == []

    face_cols = {col for col, _label, _nd in exports._FACE_OTHER}
    assert "focus_score" not in face_cols
    assert "gaze_deviation" not in face_cols


def test_gesture_inferencer_outputs_objective_motion_phrases_only():
    from gesture_analysis.core.analysis.emotion_inferencer import EmotionInferencer

    inf = EmotionInferencer()
    for score in (95.0, 80.0, 65.0, 50.0, 35.0, 10.0):
        state, _icon = inf._map_score_to_emotion(score)
        fb, _color = inf._get_emotion_feedback(score)
        assert brand.find_redline_violations(state) == [], (score, state)
        assert brand.find_redline_violations(fb) == [], (score, fb)


def test_face_summary_and_voice_pipeline_have_no_redline_words():
    from face_expression.core.analysis.emotion_engine import EmotionEngine
    from voice_interaction.pipeline.assessment_pipeline import (
        InterviewAssessmentPipeline,
        ResearchAssessmentPipeline,
    )

    eng = EmotionEngine()
    dummy_au = type("AU", (), {"au_intensities": {"au4_frown": 0.5}, "psychological_signals": {"tension_score": 0.75}})()
    summary = eng._generate_psychological_summary({"neutral": 0.8}, dummy_au, None)
    assert brand.find_redline_violations(summary) == [], summary

    ip = InterviewAssessmentPipeline()
    ip.get_next_question()
    ip.add_answer("我在实验中验证了模型方法并与团队协作优化了算法逻辑。")
    iv_text = ip.get_comprehensive_evaluation()
    assert "不构成任何能力评分" in iv_text
    assert brand.find_redline_violations(iv_text) == [], iv_text

    rp = ResearchAssessmentPipeline()
    rp.get_next_question()
    rp.add_answer("通过反复实验验证假设并分析数据分布以排查模型漏洞。")
    rs_text = rp.get_comprehensive_evaluation()
    assert "不构成任何能力评分" in rs_text
    assert brand.find_redline_violations(rs_text) == [], rs_text


def test_voice_api_root_and_demo_run_have_no_redline_words():
    voice_app = importlib.import_module("voice_interaction.api.app")
    root_info = asyncio.run(voice_app.root())
    joined = " ".join(str(v) for v in root_info["endpoints"].values())
    assert brand.find_redline_violations(joined) == [], joined

    demo_src = (REPO / "demo_run.py").read_text(encoding="utf-8")
    assert "衡才智析 · 单场一键演示" in demo_src
    assert "pose_landmarker_full.task" in demo_src
    assert '("librosa",' in demo_src
