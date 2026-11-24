"""离线单文件 HTML 报告与前端 React v2/referee_ml 全量对齐测试。"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

import report_frontend.report_generator as report_generator
from report_frontend import l1_face, l1_gesture, l1_text, l1_voice
from report_frontend.data_loader import LogDataLoader
from report_frontend.l1_report import build_l1_report
from report_frontend.report_generator import ReportGenerator

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_SID = "20260927_212143_95a8"

_BANNED = (
    "焦虑", "紧张", "压力", "抗压", "情绪稳定", "说谎", "诚信",
    "录用", "人格", "心理画像", "常模",
)
_JARGON = ("指标槽", "证据门", "过门", "同构念", "效标", "NONE 桶", "本层", "当前管线")
_SYMBOLS = ("⟹", "→", "⇒")
_INTERNAL_NOTATIONS = ("n_valid", "n_total", "**", "`")


def _all_l1_indicator_keys() -> set[str]:
    return (
        set(l1_face.indicator_names())
        | set(l1_gesture.ORDER)
        | set(l1_voice.ORDER)
        | set(l1_text.ORDER)
    )


def _text_of(html: str) -> str:
    body = re.sub(r"(?s)<!--.*?-->", "", html)
    body = re.sub(r"(?s)<(script|style).*?</\1>", "", body)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))


def _body_before_appendix(html: str) -> str:
    m = re.search(r"(?s)<details", html)
    return _text_of(html if m is None else html[: m.start()])


def _assert_common_html_hygiene(html: str, *, check_full_jargon: bool = True) -> None:
    assert "<iframe" not in html
    assert "plotly" not in html.lower()

    full_body = _text_of(html)
    before_app = _body_before_appendix(html)

    for word in _BANNED:
        assert word not in full_body, f"正文或附录出现了禁止词 {word!r}"

    jargon_scope = full_body if check_full_jargon else before_app
    for word in _JARGON:
        assert word not in jargon_scope, f"HTML 中出现了内部行话 {word!r}"

    for sym in _SYMBOLS:
        assert sym not in full_body, f"HTML 中出现了推理符号 {sym!r}"

    for tok in _INTERNAL_NOTATIONS:
        assert tok not in full_body, f"HTML 中出现了内部记号 {tok!r}"

    leaked = sorted(k for k in _all_l1_indicator_keys() if k in full_body)
    assert not leaked, f"HTML 中露出了内部指标键：{leaked}"

    long_dec = re.findall(r"\d+\.\d{8,}", full_body)
    assert not long_dec, f"HTML 中出现了 8 位以上小数：{long_dec[:5]}"

    assert before_app.count(",") == 0, f"正文出现了半角逗号：{before_app}"
    assert before_app.count(";") == 0, f"正文出现了半角分号：{before_app}"
    assert '"' not in before_app, "正文出现了半角直引号"
    stray = [m.group(0) for m in re.finditer(r".{6}:.{6}", before_app) if ":" in m.group(0)]
    assert not [s for s in stray if not re.search(r"\d:\d", s)], f"正文出现了非时间半角冒号：{stray[:3]}"


def _base_minimal_payload() -> dict:
    return {
        "engine": "l1",
        "dimensions": {
            "logical_thinking": {
                "display_name": "话语结构特征",
                "description": "",
                "algorithm": "",
                "evidence_chain": [
                    {
                        "feature": "discourse_marker_rate",
                        "human_name": "连接词密度",
                        "raw_value": 3.4,
                        "n_valid": 301,
                        "n_total": 301,
                        "std": None,
                        "observed_interval": [],
                        "instruments": ["text"],
                    }
                ],
                "evidence_gaps": ["皱眉：本场没有这个模态的日志"],
                "evidence_gaps_not_measured": [],
                "confidence": "中",
                "matched_indicators": "1/2",
                "stats": {},
            }
        },
        "vanished_dimensions": [],
        "summary_narrative": "未产出数值总结，仅呈现可核对的行为观测事实。",
        "coverage": {
            "n_slots": 3,
            "n_passed": 1,
            "n_dropped": 0,
            "n_measurable": 3,
            "n_not_measured": 0,
            "n_no_indicator": 0,
            "confidence_cap": "中",
            "summary_line": "可测量 3 槽中 1 槽通过（1/3） · 这套采集不测 0 槽 · 整维没有对应指标 0 槽",
            "explanation": "仅作可测量覆盖陈述。",
            "passed_slots": [],
        },
        "model_metadata": {"version": "l1-2", "timestamp": "2026-11-24T10:00:00"},
        "l1": {
            "n_total": 41,
            "n_measured": 1,
            "by_confidence": {},
            "by_gap": {},
            "modality_coverage": {},
            "modality_summary": {},
            "timebase": {},
            "windows": {"granularity": "answer", "n_windows": 4, "n_questions": 4},
            "indicators": [],
            "gaps": [],
            "comparability_notes": [],
            "comparability_notes_plain": [],
            "low_confidence_notes": [],
            "low_confidence_notes_plain": [],
            "dispersion": {
                "smile_ratio": {"iqr": 0.12, "max": 0.35, "median": 0.22, "min": 0.10, "n_questions": 4}
            },
            "instruments": {
                "labels": {"text": "文本"},
                "counts": {"text": 1},
                "order": ["text"],
                "footer": "共 1 条指标 · 仪器来源：文本 1",
                "n_multi": 0,
                "unmapped_columns": [],
            },
        },
    }


def _full_synthetic_payload() -> dict:
    p = copy.deepcopy(_base_minimal_payload())
    p["l1"]["referee_ml"] = {
        "available": True,
        "advisory_only": True,
        "non_scoring_guarantee": "仅做行为阶段切分与证据核验，不输出分数或个体评价",
        "evidence_gate": {
            "stage_code": "stage_2_review",
            "stage_label": "阶段 II · 证据可用需复核",
            "conformal_set": ["stage_2_review", "stage_1_verified"],
            "conformal_set_labels": ["阶段 II · 证据可用需复核", "阶段 I · 证据充分可核验"],
            "uncertainty_escalated": True,
            "target_coverage": 0.90,
            "empirical_cv_coverage": 0.945,
            "ood_anomaly_flag": True,
            "attributions": [
                {
                    "feature_key": "cross_modal_overlap_ratio",
                    "label": "跨模态有效重合度",
                    "value_display": "92.5%",
                    "contribution": 3.412,
                    "direction": "正向支撑",
                },
                {
                    "feature_key": "measured_ratio",
                    "label": "核心指标实测比例",
                    "value_display": "35.0%",
                    "contribution": -1.854,
                    "direction": "待补强",
                },
                {
                    "feature_key": "signal_usable_ratio",
                    "label": "传感窗口信号可用率",
                    "value_display": "96.0%",
                    "contribution": 1.21,
                    "direction": "正向支撑",
                },
            ],
            "summary_plain": "当前证据核验归为阶段 II · 证据可用需复核，主要正向支撑为跨模态有效重合度，待补强环节为核心指标实测比例。",
        },
        "stage_segmentation": {
            "available": True,
            "total_duration_sec": 120.0,
            "mean_dwell_sec": 30.0,
            "flicker_reduction_pct": 22.0,
            "macro_stages": [
                {
                    "stage": "pause_thinking",
                    "label": "停顿组织期",
                    "start_sec": 0.0,
                    "end_sec": 20.0,
                    "duration_sec": 20.0,
                    "color_key": "pause",
                    "ratio": 0.1667,
                    "annotation": "适应建立期",
                },
                {
                    "stage": "steady_speaking",
                    "label": "平稳展开期",
                    "start_sec": 20.0,
                    "end_sec": 70.0,
                    "duration_sec": 50.0,
                    "color_key": "steady",
                    "ratio": 0.4167,
                },
                {
                    "stage": "active_expressive",
                    "label": "高互动活跃期",
                    "start_sec": 70.0,
                    "end_sec": 100.0,
                    "duration_sec": 30.0,
                    "color_key": "active",
                    "ratio": 0.25,
                },
                {
                    "stage": "low_signal",
                    "label": "采集受限期",
                    "start_sec": 100.0,
                    "end_sec": 120.0,
                    "duration_sec": 20.0,
                    "color_key": "low_signal",
                    "ratio": 0.1667,
                },
            ],
            "change_points": [
                {"time_sec": 20.0, "magnitude": 1.82, "cue_summary": "发声连续度提升，头部动作趋平稳"},
                {"time_sec": 70.0, "magnitude": 2.35, "cue_summary": "上肢活动幅值上升，口部动态变化增强"},
            ],
            "summary_plain": "整场切分为 4 个连续行为阶段，共定位 2 处行为转折点。",
        },
    }
    p["l1"]["face_dual_instrument"] = {
        "available": True,
        "summary_plain": "共对照 2 组面部构念：1 组高度一致、1 组中度一致，主报告已按仪器分轨展示。",
        "pairs": [
            {
                "construct": "smile",
                "label": "嘴角上扬",
                "geometric_col": "au12_smile",
                "blendshape_cols": ["bs_mouthSmileLeft", "bs_mouthSmileRight"],
                "n_valid": 240,
                "spearman_rho": 0.782,
                "delta_spearman_rho": 0.654,
                "agreement_level": "高度一致",
            },
            {
                "construct": "eye_squeeze",
                "label": "眼周收紧",
                "geometric_col": "au7_eye_squeeze",
                "blendshape_cols": ["bs_eyeSquintLeft", "bs_eyeSquintRight"],
                "n_valid": 240,
                "spearman_rho": 0.512,
                "delta_spearman_rho": 0.431,
                "agreement_level": "中度一致",
            },
        ],
        "au15_pitch_diagnostic": {
            "available": True,
            "diagnostic_note": "诊断参考项：原始嘴角下压通道贴底占比 55.0%，与俯仰角线性相关，扣除俯仰分量后仍仅作参考。",
        },
    }
    p["l1"]["blendshape_calibration"] = {
        "available": True,
        "pose_mode": "3d_matrix",
        "n_raw_frames": 250,
        "n_clean_frames": 240,
        "quality_pass_rate": 0.96,
        "eligible_construct_count": 1,
        "summary_plain": "本场次共纳入 240 帧有效画面，在三维姿态矩阵校准后，1 项面部动作达到稳定融合条件。",
        "constructs": [
            {
                "key": "smile",
                "label": "嘴角上扬",
                "n_valid": 240,
                "raw_rho": 0.412,
                "calibrated_rho": 0.748,
                "rho_delta": 0.336,
                "pose_r2": 0.184,
                "eligibility": "已校准可融合",
                "note": "经时序去噪与三维头姿残差化后，两侧一致度明显提升。",
            },
            {
                "key": "brow_furrow",
                "label": "眉间收缩",
                "n_valid": 240,
                "raw_rho": 0.31,
                "calibrated_rho": 0.39,
                "rho_delta": 0.08,
                "pose_r2": 0.062,
                "eligibility": "辅助对照",
                "note": "存在一定口径差异，建议仅作辅助对照使用。",
            },
        ],
        "disclaimer": "本校准结果仅反映帧级一致性变化，不作状态结论。",
    }
    p["l1"]["feature_layer_v2"] = {
        "available": True,
        "pose_mode": "3d_matrix",
        "framework_tags": ["Praat-Parselmouth", "Silero-VAD-ONNX", "RTMW/MediaPipe-3D", "SPARC运动平滑度", "pycatch22时间序列动力学"],
        "tracks": {
            "catch22_dynamics": {"n_streams": 6},
        },
        "highlights": [
            {
                "key": "head_angular_velocity_3d_p50",
                "label": "头部三维角速度中位",
                "value": 12.45,
                "unit": "度/秒",
                "track": "face_kinematics",
                "note": "反映面部朝向每秒角度变化幅度",
            },
            {
                "key": "voiced_burst_regularity",
                "label": "发声时段规整度",
                "value": 0.684,
                "unit": "0-1",
                "track": "voice_dynamics",
                "note": "各段连续有声时长的一致性",
            },
            {
                "key": "sparc",
                "label": "上肢运动频谱平滑度",
                "value": -3.12,
                "unit": "SPARC",
                "track": "gesture_smoothness",
                "note": "动作速率频谱弧长，越接近 0 表示轨迹越平顺",
            },
            {
                "key": "low_freq_power_ratio_median",
                "label": "多模态低频功率集中度中位",
                "value": 0.625,
                "unit": "0-1",
                "track": "catch22_dynamics",
                "note": "跨通道时间序列慢变节律所占能量比例中位数",
            },
        ],
        "summary_plain": "面部采用三维姿态矩阵口径，头部角速度中位数为 12.45 度/秒；时间序列动力学共提取 6 条信号通道。",
        "inference_ms": 18.4,
    }
    return p


def test_minimal_payload_without_v2_or_referee_renders_cleanly():
    gen = ReportGenerator(output_dir="/tmp")
    base = _base_minimal_payload()
    html_missing = gen._build_html_report(base, {}, {}, {}, [])

    # 显式 available=False 也必须完全静默（不渲染空框或空附录）
    with_false = copy.deepcopy(base)
    with_false["l1"]["referee_ml"] = {"available": False}
    with_false["l1"]["blendshape_calibration"] = {"available": False}
    with_false["l1"]["face_dual_instrument"] = {"available": False}
    with_false["l1"]["feature_layer_v2"] = {"available": False}
    html_false = gen._build_html_report(with_false, {}, {}, {}, [])

    for html in (html_missing, html_false):
        body = _text_of(html)
        assert "行为阶段分段与证据核验" not in html
        assert "行为阶段分段与证据核验" not in body
        assert "行为阶段时间轴与转折点" not in html
        assert "阶段核验影响因素归因" not in html
        assert "附录：面部双仪器一致性与头部姿态校准对照" not in html
        assert "附录：第二层多通道动力学特征摘要" not in html
        _assert_common_html_hygiene(html)
        assert body.count("原始值") == 1


def test_css_parity_tokens_and_mobile_svg_rules_exist():
    html = ReportGenerator(output_dir="/tmp")._build_html_report(_full_synthetic_payload(), {}, {}, {}, [])
    css = re.search(r"(?s)<style>(.*?)</style>", html).group(1)
    for token in (
        ".referee-card", ".stage-badge", ".stage-verified", ".stage-review",
        ".stage-limited", ".cov-chip", ".attr-pos", ".attr-neg",
        "@media (prefers-color-scheme: dark)",
        "#2a78d6", "#0b5d2e", "#34a853", "#b26b00", "#e59a38", "#898781",
    ):
        assert token in css, f"样式表缺少 {token}"
    assert re.search(r"figure\s*\{[^}]*overflow-x\s*:\s*auto", css)
    assert re.search(r"figure\s+svg\s*\{[^}]*min-width", css)


def test_full_synthetic_payload_parity_and_hygiene():
    p = _full_synthetic_payload()
    html = ReportGenerator(output_dir="/tmp")._build_html_report(p, {}, {}, {}, [])

    _assert_common_html_hygiene(html)
    full_body = _text_of(html)
    before_app = _body_before_appendix(html)
    assert full_body.count("原始值") == 1

    # 正文（附录前）必须渲染行为阶段分段与证据核验卡片与两张 SVG
    assert "行为阶段分段与证据核验" in before_app
    assert "阶段 II · 证据可用需复核" in before_app
    assert "目标覆盖率 90%" in before_app
    assert "建议人工复核" in before_app
    assert "仅作行为阶段切分与证据核验，不输出分数或个体评价" in before_app
    assert "stage_1_verified" not in full_body
    assert "stage_2_review" not in full_body
    assert "stage_3_limited" not in full_body

    m_stage = re.search(r'(?s)<figure>.*?<svg[^>]*aria-label="行为阶段时间轴与转折点".*?</svg>.*?</figure>', html)
    assert m_stage, "缺少行为阶段时间轴 SVG"
    stage_fig = m_stage.group(0)
    for cls in ("stage-seg-pause", "stage-seg-steady", "stage-seg-active", "stage-seg-low", "cp-line", "cp-diamond"):
        assert cls in stage_fig
    assert ">0s<" in stage_fig
    assert ">120s<" in stage_fig
    assert "整场切分为 4 个连续行为阶段" in stage_fig
    assert "适应建立期" in before_app
    assert "20s 转折" in before_app

    m_attr = re.search(r'(?s)<figure>.*?<svg[^>]*aria-label="阶段核验影响因素归因".*?</svg>.*?</figure>', html)
    assert m_attr, "缺少阶段核验归因 SVG"
    attr_fig = m_attr.group(0)
    assert "attr-pos" in attr_fig and "attr-neg" in attr_fig
    assert "跨模态有效重合度" in attr_fig
    assert "+3.41" in attr_fig and "-1.85" in attr_fig

    # 两个新附录必须在 <details> 折叠区中，不能泄漏进正文前段
    assert "附录：面部双仪器一致性与头部姿态校准对照" in html
    assert "附录：面部双仪器一致性与头部姿态校准对照" not in before_app
    assert "三维姿态矩阵校准" in full_body
    assert "240 / 250" in full_body
    assert "96.0%" in full_body
    for col in ("面部动作", "校准前一致度", "校准后一致度", "变化量", "头姿解释占比", "融合判定"):
        assert col in full_body
    assert "+0.336" in full_body
    assert "诊断参考项：原始嘴角下压通道贴底占比 55.0%" in full_body

    assert "附录：第二层多通道动力学特征摘要" in html
    assert "附录：第二层多通道动力学特征摘要" not in before_app
    for tag in ("Praat-Parselmouth", "Silero-VAD-ONNX", "RTMW/MediaPipe-3D", "SPARC运动平滑度", "pycatch22时间序列动力学"):
        assert tag in full_body
    for tr in ("面部运动学", "语音动力学", "上肢平滑度", "多通道时序"):
        assert tr in full_body
    for bad_key in ("face_kinematics", "voice_dynamics", "gesture_smoothness", "catch22_dynamics", "head_angular_velocity_3d_p50"):
        assert bad_key not in full_body


def test_synthetic_legacy_segments_and_conformal_fallback_keys():
    """兼容 segments/macro_stage/changepoints/total_sec 与 conformal_set 缺省标签。"""
    p = _base_minimal_payload()
    p["l1"]["referee_ml"] = {
        "available": True,
        "evidence_gate": {
            "stage_code": "stage_1_verified",
            "stage_label": "阶段 I · 证据充分可核验",
            "conformal_set": ["stage_1_verified"],
            "uncertainty_escalated": False,
            "target_coverage": 0.90,
            "attributions": [],
        },
        "stage_segmentation": {
            "available": True,
            "total_sec": 60.0,
            "segments": [
                {"start_sec": 0.0, "end_sec": 40.0, "label": "平稳展开期", "macro_stage": "steady"},
                {"start_sec": 40.0, "end_sec": 60.0, "label": "停顿组织期", "macro_stage": "pause"},
            ],
            "changepoints": [{"time_sec": 40.0, "label": "停顿增多"}],
            "summary_plain": "整场共两段平稳切换。",
        },
    }
    p["l1"]["blendshape_calibration"] = {
        "available": True,
        "pose_mode": "2d_legacy",
        "n_raw_frames": 100,
        "n_clean_frames": 95,
        "quality_pass_rate": 0.95,
        "constructs": [],
    }
    html = ReportGenerator(output_dir="/tmp")._build_html_report(p, {}, {}, {}, [])
    _assert_common_html_hygiene(html)
    body = _text_of(html)
    assert "阶段 I · 证据充分可核验" in body
    assert "stage_1_verified" not in body
    assert "二维姿态代理校准" in body
    assert "95 / 100" in body
    assert ">60s<" in html
    assert "40s 转折：停顿增多" in body
    assert "阶段核验影响因素归因" not in html


def test_real_session_20260927_212143_95a8_html_parity(tmp_path, monkeypatch):
    log_dir = _REPO_ROOT / "data" / "logs"
    if not (log_dir / f"face_au_log_{_REAL_SID}.csv").is_file():
        pytest.skip("仓库内缺少真实会话 20260927_212143_95a8 日志文件")

    series_dir = _REPO_ROOT / "data" / "l0_series"
    if series_dir.is_dir():
        monkeypatch.setenv("JINGXIN_SERIES_DIR", str(series_dir))
    monkeypatch.setattr(report_generator, "LogDataLoader", lambda *a, **k: LogDataLoader(str(log_dir)))
    monkeypatch.setattr(report_generator.webbrowser, "open", lambda *a, **k: None)

    out_path = ReportGenerator(output_dir=str(tmp_path / "out")).generate_report(_REAL_SID)
    assert out_path, "真实会话未产出 HTML 报告"
    html = Path(out_path).read_text(encoding="utf-8")

    # 真实会话现有 parameter_registry.json 附录含「证据门」，因此正文前段必须零行话，全篇零禁用词/符号/内部键
    _assert_common_html_hygiene(html, check_full_jargon=False)
    before_app = _body_before_appendix(html)
    full_body = _text_of(html)

    assert "行为阶段分段与证据核验" in before_app
    assert 'aria-label="行为阶段时间轴与转折点"' in html
    assert 'aria-label="阶段核验影响因素归因"' in html
    assert "仅作行为阶段切分与证据核验，不输出分数或个体评价" in before_app

    assert "附录：面部双仪器一致性与头部姿态校准对照" in html
    assert "附录：面部双仪器一致性与头部姿态校准对照" not in before_app
    assert "二维姿态代理校准" in full_body
    assert "233 / 236" in full_body

    assert "附录：第二层多通道动力学特征摘要" in html
    assert "附录：第二层多通道动力学特征摘要" not in before_app
    assert "第二层多通道动力学特征" in full_body
    assert "面部运动学" in full_body
    assert "语音动力学" in full_body
    assert "上肢平滑度" in full_body
    assert "多通道时序" in full_body

    # 确认新增两个附录自身零行话且未重复打印「原始值」
    m_extra = re.findall(
        r"(?s)<details>\s*<summary>附录：(?:面部双仪器一致性与头部姿态校准对照|第二层多通道动力学特征摘要)</summary>.*?</details>",
        html,
    )
    assert len(m_extra) == 2
    for block in m_extra:
        b_txt = _text_of(block)
        assert "原始值" not in b_txt
        for word in _JARGON:
            assert word not in b_txt, f"新增附录含内部行话 {word!r}"
