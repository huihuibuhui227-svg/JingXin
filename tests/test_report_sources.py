# tests/test_report_sources.py
"""I4(最终全分支审查,triage 标「可现在就做」):报告必须说清**自己是哪些日志装配的**。

**失效形态(修复前,已复现):** 报告表头只有生成时间,三个模态各自按**文件名时间戳**取最新
(`data_loader.py` 的 `max(timestamp_val)`),而加载器算出来的那个 `session_id` 唯一的消费者
是两处 `print()`。于是当某个模态**本场没产出**时(验收配置下 gesture 服务已死,face 也一样),
报告会把**本场的 voice+face** 与**上一场的 gesture** 静默配在一起 —— 读者毫无提示,而报告
通篇没有任何一处能让他发现这件事。

M1-22 已把"选择策略"(按 id 选、不匹配回 409)推给 M2;这里只做**披露**:把三个被选中的
session id 打进表头,并标明是否同场。披露比策略便宜,而且它正是让"策略还没修"这件事
**可见**的那一步 —— 没有它,M2 的缺陷是隐形的。

数据源用的是 `data_loader` 里**已有的**那个 `session_id`(由文件名正则反推),不新增口径。
"""

import csv
from pathlib import Path

import pytest

import report_frontend.report_generator as report_generator
from report_frontend.data_loader import LogDataLoader
from report_frontend.report_generator import ReportGenerator, sources_disclosure

_THIS_SESSION = "20260924_141139_ab31"      # M1 形态(带 4 位十六进制随机段)
_PREV_SESSION = "20260315_103759"           # 旧形态(纯时间戳)—— 上一场


def _write_log(log_dir: Path, filename: str, row: dict) -> Path:
    """写一份最小可加载的模态日志(表头 + 1 行),文件名决定它的 session_id。"""
    path = log_dir / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    return path


def _face_row(sid: str) -> dict:
    return {"session_id": sid, "timestamp": "2026-09-24T14:11:39",
            "focus_score": 0.5, "symmetry_score": 0.9}


def _voice_row(sid: str) -> dict:
    return {"session_id": sid, "timestamp": "2026-09-24T14:11:39",
            "connective_density": 3.0, "connective_density_std": 0.5}


def _gesture_row(sid: str) -> dict:
    return {"session_id": sid, "timestamp": "2026-03-15T10:37:59", "emotion_score": 50}


def _three_modality_logs(log_dir: Path, gesture_sid: str) -> None:
    """本场的 face+voice,加一场可指定来源的 gesture(用来造"跨场拼接")。"""
    _write_log(log_dir, f"face_au_log_{_THIS_SESSION}.csv", _face_row(_THIS_SESSION))
    _write_log(log_dir, f"interview_emotion_log_{_THIS_SESSION}.csv", _voice_row(_THIS_SESSION))
    _write_log(log_dir, f"gesture_emotion_log_{gesture_sid}.csv", _gesture_row(gesture_sid))


# ---------------------------------------------------------------------------
# 一、加载器要把它选了谁**交出来**(供报告层用)
# ---------------------------------------------------------------------------

def test_loader_exposes_which_session_each_modality_came_from(tmp_path):
    """`selected_sessions` 键与数据键同名,值是那个文件自报的 session id。

    红在(修复前):没有 `selected_sessions`,本测试 AttributeError。
    红法:只在 `print()` 里报选中了谁、不落到结构化字段上 —— 报告层就拿不到(这正是修复前的样子)。
    """
    _three_modality_logs(tmp_path, gesture_sid=_PREV_SESSION)
    loader = LogDataLoader(str(tmp_path))
    loader.get_fused_latest_data()

    assert loader.selected_sessions == {
        "face": _THIS_SESSION,
        "gesture": _PREV_SESSION,
        "voice_interview": _THIS_SESSION,
    }, loader.selected_sessions


def test_selected_sessions_only_lists_modalities_that_really_loaded(tmp_path):
    """被选中但**读不出来**(空文件)的模态不得出现在数据来源里。

    否则报告会声称"用了一个根本没有数据的模态",比不说还坏。
    红法:在选文件的循环里填 `selected_sessions`(那时还不知道读不读得出来)。
    """
    _three_modality_logs(tmp_path, gesture_sid=_PREV_SESSION)
    (tmp_path / f"gesture_emotion_log_{_PREV_SESSION}.csv").write_text("", encoding="utf-8")

    loader = LogDataLoader(str(tmp_path))
    data = loader.get_fused_latest_data()

    assert "gesture" not in data, "前提不成立:空文件居然加载进去了"
    assert "gesture" not in loader.selected_sessions, loader.selected_sessions
    assert set(loader.selected_sessions) == set(data), (
        "数据来源的键必须与真正加载到的模态一一对应"
    )


# ---------------------------------------------------------------------------
# 二、披露文字本身
# ---------------------------------------------------------------------------

def test_disclosure_names_every_session_and_flags_a_mismatch():
    """模态不同场时必须**逐一点名**并明确警告。

    红在(修复前):没有 `sources_disclosure`,本测试 ImportError。
    """
    text = sources_disclosure({"face": _THIS_SESSION,
                               "gesture": _PREV_SESSION,
                               "voice_interview": _THIS_SESSION})

    assert _THIS_SESSION in text and _PREV_SESSION in text, f"没点名来源:{text}"
    assert "不同" in text, f"跨场拼接没有被指出来:{text}"


def test_disclosure_says_so_when_every_modality_agrees():
    """全部同场时也要说清是同一场 —— 否则读者无法区分"同场"与"没检查"。

    红法:让 `sources_disclosure` 只在**不一致**时返回内容,一致时返回空串。
    """
    text = sources_disclosure({"face": _THIS_SESSION, "voice_interview": _THIS_SESSION})

    assert _THIS_SESSION in text
    assert "同一场会话" in text, f"同场时没给出肯定说法(读者无法区分「检查过且一致」与「没检查」):{text}"
    assert "并非同一场面试" not in text, f"同场却被报成了不同场:{text}"


# ---------------------------------------------------------------------------
# 三、完整批处理路径:表头里真的出现了
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_report(tmp_path, monkeypatch):
    """把 `generate_report` 的加载目录与浏览器打开都换掉,只保留装配与渲染。"""
    def _run(gesture_sid: str) -> str:
        log_dir = tmp_path / "logs"
        log_dir.mkdir(exist_ok=True)
        _three_modality_logs(log_dir, gesture_sid=gesture_sid)

        monkeypatch.setattr(report_generator, "LogDataLoader",
                            lambda *a, **k: LogDataLoader(str(log_dir)))
        monkeypatch.setattr(report_generator.webbrowser, "open", lambda *a, **k: None)

        gen = ReportGenerator(output_dir=str(tmp_path / "out"))
        report_path = gen.generate_report()
        assert report_path, "报告没生成 —— 本测试的前提不成立"
        return Path(report_path).read_text(encoding="utf-8")

    return _run


def test_report_header_lists_the_sessions_it_was_assembled_from(batch_report):
    """跨场拼接必须**出现在报告里**,让读者自己看得见。

    这是本项存在的理由:验收配置下 gesture 起不来,报告会把本场的 face+voice 与上一场的
    gesture 配在一起,而修复前读者**没有任何线索**。

    红在(修复前):表头只有生成时间,`_THIS_SESSION` 与 `_PREV_SESSION` 都不出现 → 断言失败。
    """
    html = batch_report(gesture_sid=_PREV_SESSION)

    assert _THIS_SESSION in html, "报告没说自己用的是哪一场"
    assert _PREV_SESSION in html, "上一场的 gesture 被静默拼进来,报告里没有任何提示"
    # 用只属于披露段的句子,不用裸"不同" —— 报告别处本来就有「不同指标的量纲无法折算为
    # 同一尺度」(零输入那段摘要),拿裸词断言等于恒真,这条测试会失去约束力。
    assert "并非同一场面试" in html, "跨场拼接没有被标注出来"


def test_report_header_reports_a_single_session_when_they_agree(batch_report):
    """同场时表头也要给出这三个 id —— 这样"报告描述的是哪一场"有据可查。

    红法:让 `generate_report` 不把 `loader.selected_sessions` 传给渲染层(只在单元测试里
    直接调 `sources_disclosure` 时绿、走完整路径时什么都不显示)。
    """
    html = batch_report(gesture_sid=_THIS_SESSION)

    assert _THIS_SESSION in html, "同场时表头仍须点名本场 session id"
    assert "同一场会话" in html, "同场时表头没给出肯定说法"
    assert "并非同一场面试" not in html, "同场却被报成了不同场"
