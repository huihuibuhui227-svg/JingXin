# tests/test_artifact_shadowing.py
"""C1(最终全分支审查):`save_log()` 的产物不得落在报告侧的模态命名空间里。

**失效形态(修复前,已在临时目录与真仓库两处复现):**

`voice_interaction/pipeline/assessment_pipeline.py` 的 `InterviewAssessmentPipeline.save_log()`
每次被调用都写一个**新文件** `data/logs/interview/interview_emotion_log_<回答时刻>.csv`。
而 `report_frontend/data_loader.py` 递归扫 `data/logs`、按文件名正则

    ^(face|gesture|interview|research)_(.+?)_log_<YYYYMMDD>_<HHMMSS>[_<4 位十六进制>].csv$

把文件归到「模态」里,**每个模态各取文件名时间戳最新的那一个**。于是该产物:

1. 与真正的会话日志**同名形态**(都是 `interview_emotion_log_…`),落在同一棵被扫描的树里;
2. 时间戳是**回答**时刻,必然晚于会话开始时铸进 M1 文件名的那个 → **每次都被选中**;
3. 经 API 它**永远只有表头** —— prosody 列只有 `examples/` 那两个脚本会填
   (`add_answer()` 走的是 `add_qa_pair(question, answer)`,不传 prosody)。

装载器随即把这个空表丢掉 → `voice_interview` 一个模态都到不了特征引擎 →
「连接词密度」渲染成「未采集到对应数据」:**验收门的核心判据因此不可能通过**。

**两条测试钉住两头:** 产出方(它的文件名结构上就该被正则拒掉)+ 装载方
(一场真实会话的目录形状里,会话日志必须是胜者,且它的值要一路走到特征与证据门)。
"""

import csv
import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from report_frontend.data_loader import LogDataLoader
from report_frontend.feature_engine import PsychologicalFeatureEngine
from report_frontend.research_mapper import ResearchCapabilityMapper

_NOTE_TIMESTAMP = "%Y%m%d_%H%M%S"


def _session_id_older_than_now(hours: int = 1) -> str:
    """铸一个「会话开始于 N 小时前」的 id。

    必须早于「现在」:产物的文件名时间戳取自**回答**时刻(= 现在),而会话 id 里的
    时间戳取自会话开始 —— 前者晚于后者正是本 bug 的成因。写死一个字面时间会让
    测试依赖跑测试的钟点(跨过那个时刻就反过来),所以按相对时间铸。
    """
    stamp = (datetime.now() - timedelta(hours=hours)).strftime(_NOTE_TIMESTAMP)
    return f"{stamp}_9f3c"


_TS_RE = re.compile(r"(\d{8})_(\d{6})")


def _stamp_of(name: str) -> int:
    """取文件名里 `YYYYMMDD_HHMMSS` 那一段 —— 装载器排序用的就是这个整数。

    不能用 `split("_")[-2:]`:M1 会话日志的名字尾部还多一段四位十六进制
    (`…_log_<YYYYMMDD>_<HHMMSS>_<hex>.csv`),会把十六进制当成秒接上去。
    """
    m = _TS_RE.search(Path(name).stem)
    assert m, f"名字里没有时间戳:{name}"
    return int(m.group(1) + m.group(2))


def _build_real_session_log(log_dir: Path, session_id: str, densities: list) -> Path:
    """用 VoiceLogger 落一份真的会话日志(表头、文件名、首列都由生产代码决定)。"""
    from voice_interaction.utils.logger import VoiceLogger

    lg = VoiceLogger(log_type="interview", log_dir=str(log_dir), session_id=session_id)
    for i, d in enumerate(densities):
        lg.log_prosody({}, question_index=i, emotion="", feedback="",
                       connective_density=d, connective_density_std=0.0, n_rows=1)
    return lg.csv_file


def _write_csv(path: Path, fieldnames: list, rows: list) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _build_logs_tree(log_dir: Path, session_id: str, densities: list) -> Path:
    """一场真实会话的 `data/logs/` 形状:会话日志 + NONE 桶 + 旧形态文件。

    产物由 `produced_artifact` fixture 真跑产出方写进来(不手搓名字)。
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    _build_real_session_log(log_dir, session_id, densities)

    # 无 id 会话桶(报告侧必须继续看不见它:它跨天累积,会把不同场次的行混起来)
    _write_csv(log_dir / "interview_emotion_log_NONE_20260924_170000.csv",
               ["session_id", "timestamp", "connective_density"],
               [{"session_id": "NONE", "timestamp": "2026-09-24T17:00:00",
                 "connective_density": 99.0}])

    # 旧形态(纯时间戳结尾)的面部日志 —— 历史日志必须仍能被读到
    _write_csv(log_dir / "face_au_log_20200101_090000.csv",
               ["marker", "value"], [{"marker": "legacy_face", "value": 1.0}])

    return log_dir


@pytest.fixture
def log_dir(tmp_path) -> Path:
    """被扫描的那棵树,**同时**是产物的落盘根目录。

    两者在生产里是同一个 `data/logs`(`ASSESSMENT_LOG_ROOT` 与
    `LogDataLoader.log_dir` 都指它)。若测试把产物写到被扫描树**之外**,
    「产物顶掉会话日志」这件事根本不会发生 —— 本文件第一次写出来时就是这样:
    产物落在 `tmp_path/interview/`、会话日志在 `tmp_path/logs/`,于是
    test_session_log_wins_… 在"把产物名改回修复前形态"的变异下**照旧全绿**。
    """
    d = tmp_path / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def produced_artifact(log_dir, monkeypatch):
    """把产物落盘根目录指到那棵被扫描的树(生产里是 `<repo>/data/logs`),再真跑产出方。"""
    from voice_interaction.pipeline import assessment_pipeline

    monkeypatch.setattr(assessment_pipeline, "ASSESSMENT_LOG_ROOT", log_dir)
    return assessment_pipeline.InterviewAssessmentPipeline


def test_save_log_output_is_invisible_to_the_modality_selector(log_dir, produced_artifact):
    """产出方:产物写出的文件名,报告侧选取正则必须匹配不上(**结构上**排除)。

    不是"名字里没有 NONE"那种够不到的判断 —— 这里断言的是正则本身对这个名字返回 None。
    红在:把文件名改回 `interview_emotion_log_<时间戳>.csv`(修复前的形态)时,
    `match()` 返回一个 match 对象 → 本条立刻红并点名后果(见报告里的变异实验)。
    """
    produced = Path(produced_artifact().save_log())

    assert produced.parent.is_relative_to(log_dir), (
        f"产物没落在被扫描的那棵树下({log_dir}),本测试会退化成恒真 —— 这正是第一版踩过的坑"
    )
    pattern = LogDataLoader(str(log_dir)).file_pattern
    assert pattern.match(produced.name) is None, (
        f"save_log 的产物落在报告侧的模态命名空间里:{produced.name} —— "
        f"它的时间戳是回答时刻、必然晚于会话日志,会在报告里顶掉真正的会话日志"
    )


def test_session_log_wins_and_its_values_reach_the_feature_and_the_gate(log_dir, produced_artifact):
    """装载方:目录里同时有会话日志、**更新的**产物、NONE 与旧形态文件时,会话日志必须是胜者。

    红法(修复前,见报告里的变异实验):产物名字是 `interview_emotion_log_<回答时刻>.csv`
    → 时间戳比会话日志新 → 装载器选中它 → 空表被丢 → `data` 里没有 `voice_interview`
    → 第一条断言即红,报「会话日志没被选中」。修复后:会话日志中标 → 密度均值/样本量
    一路到达特征引擎 → 证据门 G2/G3 全过 → 「连接词密度」进证据链(验收门核心判据)。
    """
    session_id = _session_id_older_than_now(hours=1)
    densities = [3.0, 4.5, 2.0, 6.0, 1.5, 5.5, 3.5, 7.0]

    log_dir = _build_logs_tree(log_dir, session_id, densities)
    written = Path(produced_artifact().save_log())

    # 前提自检 —— 三件事必须成立,否则本测试红不到点子上(或退化成恒真):
    # ① 产物真落盘了;② 它确实比会话日志**新**(它就是靠这点赢下选取的);
    # ③ 它在 API 路径上只有表头(空表会被装载器丢掉,这正是"密度消失"的机制)。
    assert written.exists(), "产出方没写出产物,本测试的前提不成立"
    assert _stamp_of(written.name) > _stamp_of(f"interview_emotion_log_{session_id}.csv"), (
        "产物不比会话日志新 —— 修复前的红不是本 bug 造成的,继续下去会变成空断言"
    )
    with written.open(newline="", encoding="utf-8") as f:
        assert list(csv.DictReader(f)) == [], "产物带上了数据行,本测试的前提(空表被丢)变了"

    data = LogDataLoader(str(log_dir)).get_fused_latest_data()

    assert "voice_interview" in data, (
        f"会话日志没被选中 —— 目录里的产物顶掉了它。装入的模态:{sorted(data)};"
        f"产物的名字是 {written.name}"
    )
    voice = data["voice_interview"]
    assert len(voice) == len(densities), (
        f"选中的不是会话日志({len(voice)} 行,应有 {len(densities)} 行)"
    )

    feats = PsychologicalFeatureEngine(data).extract_all_features()
    vi = feats.get("voice_interview", {})
    expected_mean = sum(densities) / len(densities)
    key = next((k for k in vi if k.endswith("connective_density_mean")), None)
    assert key is not None, f"密度没进特征引擎:{sorted(vi)}"
    assert vi[key] == pytest.approx(expected_mean), (
        f"进特征引擎的值不是会话日志里的那列:{vi[key]} != {expected_mean}"
    )
    assert vi["_n_rows"] == float(len(densities))

    dim = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]["logical_thinking"]
    names = [ev["human_name"] for ev in dim["evidence_chain"]]
    assert "连接词密度" in names, (
        f"「连接词密度」没过证据门出分 —— 这正是验收门的核心判据。"
        f"缺口:{dim['evidence_gaps']}"
    )


def test_renamed_artifact_is_still_findable_by_its_consumer(log_dir, produced_artifact):
    """改名不能把消费者一起孤儿化:`visualize.py` 的找文件函数必须仍认得出产物。

    `voice_interaction/utils/visualize.py` 按文件名词根找日志(原先匹配
    `interview_emotion_log_`),是那份产物的**真实**消费者 —— 所以改名时它一起改了。
    红法:把 `_is_kind` 里的 `assessment_note_` 分支删掉(或改回只认旧词根)。
    """
    from voice_interaction.pipeline.assessment_pipeline import ResearchAssessmentPipeline
    from voice_interaction.utils.visualize import _is_kind

    produced = Path(produced_artifact().save_log())
    assert _is_kind(produced, "interview"), f"改名后 visualizer 找不到产物了:{produced}"
    assert not _is_kind(produced, "research"), "面试的产物被当成了科研的"

    research_note = Path(ResearchAssessmentPipeline().save_log())
    assert _is_kind(research_note, "research"), "改名的产物按目录分不出科研侧"
    assert not _is_kind(research_note, "interview")

    # 会话日志(M1 形态,文件名带 session_id)必须继续被认出来
    sid = _session_id_older_than_now(hours=2)
    assert _is_kind(log_dir / f"interview_emotion_log_{sid}.csv", "interview")
    assert _is_kind(log_dir / "interview_emotion_log_NONE_x.csv", "interview")
