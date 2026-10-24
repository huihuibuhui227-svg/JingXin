# tests/test_assessment_note_is_not_an_empty_artifact.py
"""评估产物 `assessment_note_*.csv` 不许**只有表头**。

## 查清的事实(2026-10-24 夜,可复算)

盘上 `data/logs/{interview,research}/assessment_note_*.csv` **515 份,数据行 0 行**
(每一份都只有表头)。这不是"孤儿文件"那么简单 —— 它是**结构性永远为空**:

* `save_log()` 只在 `qa.prosody_features and qa.prosody_analysis` **两者都有**时写行;
* 而 `qa_pairs` 唯一的活路径写入者 `add_answer()` 调的是
  `add_qa_pair(question, answer)` —— **一个 prosody 都不传**
  (`voice_interaction/pipeline/assessment_pipeline.py`)。
* 于是 `/interview/answer`、`/interview/answer_audio`、`/research/answer`、
  `/research/answer_audio` 每答一题就**多一个表头文件**。

`docs/开发者文档.md` §12.2 把它记成「认不出属于哪一场 ⟹ purge 时留下孤儿」。
那个描述**低估了**:孤儿的前提是"有内容",而它从来没有过内容。真正的害处是
① 每答一题往 `data/logs` 里堆一个空文件(实测 515 份);
② `voice_interaction/utils/visualize.py` 会把这种空表当成一份日志读进来;
③ 文件名不带 sid ⟹ `session_purge` 删一场时它留了下来。

## 这个文件守什么

**一行数据都没有时,不写文件,返回空串。**

返回空串而不是"写一个只有表头的文件",是本仓既有口径:没产出不许报成功
(`report_generator.generate_report` 失败时正是 `return ""`,面板据此判红)。
"""
import csv
from types import SimpleNamespace

import pytest

from voice_interaction.pipeline import assessment_pipeline as ap


def _fake_features(**kw):
    """`ProsodyFeatures` 的替身 —— 只要 `save_log` 读的那几个属性。"""
    fields = dict(pitch_mean=180.0, pitch_std=22.0, pitch_trend=0.1,
                  pitch_direction="stable", energy_mean=0.04, energy_std=0.01,
                  speech_ratio=0.8, duration_sec=12.0, pause_duration_mean=0.4,
                  pause_duration_max=1.2, pause_frequency=3.0)
    fields.update(kw)
    return SimpleNamespace(**fields)


def _fake_analysis(**kw):
    fields = dict(is_valid=True, feedback="")
    fields.update(kw)
    return SimpleNamespace(**fields)


@pytest.fixture
def note_root(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "ASSESSMENT_LOG_ROOT", tmp_path)
    return tmp_path


@pytest.mark.parametrize("kind,cls_name", [
    ("interview", "InterviewAssessmentPipeline"),
    ("research", "ResearchAssessmentPipeline"),
])
def test_no_data_rows_means_no_file_at_all(note_root, kind, cls_name):
    """★ 红法:把 `save_log` 里那句 `if not rows: return ""` 去掉。

    去掉之后它照旧写一个只有表头的文件 —— 而那正是盘上 515 份文件的来源。
    断言写成"目录里一个 csv 都没有",而不是"文件是空的":后者两种实现都满足,
    分不出修没修。
    """
    p = getattr(ap, cls_name)()
    p.add_answer("我先讲一下我的项目经历")      # 与 /interview/answer 同一调法:不传 prosody

    out = p.save_log()

    assert out == "", f"一行数据都没有却报了落盘路径:{out!r}"
    produced = sorted((note_root / kind).glob("*.csv")) if (note_root / kind).is_dir() else []
    assert produced == [], f"没有数据却写出了文件:{[x.name for x in produced]}"


@pytest.mark.parametrize("kind,cls_name", [
    ("interview", "InterviewAssessmentPipeline"),
    ("research", "ResearchAssessmentPipeline"),
])
def test_rows_with_prosody_are_still_written(note_root, kind, cls_name):
    """删空文件 ≠ 把功能删掉:真有 prosody 时,一行都不能少。

    红法:把 `save_log` 改成一律 `return ""`(修过头)。
    """
    p = getattr(ap, cls_name)()
    p.add_qa_pair("请自我介绍", "我叫……",
                  prosody_features=_fake_features(),
                  prosody_analysis=_fake_analysis())

    out = p.save_log()

    assert out, "有数据却报空路径"
    with open(out, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, f"有 1 段 prosody,却写了 {len(rows)} 行"
    assert float(rows[0]["pitch_mean"]) == pytest.approx(180.0)
    assert rows[0]["question_index"] == "0"


def test_the_live_answer_path_never_produces_a_note(note_root):
    """把整条活路径串起来再断言一次 —— 上面两条各自单独成立,不代表合起来成立。

    这条走的是 `/interview/start` → 三次 `/interview/answer` 的形状
    (`reset()` + `add_answer()` 三次),然后才调 `save_log()`。
    红法:把 `add_answer` 改成会传 prosody —— 那时它开始有行,本测试会红在
    "居然写了文件"上,提示先确认那是不是有意的。
    """
    p = ap.InterviewAssessmentPipeline()
    p.reset()
    for ans in ("第一题的回答", "第二题的回答", "第三题的回答"):
        p.add_answer(ans)

    assert p.save_log() == ""
    assert not (note_root / "interview").exists() or not list((note_root / "interview").glob("*.csv"))
