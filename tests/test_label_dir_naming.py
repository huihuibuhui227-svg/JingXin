# tests/test_label_dir_naming.py
"""录制目录名带上本场标注:`<标签>__<sid>`(使用者 2026-09-27 裁定)。

**为什么 sid 必须留在目录名尾部**:`session_id` 同时是 `data/logs/` 里三份日志的
文件名,而报告侧的 `data_loader` 就是按 `_log_{YYYYMMDD}_{HHMMSS}[_{4位hex}]` 那个
正则找日志的。目录名可以改,但必须**还能从 sid 推出来** —— 否则 `root()/<sid>`
那四个调用点(media_retention / transcript_store / session_meta / 重放工具)就
集体失效,而失效的形态是**素材写不进去**(静默丢),正是本仓最贵的那一类。

⚠️ 历史素材(`root()/<sid>`,本轮之前那 56 场)**不动也不改名**:
`resolve_recording_dir` 先看旧形态在不在,在就用它。
"""
import importlib

import pytest

media_retention = importlib.import_module("media_retention")
session_meta = importlib.import_module("session_meta")
transcript_store = importlib.import_module("voice_interaction.asr.transcript_store")

SID = "20260927_205621_3442"
LABEL = "20260927_205621-04-赵六-2021004-物理学院"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    session_meta._reset_for_tests()
    return tmp_path


def _dir_name(label: str, sid: str = SID) -> str:
    return media_retention.label_dir_name(label, sid)


# ── 解析:三种状态各走哪条路 ────────────────────────────────────────────────

def test_legacy_dir_still_wins_when_it_exists(_isolated):
    """历史素材(目录名就是 sid)必须照旧解析到它自己 —— 那 56 场不改名。"""
    (_isolated / SID).mkdir()
    assert media_retention.resolve_recording_dir(SID) == _isolated / SID


def test_renamed_dir_is_found_by_its_sid_suffix(_isolated):
    """改过名之后(`<标签>__<sid>`)仍要从 sid 找得到 —— 这是整件事的支点。

    红法:把 `resolve_recording_dir` 里的后缀匹配去掉 → 本测试红(会返回
    `root()/<sid>`,而那个路径不存在 ⟹ 素材会写到一个**新的空目录**里去)。
    """
    (_isolated / _dir_name(LABEL)).mkdir()
    assert media_retention.resolve_recording_dir(SID) == _isolated / _dir_name(LABEL)


def test_no_dir_yet_returns_the_legacy_path_and_creates_it(_isolated):
    """一场还没落任何东西时:按旧形态给路径,并且**建出来**(原有契约)。"""
    d = media_retention.recording_dir(SID)
    assert d == _isolated / SID and d.is_dir()


def test_sibling_that_merely_contains_the_sid_does_not_match(_isolated):
    """防误配:后缀必须是 `__<sid>` **结尾**,`foo__<sid>_extra` 不算。

    红法:把判据从 `endswith("__"+sid)` 改成 `sid in name` → 本测试红。
    """
    (_isolated / f"x__{SID}_extra").mkdir()
    assert media_retention.resolve_recording_dir(SID) == _isolated / SID


# ── 改名:标注存下就改,改标签再改一次 ──────────────────────────────────────

def test_saving_a_label_renames_the_session_dir(_isolated):
    session_meta.upsert_label(SID, serial="04", name="赵六",
                              student_id="2021004", department="物理学院")
    assert (_isolated / _dir_name(LABEL)).is_dir()
    assert not (_isolated / SID).exists(), "旧名不该留着一个空壳"
    assert (_isolated / _dir_name(LABEL) / "label.json").exists()


def test_re_editing_the_label_renames_again(_isolated):
    """「改标签」= 再改一次名,而且**只留一个**目录(不留上一版的名字)。"""
    session_meta.upsert_label(SID, serial="04", name="赵六",
                              student_id="2021004", department="物理学院")
    session_meta.upsert_label(SID, serial="04", name="赵六改",
                              student_id="2021004", department="物理学院")
    dirs = [p.name for p in _isolated.iterdir() if p.is_dir()]
    assert len(dirs) == 1 and "赵六改" in dirs[0]
    assert session_meta.read_label(SID)["name"] == "赵六改"


# ── 真正的失败形态:改名之后素材还写不写得进去 ──────────────────────────────

def test_retention_writes_into_the_renamed_dir(_isolated):
    """★ 这条才是整件事的意义所在:改名**之后**帧/录像还得落进**同一个**目录。

    红法:去掉 `resolve_recording_dir` 的后缀匹配 → `media_dir` 会在
    `root()/<sid>` 下**新建一个空目录**,帧全写进去,而报告/重放去看的是另一个
    目录 ⟹ 素材"看着存了、其实没在那一场里"。这就是本项目最贵的那类失效。
    """
    session_meta.upsert_label(SID, serial="04", name="赵六",
                              student_id="2021004", department="物理学院")
    d = media_retention.media_dir(SID, "face")
    assert d == _isolated / _dir_name(LABEL) / "media" / "face"
    assert d.is_dir()


def test_transcript_store_follows_the_rename_too(_isolated):
    """`session.json` / `transcript.json` 也归同一个目录 —— 否则一场会被劈成两半。"""
    session_meta.upsert_label(SID, serial="04", name="赵六",
                              student_id="2021004", department="物理学院")
    assert transcript_store.recording_dir(SID) == _isolated / _dir_name(LABEL)


def test_transcript_store_still_honours_an_explicit_root(_isolated, tmp_path):
    """显式给了 root 的调用方(测试/工具)不受目录名规则影响,按老样子走。"""
    other = tmp_path / "explicit"
    other.mkdir()
    assert transcript_store.recording_dir(SID, root=other) == other / SID


# ── 守卫:标签进目录名 = 一个新出现的路径注入面 ────────────────────────────

@pytest.mark.parametrize("bad", ["../escape", "a/b", "a\\b", "..", ".", ".hidden", "a\x00b", "a\nb"])
def test_label_that_is_not_a_safe_path_component_is_rejected(bad):
    """⚠️ 标签是**客户端可控**的,而它会被拼成目录名 —— 与 `validate_session_id`
    同一类风险(`../../jingxin/x` 能在录制根之外建目录)。放它进来等于把刚焊好的
    那道守卫撬开一个口子。

    红法:去掉 `label_dir_name` 里那次 `_safe_path_component` → 本测试红。
    """
    with pytest.raises(ValueError):
        media_retention.label_dir_name(bad, SID)


def test_rejected_label_leaves_nothing_outside_the_root(_isolated):
    """真去调一次:被拒时**盘上不该多出任何东西**(尤其根外面)。"""
    with pytest.raises(ValueError):
        session_meta.upsert_label(SID, serial="..", name="../../逃逸",
                                  student_id="", department="")
    assert list(_isolated.iterdir()) == []


def test_overlong_dir_name_is_rejected(_isolated):
    """目录名有文件系统上限(255 字节,中文 3 字节/字)。超了**当请求的毛病拒掉**,
    而不是悄悄不改名 —— 使用者要的就是盘上那个名字,悄悄不改等于说话不算数。"""
    with pytest.raises(ValueError):
        session_meta.upsert_label(SID, serial="01", name="张" * 100,
                                  student_id="", department="")
