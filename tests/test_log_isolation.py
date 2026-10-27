# tests/test_log_isolation.py
"""钉子:**跑测试不许往仓库里写日志**。

背景(`docs/下一步.md` §11.5 第 1 条):每跑一次全套件就往仓库 `data/logs/` 下
留 `assessment_note_<ts>.csv`。`tests/conftest.py` 正是为这类泄漏写的,却只挪了
三个 logger 的 `LOGS_DIR` 与录音落点,**没覆盖 `ASSESSMENT_LOG_ROOT`**。

本文件是那条例外的**钉子** —— 它不检查某个具体测试的行为,而是直接断言
「评估流水线的默认落点不在仓库里」。任何人再往 `assessment_pipeline` 里加一个
以 `ASSESSMENT_LOG_ROOT` 为根的写出,只要 conftest 没兜住,这条就红。
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _is_inside_repo(p: Path) -> bool:
    r = p.resolve()
    return r == REPO or REPO in r.parents


def test_assessment_log_root_is_redirected_out_of_repo():
    from voice_interaction.pipeline import assessment_pipeline

    root = Path(assessment_pipeline.ASSESSMENT_LOG_ROOT)
    assert not _is_inside_repo(root), (
        f"ASSESSMENT_LOG_ROOT 指向仓库内({root})⟹ 跑测试会往仓库里写 assessment_note_*.csv。"
        "修法:`tests/conftest.py` 里像 `LOGS_DIR` 那样把它挪到临时目录。"
    )


def test_logger_defaults_are_redirected_out_of_repo():
    """与上一条同族的既有钉子(三个 logger 的 `LOGS_DIR`),一起守住。"""
    import voice_interaction.utils.logger as vlog

    assert not _is_inside_repo(Path(vlog.LOGS_DIR)), f"voice LOGS_DIR 在仓库内:{vlog.LOGS_DIR}"


def test_series_dir_is_redirected_out_of_repo():
    """★ 语音帧序列缓存(`data/l0_series/`)也必须被兜住。

    2026-10-24 实测泄漏:一条用真 `save_series` 的测试没设 `JINGXIN_SERIES_DIR`,
    往仓库 `data/l0_series/` 写进了 `<假场次>.npz`。它被 gitignore,所以
    `git status` 看不出来 —— 只能靠这条钉子。

    红法:删掉 `tests/conftest.py` 里 `JINGXIN_SERIES_DIR` 那段 ⟹ 本条红。
    """
    from report_frontend import voice_series

    d = Path(voice_series.series_dir())
    assert not _is_inside_repo(d), (
        f"series_dir() 指向仓库内({d})⟹ 跑测试会往 data/l0_series/ 里写缓存。"
        "修法:`tests/conftest.py` 里像 `LOGS_DIR` 那样把它挪到临时目录。"
    )


def test_recordings_dir_is_redirected_out_of_repo():
    """★ 录制根也必须被兜住 —— 而且必须是**强制**接管,不是 `setdefault`。

    ## 2026-10-27 实测:这条隔离有个反向的坑

    原来 `tests/conftest.py` 写的是
    `os.environ.setdefault("JINGXIN_RECORDINGS_DIR", <临时目录>)`。它的本意是
    「某个用例自己设了就让它优先」,但 `setdefault` 分不清**谁**设的 ——
    调用方从 shell 里 `export` 的真录制根同样算「已经设了」,于是:

        JINGXIN_RECORDINGS_DIR=~/JingXin/recordings pytest -q   ⟹ 整个套件读真素材

    实测症状:`test_html_report_uses_the_l1_engine` 三条全 ERROR
    (`n_valid(8) > n_total(4) —— 把证据说多了`),而单跑那三条是绿的。
    看上去像报告生成器坏了,其实是**测试跑错了对象**。

    更阴的是它反过来咬人:本项目反复强调「手工跑读素材的探针前先 export 这个变量」,
    同一条习惯照搬到 `pytest` 上恰好把隔离打穿了。

    ## 为什么这里查源码而不只是查取值

    只断言「当前取值不在仓库/不是真素材」的话,在**裸跑**时 `setdefault` 版本也是绿的
    —— 那条断言没有区分力。要钉住的是「外部 export 压不过 conftest」这个性质,
    而它在同一个进程里已经发生了,只能看 conftest 用的哪个写法。
    (与本仓库既有的「三份副本源码级契约」测试同一手法。)

    红法:把 `tests/conftest.py` 里那行改回 `os.environ.setdefault(...)` ⟹ 本条红。
    """
    import os

    v = os.environ.get("JINGXIN_RECORDINGS_DIR")
    assert v, "conftest 没有设 JINGXIN_RECORDINGS_DIR ⟹ 测试会读使用者真实素材"
    assert "jingxin-test-recordings-" in v, (
        f"录制根没被 conftest 接管({v})⟹ 测试可能读到真实素材。"
    )

    src = (Path(__file__).parent / "conftest.py").read_text(encoding="utf-8")
    line = next((ln for ln in src.splitlines()
                 if "JINGXIN_RECORDINGS_DIR" in ln and not ln.lstrip().startswith("#")), "")
    assert line, "conftest 里找不到设置 JINGXIN_RECORDINGS_DIR 的那一行"
    assert "setdefault" not in line, (
        "conftest 用 setdefault 设 JINGXIN_RECORDINGS_DIR ⟹ 外部 export 的真录制根"
        "会压过隔离,整个套件静默读真素材(2026-10-27 实测)。改成无条件赋值。"
    )
