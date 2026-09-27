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
