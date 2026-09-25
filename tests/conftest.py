# tests/conftest.py
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── 把 logger 的**默认**日志目录挪出仓库(会话级,先于任何测试模块 import 生效) ──────
#
# 为什么非在这里不可:导入 `voice_interaction.api.app` 时,模块级就会构造 VoiceLogger,
# 而 `VoiceLogger.__init__` **立刻**写下 CSV 表头(`utils/logger.py` 的 `_write_csv_header`)——
# 不拦的话**每跑一次 pytest 就往仓库 `data/logs/` 里留下一个只有表头的
# `*_log_NONE_<时间戳>.csv`**(2026-09-25 实测:一次两个,面试一个、科研一个;
# 盘上 9/24 那批就是这么攒出来的,先于本轮改动)。这些空文件**不进任何聚合**
# (0 行),但让"仓库目录被测试写脏",也会进报告头的 NONE 桶计数范围。
#
# 单测自己的 `monkeypatch.setattr(logger_module, "LOGS_DIR", …)` 照旧优先(它们在这个
# 会话级值之上再覆盖);这里只是把**构造期**那一次落点挪走。
_LOGS_TMP = Path(tempfile.mkdtemp(prefix="jingxin-test-logs-"))

import voice_interaction.utils.logger as _voice_logger_module  # noqa: E402

_voice_logger_module.LOGS_DIR = str(_LOGS_TMP)
