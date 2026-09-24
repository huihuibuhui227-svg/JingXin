"""会话标识。id 由 voice 的 /interview/start 生成,显式下传三个模块。"""
from __future__ import annotations

import secrets
from datetime import datetime

NONE_SESSION = "NONE"          # 无 id 时的显式占位,报告侧整体排除(见 spec D7)


def new_session_id(now: datetime | None = None) -> str:
    """YYYYMMDD_HHMMSS_<4 位十六进制>:时间段给人看,随机段防撞。"""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{secrets.token_hex(2)}"
