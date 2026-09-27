#!/usr/bin/env python3
"""盯 `__jx_logs`,只把**值得看的**吐到 stdout(每行 = 一条事件)。

给 Monitor 用:
    P=~/miniconda3/envs/jingxin/bin/python
    $P tools/jx_console_watch.py

**过滤规则**(照长期记忆 jingxin-console-hook-cdp 定的):
  · 非 log 级别(warn/error/onerror/unhandled) —— **永远报**;
  · log 级别里只有**收尾节点**报(录像 / 上传 / 留存 / 降级 / 停止);
  · 「提问窗口已上报」「发送第 N 帧」这类**逐题逐帧的刷屏一律不报** ——
    一场十几条到上千条,那不是盯守。

页面被刷新时 `__jx_logs` 会**从头开始**(新文档),这里检测到长度回退就重置游标。
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time

_spec = importlib.util.spec_from_file_location(
    "jx_hook", str(pathlib.Path(__file__).with_name("jx_console_hook.py")))
_hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hook)

# 收尾/关键节点 —— 只这些 log 级别的话值得打断人
TAIL_KEYS = ("录像", "上传", "留存", "降级", "停止", "收尾", "摄像头流已停")

POLL = 3.0


def main() -> int:
    seen = 0
    while True:
        try:
            page = _hook._pick("localhost:5173")
            cdp = _hook.Cdp(page["webSocketDebuggerUrl"])
            try:
                logs = cdp.eval("window.__jx_logs || []")
            finally:
                cdp.close()
        except Exception:  # noqa: BLE001 - 页面在导航/浏览器在重启都不该让盯守断掉
            time.sleep(POLL)
            continue

        if not isinstance(logs, list):
            time.sleep(POLL)
            continue

        if len(logs) < seen:                      # 文档换了,日志从头开始
            print(f"↻ 页面刷新过,日志重置(原 {seen} 条 → {len(logs)} 条)", flush=True)
            seen = 0

        for x in logs[seen:]:
            lvl, msg = x.get("lvl"), (x.get("msg") or "")
            if lvl != "log" or any(k in msg for k in TAIL_KEYS):
                print(f"[{lvl}] {msg[:300]}", flush=True)
        seen = len(logs)
        time.sleep(POLL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
