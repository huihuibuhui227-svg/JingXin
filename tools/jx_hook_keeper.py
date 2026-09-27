#!/usr/bin/env python3
"""把控制台钩子**焊死**在页面上 —— 刷新(F5)也不丢。

## 为什么需要这个进程

`Page.addScriptToEvaluateOnNewDocument` 的注册**随 CDP 连接断开而失效**
(2026-09-26 实测:注册完关掉连接 → 刷新后 `__jx_hook === false`)。
所以"调一次然后断开"的装法**根本撑不过一次刷新** —— 而使用者恰恰会按 F5。

这个进程**一直握着那条连接**,注册就一直在;页面每次新建文档(含 F5、含崩溃恢复)
都会自动重跑钩子,**中间没有空窗**。连接掉了就自动重连。

## 跑法

    P=~/miniconda3/envs/jingxin/bin/python
    $P tools/jx_hook_keeper.py &            # 默认守 localhost:5173 的应用页
    $P tools/jx_hook_keeper.py --url 127.0.0.1:5000   # 守别的页(自检用)

⚠️ 它**只对同一个标签页**有效。新开一个标签页 = 新 target,得重启它(它会自己认出来并切过去)。
"""
from __future__ import annotations
import os

import argparse
import importlib.util
import json
import pathlib
import sys
import time
import urllib.request

# CDP 端点 —— 默认是**本机 Windows Edge 的调试端口**。换机器/换端口用
# `JX_CDP` 环境变量覆盖。WSL 连 Windows 的 Edge 见 README 的「跨机器运行」。
CDP = os.environ.get("JX_CDP", "http://127.0.0.1:9222")

_spec = importlib.util.spec_from_file_location(
    "jx_hook", str(pathlib.Path(__file__).with_name("jx_console_hook.py")))
_hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hook)
HOOK = _hook.HOOK


def pick(url_match: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{CDP}/json/list", timeout=5) as r:
            pages = [t for t in json.load(r) if t.get("type") == "page"]
    except Exception:  # noqa: BLE001 - 浏览器没开就是没开,外层会退避重试
        return None
    hit = [p for p in pages if url_match in (p.get("url") or "")]
    # 带扩展名的页(报告 HTML)不是应用页 —— 别把钩子装到它上面
    app = [p for p in hit if not pathlib.PurePath(p["url"].split("?")[0]).suffix]
    return (app or hit or [None])[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="localhost:5173")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    def log(msg: str) -> None:
        if not args.quiet:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    while True:
        page = pick(args.url)
        if not page:
            log(f"没找到含 {args.url!r} 的标签页,5s 后重试")
            time.sleep(5)
            continue
        tid = page["id"]
        try:
            from websockets.sync.client import connect
            ws = connect(page["webSocketDebuggerUrl"], max_size=None, open_timeout=10)
            n = 0
            ws.send(json.dumps({"id": n + 1, "method": "Page.enable"}))
            n += 1
            while True:
                r = json.loads(ws.recv())
                if r.get("id") == n:
                    break
            ws.send(json.dumps({"id": n + 1, "method": "Page.addScriptToEvaluateOnNewDocument",
                                "params": {"source": HOOK}}))
            n += 1
            while True:
                r = json.loads(ws.recv())
                if r.get("id") == n:
                    ident = r.get("result", {}).get("identifier")
                    break
            # 当前这个文档也补一遍(F5 之前就已经打开着的那份)
            ws.send(json.dumps({"id": n + 1, "method": "Runtime.evaluate",
                                "params": {"expression": HOOK, "returnByValue": True}}))
            n += 1
            while True:
                r = json.loads(ws.recv())
                if r.get("id") == n:
                    break
            log(f"已焊上 target={tid[:12]}… script id={ident} —— 握着连接不放,刷新也不会丢")

            # ★ 关键:连接**不关**,只做保活;每 10s 复核一次目标还是不是同一个
            while True:
                time.sleep(10)
                cur = pick(args.url)
                if not cur or cur["id"] != tid:
                    log("目标变了(换标签页/关了),重连")
                    break
                try:
                    ws.recv(timeout=0.01)
                except TimeoutError:
                    pass
            ws.close()
        except Exception as e:  # noqa: BLE001 - 断线是常态(浏览器重启/导航),退避重连
            log(f"连接断了({type(e).__name__}: {e}),3s 后重连")
            time.sleep(3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
