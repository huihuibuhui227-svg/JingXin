#!/usr/bin/env python3
"""JingXin 录制用的控制台钩子 —— 装 / 读 / 清。

为什么不用 web-access 代理的 /eval:那样装的钩子**整页刷新(F5)就没了**,
而 §8.4 第 3 条把「不要刷新」当成必须遵守的纪律。用原始 CDP 的
`Page.addScriptToEvaluateOnNewDocument` 装,钩子在**每个新文档开头**自动重跑,
**F5 之后依然在** —— 少一条要人记的纪律。

用法(在 WSL 里跑,连的是 Windows Edge 的 9222):
    P=~/miniconda3/envs/jingxin/bin/python
    $P tools/jx_console_hook.py install          # 装(并打印 script id)
    $P tools/jx_console_hook.py status           # 钩子在不在、攒了多少条
    $P tools/jx_console_hook.py dump             # 打出全部日志(JSON)
    $P tools/jx_console_hook.py dump --errors    # 只打 warn/error/onerror/unhandled
    $P tools/jx_console_hook.py uninstall <id>   # 卸掉持久钩子
"""
from __future__ import annotations

import argparse
import json
import os.path
import sys
import urllib.request

# CDP 端点 —— 默认是**本机 Windows Edge 的调试端口**。换机器/换端口用
# `JX_CDP` 环境变量覆盖。WSL 连 Windows 的 Edge 见 README 的「跨机器运行」。
CDP = os.environ.get("JX_CDP", "http://127.0.0.1:9222")
URL_MATCH = "localhost:5173"

# ⚠️ 与长期记忆 jingxin-console-hook-cdp 里的片段保持一致 —— 它被真会话验证过。
HOOK = r"""
(() => {
  if (window.__jx_hook) return "已装过";
  window.__jx_logs = [];
  for (const lvl of ["log","warn","error","info"]) {
    const o = console[lvl].bind(console);
    console[lvl] = (...a) => {
      try {
        window.__jx_logs.push({t: Date.now(), lvl, msg: a.map(x => {
          try { return typeof x === "string" ? x : JSON.stringify(x); }
          catch (e) { return String(x); }
        }).join(" ")});
      } catch (e) {}
      o(...a);
    };
  }
  window.addEventListener("error", e =>
    window.__jx_logs.push({t: Date.now(), lvl: "onerror", msg: String(e.message)}));
  window.addEventListener("unhandledrejection", e =>
    window.__jx_logs.push({t: Date.now(), lvl: "unhandled", msg: String(e.reason)}));
  window.__jx_hook = true;
  return "hooked";
})()
"""

NOISY = {"warn", "error", "onerror", "unhandled"}


def _pages() -> list[dict]:
    with urllib.request.urlopen(f"{CDP}/json/list", timeout=5) as r:
        return [t for t in json.load(r) if t.get("type") == "page"]


def _pick(url_match: str) -> dict:
    pages = _pages()
    hit = [p for p in pages if url_match in (p.get("url") or "")]
    # ⚠️ 报告 HTML 也是 vite dev server 发的,URL 形如
    #    `localhost:5173/output/Research_..._Report_*.html` —— 只按子串匹配会把它
    #    和真正的应用页一起选中,而 `_pick` 取第一个 ⟹ **注入/清空可能打到报告页上**。
    #    带文件扩展名的页面(报告、PDF)永远不是应用页,先排除。
    app = [p for p in hit if not os.path.splitext(p["url"].split("?")[0])[1]]
    if app:
        hit = app
    if not hit:
        print(f"✗ 没有 URL 含 {url_match!r} 的标签页。当前有:", file=sys.stderr)
        for p in pages:
            print(f"    {p['id']}  {p['url'][:90]}", file=sys.stderr)
        sys.exit(2)
    if len(hit) > 1:
        print(f"⚠️  有 {len(hit)} 个匹配的标签页,用第一个:{hit[0]['url'][:80]}", file=sys.stderr)
    return hit[0]


class Cdp:
    def __init__(self, ws_url: str):
        from websockets.sync.client import connect
        self._ws = connect(ws_url, max_size=None, open_timeout=10)
        self._id = 0

    def call(self, method: str, **params):
        self._id += 1
        self._ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            msg = json.loads(self._ws.recv())
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def eval(self, expr: str):
        r = self.call("Runtime.evaluate",
                      expression=expr, returnByValue=True, awaitPromise=True)
        if r.get("exceptionDetails"):
            raise RuntimeError(f"JS 抛错: {r['exceptionDetails']}")
        return r["result"].get("value")

    def close(self):
        self._ws.close()


def cmd_install(args) -> int:
    page = _pick(args.url)
    cdp = Cdp(page["webSocketDebuggerUrl"])
    try:
        # ⚠️ `Page.enable` 是**必须的**,少了它下面那句会返回一个 script id
        #    然后**什么都不做** —— 刷新后钩子照样没有(2026-09-26 实测:
        #    不 enable → 刷新后 __jx_hook=False;enable 后 → True)。
        #    "返回成功但没有效果" 是这个项目反复踩的形态,所以这里显式写上。
        cdp.call("Page.enable")
        # ① 持久:每个新文档(含 F5 之后)自动装
        r = cdp.call("Page.addScriptToEvaluateOnNewDocument", source=HOOK)
        sid = r.get("identifier")
        # ② 当前这个文档立即生效
        now = cdp.eval(HOOK)
        alive = cdp.eval("typeof window.__jx_logs === 'object' && !!window.__jx_hook")
        print(f"标签页   : {page['url']}")
        print(f"当前文档 : {now}  (__jx_hook={alive})")
        print(f"持久钩子 : script id = {sid}   ← F5 之后也在;要撤用 uninstall {sid}")
        return 0
    finally:
        cdp.close()


def cmd_status(args) -> int:
    page = _pick(args.url)
    cdp = Cdp(page["webSocketDebuggerUrl"])
    try:
        n = cdp.eval("window.__jx_hook ? window.__jx_logs.length : null")
        print(f"标签页 : {page['url']}")
        if n is None:
            print("钩子   : ✗ 没装(或刚被刷新过而持久钩子也没装)")
            return 1
        print(f"钩子   : ✓ 已装,攒了 {n} 条")
        return 0
    finally:
        cdp.close()


def cmd_dump(args) -> int:
    page = _pick(args.url)
    cdp = Cdp(page["webSocketDebuggerUrl"])
    try:
        logs = cdp.eval("window.__jx_logs || []")
    finally:
        cdp.close()
    if not isinstance(logs, list):
        print("✗ 读不到日志(钩子没装?)", file=sys.stderr)
        return 1
    if args.errors:
        logs = [x for x in logs if x.get("lvl") in NOISY]
    if args.json:
        print(json.dumps(logs, ensure_ascii=False, indent=2))
    else:
        if not logs:
            print("(空)")
        for x in logs:
            print(f"[{x.get('lvl'):9s}] {x.get('msg')}")
    print(f"—— 共 {len(logs)} 条 ——", file=sys.stderr)
    return 0


def cmd_clear(args) -> int:
    page = _pick(args.url)
    cdp = Cdp(page["webSocketDebuggerUrl"])
    try:
        n = cdp.eval("(window.__jx_logs || []).length")
        cdp.eval("window.__jx_logs = []")
        print(f"✓ 清掉 {n} 条 —— 之后攒的都属于这一场")
        return 0
    finally:
        cdp.close()


def cmd_uninstall(args) -> int:
    page = _pick(args.url)
    cdp = Cdp(page["webSocketDebuggerUrl"])
    try:
        cdp.call("Page.removeScriptToEvaluateOnNewDocument", identifier=args.id)
        print(f"✓ 已卸掉持久钩子 {args.id}(当前文档里那份还在,刷新才消失)")
        return 0
    finally:
        cdp.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=URL_MATCH, help=f"标签页 URL 子串(默认 {URL_MATCH})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("install", help="装钩子(持久 + 当前文档)")
    sub.add_parser("status", help="钩子在不在")
    sub.add_parser("clear", help="清空已攒的日志(开录前跑一次)")
    d = sub.add_parser("dump", help="打出日志")
    d.add_argument("--errors", action="store_true", help="只看 warn/error/onerror/unhandled")
    d.add_argument("--json", action="store_true")
    u = sub.add_parser("uninstall", help="卸掉持久钩子")
    u.add_argument("id", help="install 打印的 script id")
    args = ap.parse_args()
    return {"install": cmd_install, "status": cmd_status, "clear": cmd_clear,
            "dump": cmd_dump, "uninstall": cmd_uninstall}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
