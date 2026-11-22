"""FunASR 的薄适配器:把客户端结果规整成落盘与算特征要的形状。

四个已知坑在这里收口(spec §3/§8):
  1. 逐字时间戳在 raw['timestamp'],不在 ASRResult 属性上
  2. VAD 多段 —— 客户端已拼接,这里保留逐段原文并标记 vad_split
  3. 段内相对时间戳 —— 不假装是绝对时间(带 ts_origin 标记)
  4. 这条同步路径(`recognize_pcm` → `arecognize_pcm`)本身不吞异常(吞异常的是
     `ASRSession` 那条线),这里同样不吞、直接上抛;构造函数里的 `timeout_s` 只作为
     客户端的 `timeout` 传下去,而它**只兜住推完音频后收尾 drain 的等待** ——
     不是"中途卡死的看门狗":推流过程中 socket 若挂住,本适配层不会自己中断
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import funasr_client

_CJK = re.compile(r"[㐀-䶿一-鿿]")
_CONFIG_PATH = Path(__file__).with_name("asr_config.json")


def load_config() -> dict[str, Any]:
    """读采集侧配置(公开访问器:store 与 Task 2 的端点都从这里取,别跨模块用私有名)。"""
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


def count_cjk_chars(text: str) -> int:
    """字数 = 中文字符数(去标点与空白,spec §6.5)。"""
    return len(_CJK.findall(text or ""))


def count_non_space_chars(text: str) -> int:
    """字数 = **非空白字符**数(汉字、字母、数字、标点都算)。

    ## 为什么另立一个函数,而不是改 `count_cjk_chars`

    2026-11-21 定位到的病根:`count_cjk_chars` 只数汉字(`_CJK` 那个正则),
    于是**英文/数字段一律算 0 字**。实测 `data/transcripts/20260928_180131_8102.json`:

    ```
    segments[0].text  "good evening， respect teachers。 i'm wonto from …"  202 字符 → n_chars = 0
    segments[1].text  "， and if i'm given the opportunity to enter …"      196 字符 → n_chars = 0
    ```

    两段都是**有人说话**的真实回答(整场口试里有英文自我介绍 + 数字),
    报成 0 字等于说"这段没人说话"。所以 `n_chars` 那一列在正式语料上
    系统性低估字数,而 `speech_rate` 恰好用着它。

    ⚠️ **不改 `count_cjk_chars` 的语义**:`speech_rate` 与 `n_chars` 列都挂在它上面,
    改它会让既有读数整体漂移(历史报告不可复核)。要"能算英文/数字"的那条量
    另起名字、另算一遍(见 `report_frontend/l1_voice` 的 `answer_detail_chars_p50`)——
    本仓对"改了会动既有值"的东西一律新增而不是覆写。

    ## 构念边界(写清楚,免得读者按"汉字数"理解)

    * **含标点**:`，`、`。`、`'` 都算一个字符。中文 ASR 输出的标点约占 3%~5%,
      所以本口径的读数系统性**高于**同段的汉字数;
    * **含面试官**:转写是整场混音(无说话人分离),这一层不区分谁在说;
    * **不去重**:重复说同一个词算多次 —— 它数的是"说了多少",不是"会多少词"。
    """
    return sum(1 for ch in (text or "") if not ch.isspace())


@dataclass
class AsrUtterance:
    text: str = ""
    n_chars: int = 0
    n_segments: int = 1
    vad_split: bool = False
    segments: list[dict[str, Any]] = field(default_factory=list)


class FunASREngine:
    def __init__(self, client: Any = None, host: str | None = None,
                 port: int | None = None, timeout_s: float | None = None):
        cfg = load_config()
        # ⚠️ 2026-11-22 修「配置改了但进程还在用旧地址」:
        #   原实现把 host/port **冻结在构造时**,而 `voice_interaction/api/app.py` 里
        #   `asr_engine = FunASREngine()` 是 **import 时** 执行的 ⟹ 改
        #   `asr_config.json` 后不重启进程,实时链路仍连旧地址(实测:配置已是
        #   127.0.0.1、端点也活着,但 /asr 仍 10s 握手超时;对照实验同一段音频
        #   `FunASREngine(host="127.0.0.1")` 4.4s 成功)。
        #   现在:显式传入的 host/port 仍最高优先(测试与特派用),**没显式传的每次调用
        #   现读配置** —— 改配置即生效,不必重启。
        self._explicit_host = host
        self._explicit_port = port
        self._explicit_timeout = timeout_s
        self.host = host or cfg["funasr_host"]
        self.port = port or cfg["funasr_port"]
        self.timeout_s = timeout_s or float(cfg["timeout_s"])
        self.client = client or funasr_client

    def endpoint(self) -> tuple[str, int]:
        """当前要连的 FunASR 地址。**没显式传就现读配置**（见 `__init__` 的说明）。"""
        if self._explicit_host is not None and self._explicit_port is not None:
            return self._explicit_host, int(self._explicit_port)
        cfg = load_config()
        host = self._explicit_host or cfg["funasr_host"]
        port = int(self._explicit_port or cfg["funasr_port"])
        self.host, self.port = host, port      # 同步到属性,便于 /health 回显
        return host, port

    def transcribe_pcm(self, pcm: bytes) -> AsrUtterance:
        """识别一段 16 kHz/16 bit/单声道 PCM。异常上抛,不吞。

        关键字名以 vendored 客户端为准:`recognize_pcm(pcm, host=, port=, timeout=)`
        (timeout 是客户端自己的收尾等待上限;不是本地看门狗)。
        """
        host, port = self.endpoint()          # 每次调用现读（除非显式传过）
        res = self.client.recognize_pcm(
            pcm, host=host, port=port, timeout=self.timeout_s)
        return self._to_utterance(res)

    @staticmethod
    def _to_utterance(res: Any) -> AsrUtterance:
        segs = list(getattr(res, "segments", None) or [])
        if not segs:
            segs = [res]
        out: list[dict[str, Any]] = []
        for i, s in enumerate(segs):
            raw = getattr(s, "raw", None) or {}
            out.append({
                "index": i,
                "text": getattr(s, "text", "") or "",
                "n_chars": count_cjk_chars(getattr(s, "text", "") or ""),
                "timestamps_ms": raw.get("timestamp") or [],
                "punc_array": raw.get("punc_array") or [],
                "ts_origin": "segment_relative",
            })
        text = getattr(res, "text", "") or ""
        return AsrUtterance(text=text, n_chars=count_cjk_chars(text),
                            n_segments=len(out), vad_split=len(out) > 1, segments=out)
