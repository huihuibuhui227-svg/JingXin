"""FunASR 的薄适配器:把客户端结果规整成落盘与算特征要的形状。

四个已知坑在这里收口(spec §3/§8):
  1. 逐字时间戳在 raw['timestamp'],不在 ASRResult 属性上
  2. VAD 多段 —— 客户端已拼接,这里保留逐段原文并标记 vad_split
  3. 段内相对时间戳 —— 不假装是绝对时间(带 ts_origin 标记)
  4. 客户端会吞异常/卡住 —— 这里把异常上抛、给每次调用加超时
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


def _config() -> dict[str, Any]:
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


def count_cjk_chars(text: str) -> int:
    """字数 = 中文字符数(去标点与空白,spec §6.5)。"""
    return len(_CJK.findall(text or ""))


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
        cfg = _config()
        self.host = host or cfg["funasr_host"]
        self.port = port or cfg["funasr_port"]
        self.timeout_s = timeout_s or float(cfg["timeout_s"])
        self.client = client or funasr_client

    def transcribe_pcm(self, pcm: bytes) -> AsrUtterance:
        """识别一段 16 kHz/16 bit/单声道 PCM。异常上抛,不吞。

        关键字名以 vendored 客户端为准:`recognize_pcm(pcm, host=, port=, timeout=)`
        (timeout 是客户端自己的收尾等待上限;不是本地看门狗)。
        """
        res = self.client.recognize_pcm(
            pcm, host=self.host, port=self.port, timeout=self.timeout_s)
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
