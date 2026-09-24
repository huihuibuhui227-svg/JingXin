#!/usr/bin/env python3
"""
FunASR 客户端库 —— 直接嵌入你自己的系统使用

设计目标: 把 WebSocket 协议细节全部封装, 你的业务代码只需要调用
          recognize_pcm() / recognize_file() / ASRSession 这几个接口。

依赖:
    pip install websockets numpy

三种用法:
    1. 一次性识别(文件/整段音频)   -> recognize_file() / recognize_pcm()
    2. 实时流式(麦克风/持续音频流) -> ASRSession
    3. 需要 await 的场景           -> arecognize_* 异步版本

把本文件拷到你的项目里, import 即可, 无需改动服务端。
"""
from __future__ import annotations

import asyncio
import json
import threading
import wave
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Optional

import numpy as np
import websockets

# ---------------------------------------------------------------------------
# 服务端地址: 局域网内其他机器改成 192.168.72.30 或 10.31.26.238
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 10095

SAMPLE_RATE = 16000          # FunASR 固定要求 16kHz
BYTES_PER_SAMPLE = 2         # int16

# chunk_size=[5,10,5] + chunk_interval=10 => 每块 60ms, 每 600ms 出一次结果
CHUNK_SIZE = [5, 10, 5]
CHUNK_INTERVAL = 10
BLOCK_MS = 60 * CHUNK_SIZE[1] / CHUNK_INTERVAL          # = 60ms
BLOCK_BYTES = int(SAMPLE_RATE * BLOCK_MS / 1000) * BYTES_PER_SAMPLE   # = 1920


# ---------------------------------------------------------------------------
@dataclass
class ASRResult:
    """一次识别的结果"""
    text: str = ""
    is_final: bool = False          # True 表示这是最终精修结果
    mode: str = ""                  # 2pass-online / 2pass-offline / online / offline
    raw: dict = field(default_factory=dict)
    # 服务端 VAD 会把长音频切成多段, 每段发一条 2pass-offline 结果。
    # 合并后的结果在这里保留每一段的原文, text/raw 是全段的拼接。
    segments: list["ASRResult"] = field(default_factory=list)

    @property
    def is_partial(self) -> bool:
        """是否是中间增量结果(打字机效果)"""
        return not self.is_final


# ---------------------------------------------------------------------------
def pcm_from_wav(path: str) -> bytes:
    """读 wav -> 16kHz/mono/int16 PCM。多声道会混合, 非 16k 会重采样。"""
    with wave.open(path, "rb") as w:
        ch, sw, fr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)

    if sw != 2:
        raise ValueError(f"只支持 16bit PCM, 当前 {sw*8}bit。"
                         f"请先转码: ffmpeg -i in.wav -ar 16000 -ac 1 -c:a pcm_s16le out.wav")

    arr = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1).astype(np.int16)
    if fr != SAMPLE_RATE:
        idx = np.linspace(0, len(arr) - 1, int(len(arr) * SAMPLE_RATE / fr))
        arr = np.interp(idx, np.arange(len(arr)), arr).astype(np.int16)
    return arr.tobytes()


def _iter_blocks(pcm: bytes) -> Iterator[bytes]:
    for i in range(0, len(pcm), BLOCK_BYTES):
        yield pcm[i:i + BLOCK_BYTES]


# ---------------------------------------------------------------------------
# 异步核心
# ---------------------------------------------------------------------------
async def arecognize_pcm(
    pcm: bytes,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    mode: str = "2pass",
    on_partial: Optional[Callable[[ASRResult], None]] = None,
    realtime: bool = False,
    timeout: float = 60.0,
) -> ASRResult:
    """
    识别一段完整 PCM (整段音频已拿到手时用这个)

    Args:
        pcm:        16kHz/mono/int16 的原始 PCM 字节
        mode:       "2pass"(推荐, 实时+精修) | "online"(最快) | "offline"(整段)
        on_partial: 回调, 每收到一个中间结果调用一次(做实时 UI 用)
        realtime:   True 则按真实时间节奏发送(模拟麦克风, 用于测试延迟)

    Returns:
        最终 ASRResult。**长音频会被服务端 VAD 切成多段**, 每段一条 2pass-offline
        结果 —— 这里会把所有段按顺序拼起来(text 累加, timestamp 逐段接续),
        逐段原文见返回值的 .segments。

        注意: 早期版本是 `final = r` 覆盖赋值, 只会留下【最后一段】,
        前 30 多秒的文本会被静默丢掉(表现为"前面没识别进去")。
    """
    uri = f"ws://{host}:{port}"
    finals: list[ASRResult] = []

    async with websockets.connect(
        uri, subprotocols=["binary"], ping_interval=None, max_size=None,
    ) as ws:
        # 1) 配置
        await ws.send(json.dumps({
            "mode": mode,
            "chunk_size": CHUNK_SIZE,
            "chunk_interval": CHUNK_INTERVAL,
            "wav_name": "api",
            "audio_fs": SAMPLE_RATE,
            "is_speaking": True,
        }))

        # 2) 推音频
        for block in _iter_blocks(pcm):
            await ws.send(block)
            if realtime:
                await asyncio.sleep(BLOCK_MS / 1000)
            # 顺手收掉已到达的增量结果
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.001)
                except asyncio.TimeoutError:
                    break
                r = _parse(msg)
                if r.is_final and r.text:
                    finals.append(r)
                elif r.text and on_partial:
                    on_partial(r)

        # 3) 结束标记
        await ws.send(json.dumps({"is_speaking": False, "is_end": True}))

        # 4) 收尾
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                break
            r = _parse(msg)
            if r.text and r.is_final:
                finals.append(r)
            elif r.text and on_partial:
                on_partial(r)
            if r.raw.get("is_end"):
                break

    return _merge_finals(finals)


def _merge_finals(segments: list[ASRResult]) -> ASRResult:
    """把服务端按 VAD 段发来的多条离线结果拼成一条。

    text 直接首尾相接(每段本身以标点结尾); timestamp 逐段接续 ——
    第 n 段的偏移取第 n-1 段的结束时间。**段间静音不会计入**, 所以
    第 2 段之后的时间戳会比真实位置略偏早(偏差 = 段间静音长度)。
    要精确的绝对时间戳就按静音自己切段、逐段单独识别。
    """
    if not segments:
        return ASRResult()
    if len(segments) == 1:
        s = segments[0]
        return ASRResult(text=s.text, is_final=s.is_final, mode=s.mode,
                         raw=s.raw, segments=[s])

    offset = 0
    text_parts: list[str] = []
    stamps: list[list[int]] = []
    puncs: list[int] = []
    for s in segments:
        text_parts.append(s.text)
        ts = s.raw.get("timestamp") or []
        stamps.extend([[a + offset, b + offset] for a, b in ts])
        puncs.extend(s.raw.get("punc_array") or [])
        if ts:
            offset += ts[-1][1]          # 上一段的结束时间 = 下一段的起点估计

    raw = dict(segments[-1].raw)
    raw["text"] = "".join(text_parts)
    raw["timestamp"] = stamps
    raw["punc_array"] = puncs
    raw["segment_count"] = len(segments)
    return ASRResult(text=raw["text"], is_final=True,
                     mode=segments[-1].mode, raw=raw, segments=segments)


def _parse(msg: str) -> ASRResult:
    d = json.loads(msg)
    mode = d.get("mode", "")
    # 2pass-offline / offline 视为最终结果; 2pass-online / online 是增量
    is_final = mode in ("2pass-offline", "offline") and d.get("is_final", False)
    return ASRResult(text=d.get("text", ""), is_final=is_final, mode=mode, raw=d)


# ---------------------------------------------------------------------------
# 同步封装 (不想碰 asyncio 的业务代码用这些)
# ---------------------------------------------------------------------------
def recognize_pcm(pcm: bytes, **kw) -> ASRResult:
    """同步版: 识别一段 PCM"""
    return asyncio.run(arecognize_pcm(pcm, **kw))


def recognize_file(path: str, **kw) -> ASRResult:
    """同步版: 识别一个 wav 文件"""
    return recognize_pcm(pcm_from_wav(path), **kw)


# ---------------------------------------------------------------------------
# 实时流式会话 —— 麦克风/持续音频流用这个
# ---------------------------------------------------------------------------
class ASRSession:
    """
    实时流式识别会话。在你的代码里这样用:

        def on_text(r):
            print(r.text, "(最终)" if r.is_final else "")

        with ASRSession(on_result=on_text) as s:
            while recording:
                s.feed(pcm_chunk)      # 喂任意长度 PCM, 内部自动分块

    回调 on_result 会在【后台线程】被调用, 拿到的 text 是"到目前为止的完整句子"
    (服务端返回的是累积文本, 不是增量片段)。
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        mode: str = "2pass",
        on_result: Optional[Callable[[ASRResult], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        self.host, self.port, self.mode = host, port, mode
        self.on_result = on_result
        self.on_error = on_error

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ws = None
        self._q: Optional[asyncio.Queue] = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._buf = bytearray()

    # ---- 生命周期 ----
    def start(self, timeout: float = 30.0):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("连接 FunASR 服务超时, 检查地址/端口/服务是否启动")
        return self

    def stop(self):
        """结束会话, 等待服务端返回最后结果"""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---- 喂数据 ----
    def feed(self, pcm: bytes):
        """喂任意长度的 16k/mono/int16 PCM, 内部自动按 60ms 分块"""
        if self._loop is None or self._q is None:
            raise RuntimeError("会话未启动, 先调用 start()")
        self._buf.extend(pcm)
        while len(self._buf) >= BLOCK_BYTES:
            block = bytes(self._buf[:BLOCK_BYTES])
            del self._buf[:BLOCK_BYTES]
            self._loop.call_soon_threadsafe(self._q.put_nowait, block)

    def feed_file(self, path: str, realtime: bool = True):
        """把整个 wav 喂进去(调试用); realtime=True 按真实速度"""
        import time
        pcm = pcm_from_wav(path)
        for block in _iter_blocks(pcm):
            self.feed(block)
            if realtime:
                time.sleep(BLOCK_MS / 1000)

    # ---- 内部 ----
    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:
            if self.on_error:
                self.on_error(e)
        finally:
            self._loop.close()

    async def _main(self):
        self._q = asyncio.Queue()
        uri = f"ws://{self.host}:{self.port}"
        try:
            async with websockets.connect(
                uri, subprotocols=["binary"], ping_interval=None, max_size=None,
            ) as ws:
                self._ws = ws
                await ws.send(json.dumps({
                    "mode": self.mode,
                    "chunk_size": CHUNK_SIZE,
                    "chunk_interval": CHUNK_INTERVAL,
                    "wav_name": "session",
                    "audio_fs": SAMPLE_RATE,
                    "is_speaking": True,
                }))
                self._ready.set()

                async def pump():            # 持续发送
                    while not self._stop.is_set():
                        try:
                            block = await asyncio.wait_for(self._q.get(), timeout=0.2)
                        except asyncio.TimeoutError:
                            continue
                        await ws.send(block)
                    await ws.send(json.dumps({"is_speaking": False, "is_end": True}))

                async def recv():            # 持续接收
                    try:
                        async for msg in ws:
                            r = _parse(msg)
                            if r.text and self.on_result:
                                self.on_result(r)
                    except Exception:
                        pass

                await asyncio.gather(pump(), recv())
        except Exception as e:
            self._ready.set()               # 防止 start() 卡死
            if self.on_error:
                self.on_error(e)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 自测: 识别同目录下的 asr_example.wav
    import sys, time

    wav = sys.argv[1] if len(sys.argv) > 1 else "asr_example.wav"
    host = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_HOST

    print("=== 1) 一次性识别 ===")
    t0 = time.time()
    r = recognize_file(wav, host=host)
    print(f"结果: {r.text}")
    print(f"耗时: {time.time()-t0:.2f}s")

    print("\n=== 2) 实时流式(带增量回调) ===")
    def on_text(res: ASRResult):
        tag = "最终" if res.is_final else "实时"
        print(f"  [{tag}] {res.text}")

    with ASRSession(host=host, mode="2pass", on_result=on_text) as s:
        s.feed_file(wav, realtime=True)
        time.sleep(1.5)
    print("\n完成")
