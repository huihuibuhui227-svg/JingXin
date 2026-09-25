"""会话相对时钟:把「距会话开始的毫秒数」做成一个显式对象。

为什么要有这个文件(M2.5 spec §4 / D2):
  * 实时路径**没有视频文件**,时间只能实测 —— 墙钟在实时流里就是真实时间;
  * 而 mediapipe 的 VIDEO 模式要求时间戳**严格递增**,否则抛错;
  * 于是「实测」不能是 `int((now - start) * 1000)` 就完事:同一毫秒内的两帧会撞值。
    本类把那 1 ms 的抖动吃在内部,对外保证严格递增。

离线批量路径**不用**这个类 —— 那里时间由 `帧序号 × FRAME_SKIP / src_fps` 决定,
是确定性的,不该混进墙钟(spec §4 表)。

与 `logging_config.py` 同层放在仓库根:face / gesture 两个服务都以 `-m <模块>.api.app`
启动,仓库根在 sys.path 上,所以两边都 import 得到。
"""

import time
from typing import Callable


class SessionClock:
    """距会话开始的毫秒数,**严格递增**。"""

    def __init__(self, now: Callable[[], float] = time.monotonic):
        # 用 monotonic 而不是 time():后者会被 NTP 回拨,回拨即时间戳回退。
        self._now = now
        self.reset()

    def reset(self) -> None:
        """`/reset` 与新会话都走这里。"""
        self._start = self._now()
        self._last_ms = -1
        self.n_stamps = 0

    def stamp_ms(self) -> int:
        """本帧的时间戳。保证 `> 上一次返回值`(不是 `>=`)。"""
        elapsed_ms = int((self._now() - self._start) * 1000)
        if elapsed_ms <= self._last_ms:
            # 同一毫秒内的两帧 —— 抬到上一次 +1,而不是放行相等值。
            elapsed_ms = self._last_ms + 1
        self._last_ms = elapsed_ms
        self.n_stamps += 1
        return elapsed_ms

    def measured_fps(self) -> float:
        """到当前为止的**滑动实测**帧率 —— 会话收尾记账用(spec §5.5)。

        每次 `stamp_ms()` 就是收到一帧,所以这里不需要额外的帧计数:
        `n_stamps / 已流逝秒`。gesture 服务没有 `VideoPipeline` 可问,
        它的收尾日志就靠这个方法;face 那边同名的量由管线自己报(它另有 first/last_ts)。
        """
        if self.n_stamps == 0 or self._last_ms <= 0:
            return 0.0
        return round(self.n_stamps / (self._last_ms / 1000.0), 3)
