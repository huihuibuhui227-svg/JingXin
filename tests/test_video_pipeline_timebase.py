# tests/test_video_pipeline_timebase.py
"""M2.5:管线的每一处时间都由 `timestamp_ms` 导出,且 `is_blink` 真的进序列化。

不 import mediapipe:探测器注入假件,替身交 `None` 走 no_face 分支,于是不必造 468 个 landmark
就能验时间逻辑 —— 时间逻辑被抽成了可直接调用的 `_update_blink_state`。
"""

from types import SimpleNamespace

import numpy as np

from face_expression.pipeline.video_pipeline import VideoPipeline

_FRAME = np.zeros((48, 48, 3), dtype=np.uint8)


class _FakeDetector:
    def __init__(self):
        self.seen = []
        self.resets = 0
        self.closed = False

    def detect(self, image_rgb, timestamp_ms):
        self.seen.append(timestamp_ms)
        return None                    # 没检出脸 —— 走 no_face 分支

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


def _stub_au(timestamp_ms: int):
    """历史帧的最小替身:`get_summary` 与时间窗只用这两样。"""
    return SimpleNamespace(timestamp_ms=timestamp_ms, focus_score=0.5)


def test_pipeline_forwards_the_timestamp_to_the_detector():
    """接线:调用方给的时间戳必须原样到达探测器。

    红法:`process_frame` 里自己造一个时间戳(或忽略入参)。
    """
    d = _FakeDetector()
    p = VideoPipeline(session_id="s", detector=d)
    for ts in (0, 1000, 2000):
        p.process_frame(_FRAME, ts)

    assert d.seen == [0, 1000, 2000], d.seen


def test_blink_count_uses_real_seconds_not_the_wall_clock():
    """★ `blink_rate_per_min`(实为「60 秒窗内的次数」)必须按**真实时间**算。

    造法:0/500/1000/1500 ms 四帧都闭眼(ear < 0.21),去抖 0.3 s 放行每一次 ——
    相邻间隔都是 0.5 s > 0.3 s,所以四次全部计入,期望 [1, 2, 3, 4]。

    红法:把去抖与 60 秒窗改回 `time.time()`(见 `video_pipeline.py:59,62,66`)——
    四次调用在挂钟上只隔几微秒,`now - last_blink_time > 0.3` 只有第一次成立,
    于是结果退化成 [1, 1, 1, 1],红。
    """
    p = VideoPipeline(session_id="s", detector=_FakeDetector())
    counts = [p._update_blink_state(0.10, ts)[1] for ts in (0, 500, 1000, 1500)]

    assert counts == [1, 2, 3, 4], f"眨眼计数没按真实时间窗算:{counts}"


def test_eye_closed_seconds_uses_the_real_gap_between_frames():
    """★ `eye_closed_sec` 累加的是**相邻帧的真实间隔**,不是 `1/fps`。

    造法:间隔刻意不均匀(300 / 600 / 100 ms)。

    红法:`self.eye_closed_duration += 1 / self.fps`(见 `video_pipeline.py:70`)——
    三次合计会是 3/30 = 0.1 s 而不是 1.0 s。
    """
    p = VideoPipeline(session_id="s", detector=_FakeDetector())
    p._update_blink_state(0.10, 0)          # 第 1 帧:起点,不计时长
    p._update_blink_state(0.10, 300)        # 闭眼,记 0.3
    p._update_blink_state(0.10, 900)        # 闭眼,记 0.6
    _, _, eye_closed = p._update_blink_state(0.10, 1000)   # 闭眼,记 0.1

    assert abs(eye_closed - 1.0) < 1e-6, f"应当是 1.0 秒(0.3+0.6+0.1),实际 {eye_closed}"


def test_no_face_gap_is_not_credited_to_eye_closure():
    """★ 审查 F2:`_update_blink_state` 只在**有脸帧**上跑,而时钟每帧都在走 ——
    于是"上一帧"停在最后一个有脸帧,中间整段没人看的时间会被算成一个闭眼帧的时长。

    实测(修前):一个有脸闭眼帧(t=0)+ 10 秒无脸 + 一个有脸闭眼帧(t=10000)
    → `eye_closed_sec = 10.0`。真实会话里按实测 no_face 率(5 帧 1 帧),每个跟在
    无脸帧后面的闭眼帧会被记成 ≈2 s 而不是 ≈1 s。该列进 TemporalStats 与 tension_engine。

    红法:no_face 分支里不更新 `history_last_ms`。
    """
    p = VideoPipeline(session_id="s", detector=_FakeDetector())   # 替身恒返回 None = no_face
    p._update_blink_state(0.10, 0)              # 有脸:闭眼,起点

    for ts in range(1000, 10000, 1000):         # 1000..9000 全是无脸帧
        p.process_frame(_FRAME, ts)

    _, _, eye_closed = p._update_blink_state(0.10, 10000)
    assert abs(eye_closed - 1.0) < 1e-6, (
        f"只应记最后一帧间隔 1.0 s,实际 {eye_closed} —— 无脸那段被算成闭眼了")


def test_is_blink_reaches_the_serialized_row(monkeypatch):
    """★ 审查 F4:spec §3.4 的三列修复(`current_au.is_blink/...`)**此前没有任何测试**
    —— 于是账本里那条"is_blink 1/5 非零"实际是 pandas 把 NaN 也算成 `!= 0` 的假象。
    这里直接钉序列化出来的那个 dict。

    红法:去掉 `current_au.is_blink = is_blink` 那三行(值只写深拷贝)。
    """
    from face_expression.models.features import AUFeatures
    from face_expression.pipeline import video_pipeline as vp

    class _FaceDetector:
        def detect(self, image_rgb, timestamp_ms):
            return [(0.5, 0.5)] * 500          # 一帧"有脸"(几何退化由管线自己兜住)
        def reset(self):
            pass
        def close(self):
            pass

    p = vp.VideoPipeline(session_id="s", detector=_FaceDetector())
    monkeypatch.setattr(p.feature_calculator, "calculate",
                        lambda *a, **k: AUFeatures(avg_ear=0.10), raising=True)

    _, _, d = p.process_frame(_FRAME, 1000)

    assert d["is_blink"] == 1, f"序列化的行里 is_blink 不是真的:{d.get('is_blink')}"
    assert d["blink_rate_per_min"] == 1, d.get("blink_rate_per_min")


def test_no_face_row_carries_the_real_timestamp():
    """★ E2E 抓到的真缺陷:`no_face` 提前返回的 dict 以前**不带** timestamp,
    于是 face 的 logger 用 `datetime.now().timestamp()` **编了一个墙钟**
    (`face_expression/utils/logger.py:90-91`)—— 整列时间戳里混进一个 1.79e9 的绝对秒值。

    实测(真服务 + 真脸图,5 帧):恰有 1 帧没检出脸,那一行 timestamp = 1.790318e+09,
    其余四行是 0.00 / 0.09 / 0.15 / 0.19。这是 spec §5.1「时间只剩一个来源」的直接违反。

    红法:把 `{"emotion": "no_face"}` 改回不带 timestamp —— logger 的兜底会再次兜出墙钟。
    """
    p = VideoPipeline(session_id="s", detector=_FakeDetector())
    result, _, d = p.process_frame(_FRAME, 4321)

    assert result is None and d["emotion"] == "no_face"
    assert d["timestamp"] == 4.321, f"no_face 行没带上真实时间戳:{d}"


def test_no_history_yet_gives_zero_not_an_exception():
    """★ Review Focus ③:`no_face` 帧不推进历史 —— 一帧都没进过时 `get_summary` 不许抛。

    红法:在 `get_summary` 里无条件读 `self.au_history[0]`。
    """
    p = VideoPipeline(session_id="s", detector=_FakeDetector())
    p.process_frame(_FRAME, 0)              # 没检出脸 → au_history 仍为空

    s = p.get_summary()
    assert s["frame_count"] == 0
    assert s["duration_sec"] == 0


def test_single_frame_session_has_zero_duration():
    """★ Review Focus ⑤:只有一帧时 `duration_sec` = 0,不许抛、不许 NaN。"""
    p = VideoPipeline(session_id="s", detector=_FakeDetector())
    p.au_history.append(_stub_au(timestamp_ms=1234))

    s = p.get_summary()
    assert s["duration_sec"] == 0, s["duration_sec"]
    assert s["frame_count"] == 1


def test_summary_duration_is_the_session_span_not_the_history_window():
    """★ 审查 F7:`get_summary()["duration_sec"]` 取自 `au_history` 的首末帧,而那个 deque
    被 3 秒窗修剪 —— 于是它**恒 ≤ 3 秒**,不管会话多长。该字段进 `/session/{id}/summary`,
    也进报告层的实时行(`report_frontend/data_loader.py:260`)。

    实测(修前):60 帧 @1 fps → `duration_sec 3.0, frame_count 4`。字段叫 duration_sec,
    给的却是"历史窗长度",读者无从分辨 —— 与 F6 同源。

    会话真正的跨度管线一直知道(`first_ts`/`last_ts`),没有理由不用。

    红法:把 `duration_sec` 改回 `(last - first) / 1000.0`(那是 history 的首末帧)。
    """
    p = VideoPipeline(session_id="s", detector=_FakeDetector())
    for i in range(60):
        p.process_frame(_FRAME, i * 1000)        # 60 秒会话(替身恒无脸)
    p.au_history.append(_stub_au(timestamp_ms=59000))

    s = p.get_summary()
    assert s["duration_sec"] == 59.0, (
        f"应当是**会话跨度** 59.0 秒,实际 {s['duration_sec']}(那是 3 秒历史窗)")


def test_summary_reports_which_population_each_rate_counts():
    """★ 审查 F6:同一个 dict 里 `fps` 数的是**全部提交帧**、`frame_count`/`duration_sec`
    数的是**有脸帧** —— 口径不同却没说明,读者会信错一个。

    这条钉住修好之后的口径:`duration_sec` = 会话跨度;`fps` = 提交帧 ÷ 会话跨度;
    而 `frame_count` 仍是历史里有脸帧的数(它本来就只统计那些)。三者各自说得清。

    红法:让 `fps` 与 `duration_sec` 用两个不同的时间基准(例如 fps 用 history 跨度)。
    """
    p = VideoPipeline(session_id="s", detector=_FakeDetector())
    for i in range(10):
        p.process_frame(_FRAME, i * 1000)

    s = p.get_summary()
    assert s["duration_sec"] == 9.0, s["duration_sec"]
    assert abs(s["fps"] - 10 / 9.0) < 0.01, (
        f"fps 应是「提交帧 ÷ 会话跨度」= {10 / 9.0:.3f},实际 {s['fps']}")


def test_duration_comes_from_timestamps_not_frame_count():
    """`duration_sec` = 首末**提交帧**之间的秒数,不是 `帧数/fps`。

    ⚠️ 审查 F7 之后这条的定义收紧了:跨度取自 `first_ts`/`last_ts`(会话跨度),
    而**不是** `au_history` 的首末帧 —— 后者被 3 秒窗修剪,会让长会话报出 ≤3 秒。
    所以这里必须真的走 `process_frame` 把 first_ts/last_ts 立起来,
    手搓 `au_history` 已经代表不了会话了(那正是先前这条测试钉错的地方)。

    红法:把 `duration_sec` 改回 `frame_count / 30` —— 三帧会给出 0.1 而不是 9.5。
    """
    p = VideoPipeline(session_id="s", detector=_FakeDetector())
    for ts in (0, 4000, 9500):
        p.process_frame(_FRAME, ts)          # 替身恒无脸,但时间戳照记

    assert p.get_summary()["duration_sec"] == 9.5
