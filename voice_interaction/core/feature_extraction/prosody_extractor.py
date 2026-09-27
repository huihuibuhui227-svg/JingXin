"""
语音韵律特征提取器

负责从原始音频中提取音高、能量、语速等基础特征
"""

import math

import numpy as np
import librosa
from typing import Dict, Any


class ProsodyFeatureExtractor:
    """语音韵律特征提取器"""

    def __init__(self, sample_rate: int = 16000):
        """
        初始化特征提取器

        参数:
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate
        self.fmin = librosa.note_to_hz('C2')
        self.fmax = librosa.note_to_hz('C7')

    def extract_pitch_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        提取音高相关特征

        参数:
            audio: 音频数据

        返回:
            包含音高特征的字典
        """
        # 空音频 = **一帧都没有** ⟹ 七个键一律留空(None → 落盘空串),**不写 0.0**。
        # 判据与下面「零浊音帧」那一支完全相同(那里有完整说明):「没测到」与「量到了 0」不是一回事。
        if len(audio) == 0:
            return {"pitch_mean": None, "pitch_std": None,
                    "pitch_trend": None, "pitch_direction": "无法判断",
                    "pitch_p10": None, "pitch_p90": None,
                    "voiced_prob_mean": None}

        # 计算基频。★ 第 3 个返回值 `voiced_prob`(逐帧「是浊音」的概率)**此前被 `_` 丢掉** ——
        # `docs/superpowers/specs/2026-09-21-jingxin-feature-redesign-design.md:235` 的依据栏点名的就是它(«pyin 无能量门限且丢 `voiced_prob`»):
        # 它是给 pitch 族加门控的原料,也是日志列 `voiced_prob_mean` 的产出方(Task 5)。
        #
        # ⚠️ **但"拿它当门控"这件事实测做不到 —— 别照着上面那句直接写出 `f0[voiced_prob > T]`。**
        # 2026-09-27 在三场正式素材的 24 段回答上实测(`experiments/voice_pitch_gate_probe.py`):
        #   · 门控**去掉**的那些帧里「噪声帧」(f0 > 400 Hz)的占比 = 0.016~0.019,
        #     而全体浊音帧里的噪声帧占比 = 0.017~0.018 —— **两者相等** ⟹ 它没有挑出噪声,
        #     只是在按比例删帧;
        #   · 阈值 0.02/0.05/0.1 下,最坏那段的 `pitch_mean` 反而从 398.20 Hz **升到**
        #     988.67 / 932.84 / 800.02 Hz(留下的帧比平均数更偏高频);
        #   · 3/24 段被整段删空,而删空的偏偏是**最干净**的那几段(p90 ≈ 96 Hz);
        #   · 阈值 0.2/0.4 才能把最大值压到 332/345 Hz,代价是只留 3~5% 的帧、4/24 段为空。
        # 所以 pitch 那四行**仍是未门控的现状**,`pitch_mean` 实测最大仍是 398.20 Hz、
        # `pitch_p90` 最大仍是 1729.76 Hz。门控要用什么量、阈值取多少,是**标定集**的事
        # (M3.1 已被使用者裁定跳过)—— 那两行留在 `pending` 就是这个原因。
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio, fmin=self.fmin, fmax=self.fmax, sr=self.sample_rate
        )
        f0_voiced = f0[voiced_flag]

        # ★ 零浊音帧(全静音):`voiced_flag` 全 `False` ⟹ `f0_voiced` 是空的、`f0` 全 `nan`。
        # 这一支里**两类量必须分开处置**(plan `docs/superpowers/plans/2026-09-26-m3-0-l0-column-table.md:38`):
        #   · `voiced_prob_mean` = **0.0 是真值**(确实一个浊音帧都没有,pyin 的概率恒 0)⟹
        #     **不做特殊处理**。把它写成空等于把「量到了 0」降级成「没测到」,而 0 在这里
        #     是最强的断言(「这不是语音」)。
        #   · **`f0_voiced` 上的五个统计量一律留空**(`None` → 落盘空串):`pitch_mean` /
        #     `pitch_std` / `pitch_trend` / `pitch_p10` / `pitch_p90`。`f0` 全 `nan` ⟹ 它们
        #     **全都没有定义**。
        #     ⚠️ **改前的实情(2026-09-26 复核实测,别把它记错)**:这一支当时只写 `pitch_std` /
        #     `pitch_trend` 为 `0.0`,而 `pitch_p10` / `pitch_p90` **连键都没有**(落盘由 logger 的
        #     `.get` 兜成空串)。所以"把没测到写成 0"的受害列是**两列**(std / trend),
        #     `p10` / `p90` 的问题是**缺键** —— 缺键与显式空值对下游不是一回事(按列名读的代码
        #     拿到 `KeyError`,而 `.get` 与 `if` 判断会给出两种不同结论)。
        #     两者一起收口:`0.0` 是「一个看着像测量值、其实什么都没量到的数」
        #     (2026-09-26 使用者裁定:五个一起改,别只改均值 —— 留一个修好、四个不修,
        #     读的人无法判断这一支可不可信);**键必须存在**(显式的空值)。
        #     同一个判据只写一遍:这一支是那五个量的**唯一**出口,不在别处再复制。
        if len(f0_voiced) == 0:
            return {
                "pitch_mean": None,
                "pitch_std": None,
                "pitch_trend": None,
                "pitch_p10": None,
                "pitch_p90": None,
                "pitch_direction": "无法判断",
                "voiced_prob_mean": float(np.mean(voiced_prob)),
            }

        pitch_mean = float(np.mean(f0_voiced))
        pitch_std = float(np.std(f0_voiced))

        # 分析语调趋势。★ 门槛是 **≥3 帧**:切"首/末三分之一"要 `n//3 ≥ 1`,少于 3 帧切出来是**空切片**,
        # `np.mean([])` 给 `nan` —— 于是 `pitch_trend` 会变成 `nan`(落盘成字符串 `"nan"`,又一个
        # "看着像测量值、其实什么都没量到"的数),而 `pitch_direction` 还会写 `"平稳"`
        # (因为 `nan > 10` 与 `nan < -10` 都是 False)—— 那是一句**假话**(根本没量到趋势)。
        # 实测(2026-09-26,复核 Minor 7 的同一条判据):1 帧 → 0.0、2 帧 → nan + "平稳"、
        # 3 帧 → 有值。⟹ <3 帧一律留空 + `"无法判断"`。
        if len(f0_voiced) >= 3:
            n = len(f0_voiced)
            first_third = float(np.mean(f0_voiced[:n//3]))
            last_third = float(np.mean(f0_voiced[-n//3:]))

            # ★ `pitch_trend` 的单位是**半音**,不是 Hz 差(L0 行 `pitch_trend` 验收①;
            #   依据 spec §4.3:236 ——「是首末均值之差,单位 Hz 不是 Hz/s(拉长 4 倍几乎不变)」)。
            #   `12·log2(末/首)` 是**同量纲比值**的对数 ⟹ 跨说话人可比;裸 Hz 差不是:
            #   同样是「升了一个八度」,男声量到 ~100 Hz、女声量到 ~200 Hz。
            #   单位必须与 L1 的 `pitch_range_semitone` 一致(§5.4:363),否则差 100 倍。
            #   为什么是「改半音」而不是「改名」:改名要同时动 `VoiceLogger.fieldnames`
            #   (列契约)、双向钉子与历史 CSV 列序,改半音只动这一个返回值(见 L0 行 basis③)。
            hz_diff = last_third - first_third
            if first_third > 0 and last_third > 0:
                pitch_trend = 12.0 * math.log2(last_third / first_third)

                # ⚠️ `pitch_direction` 是白名单里的**历史列**,门限历来是 ±10 **Hz**,
                #    它必须继续吃 `hz_diff` —— **不许**跟着改成吃半音值:那样「10」就成了
                #    10 半音(≈ 一个八度),「上扬/下降」几乎再也发不出来,且**不报错**
                #    (nan 与超阈都是静默的,这正是本项目反复栽的那一类)。
                #    `tests/test_voice_first_order_columns.py::test_pitch_direction_keeps_its_hertz_threshold`
                #    用 1000→1020 Hz 钉住这一点(Hz 差 20 > 10 判「上扬」,而半音只有 ≈0.343)。
                if hz_diff > 10:
                    pitch_direction = "上扬"
                elif hz_diff < -10:
                    pitch_direction = "下降"
                else:
                    pitch_direction = "平稳"
            else:
                # 均值 ≤ 0 时比值无定义(f0 不该 ≤ 0,但 pyin 的边界行为不作保证)。
                # 与「零浊音帧」同一条判据:算不出来就**留空**,不写 0.0 —— 0 是个真值。
                # 趋势没量到就不许写方向(与 <3 帧那一支同理)。
                pitch_trend = None
                pitch_direction = "无法判断"
        else:
            # ★ **少于 3 帧浊音也算不出「首末之差」** ⟹ 与零浊音帧同一条判据:留空,不写 0.0
            # (写 0.0 就是"一个看着像测量值、其实什么都没量到的数";2026-09-26 复核 Minor 7)。
            # `pitch_direction` 有它自己的"测不出"取值(分类型),照旧写 `"无法判断"`。
            pitch_trend = None
            pitch_direction = "无法判断"

        # 分位数:与 mean/std 不同,它们**不受极端帧拖拽**,而且是"会话内归一"的原料
        # (报告的 energy/pitch 封停理由之一就是"跨会话不可比,需会话内归一")。
        # ⚠️ 这两行**故意不带 `if len(f0_voiced) else 0.0` 兜底**(2026-09-26 复核 Minor 6 删掉):
        # `:64` 那一支已经把"零浊音帧"挡掉了,兜底在这里是**死代码**,而它是本函数里**仅存**的
        # 一处"没数据就写 0.0"的写法 —— 留着会削弱"这一支只有一个出口"的可读性。
        p10 = float(np.percentile(f0_voiced, 10))
        p90 = float(np.percentile(f0_voiced, 90))

        # ★ 浊音概率的均值:取**全部帧**(不是"浊音帧的均值")。只在 `voiced_flag` 为真的帧上
        # 取均值会与 `voiced_flag` 的含义重合,而且全静音段会变成"无值";取全部帧的均值才同时
        # 表达「这声音有多像语音」与「有多少帧像语音」,全静音时它是 0.0 —— 一个真值。
        # 不四舍五入:概率的低端正是门控要用的地方(实测:真素材 17 段 0.0102~0.0514,
        # 合成正弦段 0.6245,白噪声 0.0100 —— 四位小数才分得开噪声底与真回答)。
        voiced_prob_mean = float(np.mean(voiced_prob))
        return {
            "pitch_mean": round(pitch_mean, 2),
            "pitch_std": round(pitch_std, 2),
            "pitch_trend": None if pitch_trend is None else round(pitch_trend, 2),
            "pitch_direction": pitch_direction,
            "pitch_p10": round(p10, 2),
            "pitch_p90": round(p90, 2),
            "voiced_prob_mean": voiced_prob_mean
        }

    def extract_energy_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        提取能量相关特征

        参数:
            audio: 音频数据

        返回:
            包含能量特征的字典
        """
        # 空音频 = 一帧都没有 ⟹ 四个键**一律留空**,不写 0.0 —— 判据与
        # `extract_pitch_features` 的空音频支完全相同(那里有完整说明;2026-09-26 整支复核 Minor 8 同批收)。
        if len(audio) == 0:
            return {"energy_mean": None, "energy_std": None,
                    "energy_p10": None, "energy_p90": None}

        rms = librosa.feature.rms(y=audio)[0]
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))

        e10 = float(np.percentile(rms, 10))
        e90 = float(np.percentile(rms, 90))
        return {
            "energy_mean": round(energy_mean, 4),
            "energy_std": round(energy_std, 4),
            "energy_p10": round(e10, 4),
            "energy_p90": round(e90, 4)
        }

    def extract_speech_ratio(self, audio: np.ndarray) -> float:
        """
        计算说话时间占比（语速/流畅度）

        参数:
            audio: 音频数据

        返回:
            说话时间占比 (0-1)
        """
        if len(audio) == 0:
            return 0.0

        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        non_silent = np.sum(spectral_centroids > np.mean(spectral_centroids) * 0.1)
        speech_ratio = float(non_silent / len(spectral_centroids)) if len(spectral_centroids) > 0 else 0.0

        return round(speech_ratio, 2)

    def extract_pause_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        提取停顿相关特征

        参数:
            audio: 音频数据

        返回:
            包含停顿特征的字典
        """
        # 空音频 = 一帧都没有 ⟹ 五个键**一律留空**(`speech_onset_sec` 本来就是 `None`),
        # **不写 0.0** —— 那四个 0.0 正是「没测到写成了 0」这个形态。
        # 判据与 `extract_pitch_features` 的空音频支完全相同(那里有完整说明);
        # 2026-09-26 M3.0 整支复核 Minor 8 把三处同形态一并收口。
        if len(audio) == 0:
            return {"pause_duration_mean": None, "pause_duration_max": None,
                    "pause_frequency": None, "speech_duration_sec": None,
                    "speech_onset_sec": None}

        rms = librosa.feature.rms(y=audio)[0]
        duration = len(audio) / self.sample_rate

        # 使用能量阈值检测停顿
        energy_threshold = np.mean(rms) * 0.3
        is_speech = rms > energy_threshold

        # ★ **有声时长**(秒)与帧步长:下面 `pause_frequency` 的**分母**要用它。
        # 此前这两个量算在停顿统计**之后** —— 顺序换过来不是风格问题:`pause_frequency`
        # 的分母由「整段时长」改成「语音时长」之后,它必须在算频率之前就存在
        # (spec §4.3:239「现用总时长(含静音)→ 改语音时长」;分母用本表的 `speech_duration_sec`)。
        frame_sec = duration / len(rms) if len(rms) else 0.0
        speech_seconds = round(float(np.count_nonzero(is_speech)) * frame_sec, 2)

        # 找到所有停顿区间
        pause_intervals = []
        in_pause = False
        pause_start = 0

        for i, speech in enumerate(is_speech):
            if not speech and not in_pause:
                in_pause = True
                pause_start = i
            elif speech and in_pause:
                in_pause = False
                pause_duration = (i - pause_start) * frame_sec
                if pause_duration > 0.1:
                    pause_intervals.append(pause_duration)

        # ★ **收尾**(2026-09-27,`pause_duration_mean`/`pause_duration_max` 验收①):
        # 音频以静音结尾时,`in_pause` 在循环结束时仍为真,那段停顿此前**整个丢掉** ——
        # 症状原文:「尾部 4s 静音报 0 次停顿」(spec §1.2:55)。循环里那一支只处理
        # 「静音之后又出现语音」,末尾这一支没有「之后」,所以必须单独补。
        # 长度算到 `len(rms)`:帧数 × 帧步长 = 该段停顿覆盖的秒数,与循环里同一把尺子。
        if in_pause:
            pause_duration = (len(rms) - pause_start) * frame_sec
            if pause_duration > 0.1:
                pause_intervals.append(pause_duration)

        # 计算停顿统计
        if len(pause_intervals) > 0:
            pause_duration_mean = round(float(np.mean(pause_intervals)), 2)
            pause_duration_max = round(float(np.max(pause_intervals)), 2)
            # ★ 分母 = **语音时长**(不是整段时长):静音不算说话时间,拿整段当分母等于
            # 「说得越慢、停顿越少」。`speech_seconds == 0`(整段一个有声帧都没有)时
            # 这个率**没有定义** ⟹ 留空,不写 0(0 是一个真实的率,「没测到」不是 0)。
            pause_frequency = (round(len(pause_intervals) / speech_seconds * 60, 2)
                               if speech_seconds > 0 else None)
        else:
            pause_duration_mean = 0.0
            pause_duration_max = 0.0
            pause_frequency = 0.0

        # ★ **首次开口在音频里的位置**(秒):反应延迟要用它。
        #   "录音开始"到"真的开口"之间那段前导静音,不是反应时间 —— 但推首次开口
        #   的**墙钟**必须把它算进去,否则整段前导静音都会被算成反应时间。
        first_voiced = next((i for i, v in enumerate(is_speech) if v), None)
        speech_onset = round(first_voiced * frame_sec, 2) if first_voiced is not None else None
        return {
            "pause_duration_mean": pause_duration_mean,
            "pause_duration_max": pause_duration_max,
            "pause_frequency": pause_frequency,
            "speech_duration_sec": speech_seconds,
            "speech_onset_sec": speech_onset
        }

    def extract_all_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        提取所有韵律特征

        参数:
            audio: 音频数据

        返回:
            包含所有特征的字典
        """
        if len(audio) == 0:
            return {}

        # 提取各类特征
        pitch_features = self.extract_pitch_features(audio)
        energy_features = self.extract_energy_features(audio)
        speech_ratio = self.extract_speech_ratio(audio)
        pause_features = self.extract_pause_features(audio)
        duration = len(audio) / self.sample_rate

        # 合并所有特征
        all_features = {
            **pitch_features,
            **energy_features,
            "speech_ratio": speech_ratio,
            "duration_sec": round(duration, 2),
            **pause_features
        }

        return all_features
