"""语音**重放腿**写 `VoiceLogger` 契约时的登记 + 两条推断规则(甲-4,2026-10-03)。

## 为什么需要这个模块

`experiments/replay_retained.py --modality voice` 拿**留存音频段**重跑活路径那个真产出方
(`voice_interaction.api.app.prosody_features_from_pcm`),产出一张 CSV。它此前写的是
**自己拼的列**(`file` / `seq` / `source_endpoint` / `seconds` + 提取器的键),
而报告层读的是 `VoiceLogger` 的 **32 列契约** ⟹ 那种 CSV 喂给报告时语音系的槽
**成片掉光**,也与**同一场活跑**的日志逐列比不了(甲-4 原文)。

现在行的形状由 `VoiceLogger.build_row` 决定(两处**共用一个实现**),产不出来的列
**逐条登记在 `NOT_WRITTEN_BY_REPLAY` 里并写明为什么** —— 留空,不编。

## 实测(场 `20260927_212143_95a8`,8 段回答)

* `question_index` 按下面那条时间规则推断 ⟹ 与活日志 **8/8 全对**;
* `reaction_time` 重算 ⟹ 与活日志吻合到 **±5 ms**(前提是用 `arrival_wall_of` 取的时刻);
* 提取器给的 18 列与活日志**逐列相同**(`duration_sec` / `speech_onset_sec` 等
  连小数位都一样 —— 因为重放喂的就是当时那批字节)。
"""
from __future__ import annotations

from typing import Any, Iterable, Optional


#: 重放腿**写不出**的列 → 为什么。留空(不是 0、不是占位符)。
#:
#: ⚠️ 这份登记有**守卫**:`tests/test_voice_replay_contract.py` 断言
#: ① 这 6 列的并集 + 重放实际写的列 == `VoiceLogger.fieldnames`(一列不漏、不多);
#: ② 重放产出的文件里,这 6 列**逐行为空** —— 免得将来有人"顺手补个 0"把
#:    「没算出来」写成「算出来是 0」(本项目一路在杀的那个形态)。
NOT_WRITTEN_BY_REPLAY: dict[str, str] = {
    # ── 第一类:要**文本**,而文本来自 ASR;重放腿不跑 ASR(它只负责声学) ──
    "n_chars": "字数来自 ASR 文本(`count_cjk_chars(text)`);重放腿只吃音频、不跑 ASR",
    "chars_per_sec": "同上(n_chars ÷ 有声秒);分子产不出,比值也不许编",
    "connective_density": "连接词密度要逐句分词统计,输入是 ASR 文本",
    "connective_density_std": "同上(分句密度的标准差)",
    "n_rows": "参与密度计算的句数,同样来自文本分句",
    # ── 第二类:活路径**调用方**的占位判断,不是测量 ──
    "is_valid": "活路径恒写 True(`log_prosody` 的默认值,调用方从未传别的值),"
                "封停表里那条「恒 1 且下游从未使用」记的就是它。"
                "重放**不复制一个它没有做出的判断** ⟹ 留空",
}


def _wall_of(rec: Optional[dict]) -> Optional[float]:
    if not rec:
        return None
    v = rec.get("received_at_wall")
    return float(v) if isinstance(v, (int, float)) else None


def arrival_wall_of(segment_rec: dict, ledger_records: Iterable[dict]) -> Optional[float]:
    """这一段的「到达时刻」= **同 `seq` 的 `raw` 记录**的 `received_at_wall`。

    ⚠️ **这是个实测定位出来的坑,不许想当然**(2026-10-03):

    同一次上传会落**两条**留存记录(`raw` 与 `converted`,如 `0003.webm` / `0003_converted.wav`),
    而 `media_retention._record()` 是**每条各采一次 `time.time()`** ⟹ 两条的
    `received_at_wall` **相差 150~283 ms**(95a8 的 8 段实测 8/8 同向:converted 更晚)。

    `reaction_time` 用的是 **raw** 那条 —— 拿 95a8 的活日志反解出来的到达时刻与 raw 记录
    吻合到 **±5 ms**,与 converted 记录差 **+150~283 ms**。所以:

    * **音频字节**用 `converted`(那是过了 16 kHz/单声道/16 bit 闸、端点在用的那份);
    * **时刻**用 `raw` 的。两者混用会让 `reaction_time` 系统性偏晚约 0.2 秒,
      而 0.2 秒在一个 1~65 秒量级的量上**看不出来**。

    找不到同 seq 的 raw 记录时返回 `None`(而不是退回 converted 值)—— 「宁缺勿偏」:
    退回就等于悄悄用了那个已知偏晚的时刻。`reaction_time` 那一组会因此整组不写
    (`reaction_time_features` 收到 `None` 就交 `{}`,不猜、不补 0)。
    """
    seq = segment_rec.get("seq")
    if seq is None:
        return None
    for rec in ledger_records:
        if rec.get("kind") == "raw" and rec.get("seq") == seq:
            return _wall_of(rec)
    return None


def infer_question_index(arrived_wall: Optional[float], questions: Iterable[dict]
                         ) -> Optional[int]:
    """把这一段归到某一题:**最后一个 `ask_end ≤ 到达时刻`** 的题号。

    为什么是**推断**而不是"读出来":活路径**知道**题号(那时它正在处理那一题,
    题号 = 已答对数),而重放手上只有留存记录 —— 题号必须从题台账
    (`questions.jsonl` 的 `ask_end`)与到达时刻推出来。这条推断**不许静默**:
    它是重放腿与活路径唯一的语义差,所以写在这里、由测试钉住、并由实测背书。

    **实测背书**(场 `20260927_212143_95a8`,8 段回答):与活日志的 `question_index`
    **8/8 全对**。复核命令见 `docs/下一步.md`「甲、4」。

    为什么不用"第 k 段 = 第 k 题":那假设了"每题的音频都留下来了、且顺序不乱、
    且没有补推/重录"。按 `ask_end` 对时间不依赖这三条,而且它**与 `reaction_time`
    用的是同一把尺子**(那个量本来就定义为「首次开口 − 该题 `ask_end`」)。

    到达时刻早于所有 `ask_end`(或没有题台账)⟹ 返回 `None`,不猜。
    """
    if arrived_wall is None:
        return None
    hit = [q for q in questions
           if isinstance(q.get("ask_end"), (int, float)) and float(q["ask_end"]) <= arrived_wall]
    if not hit:
        return None
    best = max(hit, key=lambda q: float(q["ask_end"]))
    idx = best.get("index")
    return int(idx) if isinstance(idx, int) else None


def columns_written_by_replay(fieldnames: Iterable[str]) -> list[str]:
    """重放腿**会**写的列 = 契约全部列 − `NOT_WRITTEN_BY_REPLAY`。"""
    return [c for c in fieldnames if c not in NOT_WRITTEN_BY_REPLAY]
