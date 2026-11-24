# report_frontend/report_generator.py

import math
import os
import re
import webbrowser
from html import escape
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
from .data_loader import LogDataLoader
from . import brand, l1_mapper, l1_quality
from .instrument_provenance import INSTRUMENT_UNKNOWN
from .l1_report import build_l1_report


#: 维度数 → 中文数字。只到六维就够(当前最多五维,消失一维后四维);
#: 表外的数原样印阿拉伯数字 —— 宁可印 `7 个维度`,不许印一个错的中文数。
_CN_NUM = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六"}


def _zh_punct(html: str) -> str:
    """把**标签之外**的半角标点换成全角,让中文正文读起来不是拼出来的。

    ## 为什么要这一道(2026-10-26)

    实测那一版正文里 `,` 出现 48 次、`，` 17 次;`:` 20 次、`：` 17 次。
    两种混着,读起来像机器把几段模板拼起来 —— 而报告是给人读的。

    ## 为什么不能直接对整份 HTML 做替换

    `style` / `script` / 标签**内部**的 `,` `:` `;` 是有语法的(CSS 声明分隔、
    `class="a,b"`)。替换了会直接改坏样式与结构。所以先按标签切开,
    只对文本节点替换。

    ## 边界

    * 只换 `,` `;` `:` —— 括号不动:代码式的 `(3 / 3)` 与中文式的
      「(见右端)」在正文里本来就该按上下文选,统一换反而更怪。
    * `:` 后面紧跟数字的不换(`10:29` 是时间,不是标点)。
    * 已经全角的原样留着(幂等)。
    """
    def _fix(text: str) -> str:
        text = text.replace(",", "，").replace(";", "；")
        # 时间/版本号里的冒号不动:`10:29`、`l1:1`。
        return re.sub(r":(?![0-9])", "：", text)

    # 标签(含 style/script 整块)原样保留,只改中间的文本节点。
    out: list[str] = []
    pos = 0
    for m in re.finditer(r"(?s)<(script|style)\b.*?</\1\s*>|<[^>]+>", html):
        out.append(_fix(html[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(_fix(html[pos:]))
    return "".join(out)


def _fmt_num(v) -> str:
    """实测值 → 给人读的字符串。**小数位数按量级选,不一律两位。**

    实测产出过 `3.4175334323922733`(float64 的内存映像)。前端
    `L1IndicatorTable` 的 `fmtValue` 早就有一条按量级的规则,HTML 报告是
    同一个数的另一个出口,不该有两套精度:

    * ≥1000 取整;≥1 三位;≥0.001 四位;更小用科学计数。

    ⚠️ **只在渲染处收**。载荷里的 `raw_value` 保持原始 float ——
    前端要拿它排序/比较,在那里四舍五入就是丢测量结果。
    钉子:`tests/test_report_numbers_and_reasons_are_readable.py`。
    """
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(x):
        return "—"
    if x == 0:
        return "0"
    a = abs(x)
    if a >= 1000:
        return f"{x:.0f}"
    if a >= 1:
        return f"{x:.3f}"
    if a >= 0.001:
        return f"{x:.4f}"
    return f"{x:.2e}"


def _interval_text(interval) -> str:
    """本场会话内观测区间。拿不到就说清是缺什么,而不是留空。

    区间是 `原始值 ± 会话内标准差`(与原始值同量纲、来自本场会话自己的测量),
    不是人群中位置 —— spec §5.5 禁止任何暗示人群位置的区间。
    """
    if not interval:
        return "未采集到会话内变异信息"
    return f"{interval[0]} – {interval[1]}"


def _fmt_compact(v) -> str:
    """**图表标签**用的短数字 —— 末尾的 0 去掉,位数按量级选。

    与 `_fmt_num` 的区别只在"要不要保留定长小数":表格里定长便于纵向对齐,
    而图上的直接标注是散文式的(`0.25` 比 `0.2500` 好读,`3` 比 `3.000` 好读)。

    ⚠️ 同样**只用于渲染**,载荷里的 `raw_value` 保持原始 float。
    """
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(x):
        return "—"
    if x == 0:
        return "0"
    a = abs(x)
    if a >= 1000:
        s = f"{x:.0f}"
    elif a >= 10:
        s = f"{x:.1f}"
    elif a >= 1:
        s = f"{x:.2f}"
    elif a >= 0.01:
        s = f"{x:.3f}"
    elif a >= 0.0001:
        s = f"{x:.4f}"
    else:
        return f"{x:.2e}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _zero_note(item: Dict[str, Any]) -> str:
    """一个 **0 值**旁边那句解释。非 0 或未观测返回空串；按 5 类指标语义精确选句。"""
    msg = l1_quality.explain_zero(
        str(item.get("feature") or ""),
        item.get("raw_value"),
        item.get("n_valid"),
        item.get("n_total"),
    )
    if not msg:
        return ""
    return f'<span class="zero-note">{escape(msg)}</span>'


def _instrument_cell(item: Dict[str, Any],
                     labels: Optional[Dict[str, str]] = None) -> str:
    """一行指标 → 「仪器来源」那一格(A5,2026-11-21)。

    ## 为什么这一格必须存在

    本系统的面部测量**同时有两套仪器**:手写的关键点几何代理(`au12_smile` 这类比率)
    与 FaceLandmarker 模型回归出的 52 列系数(`bs_mouthSmileLeft`)。离线实测
    (32 场 / 389 条)里,12 个构念只有 1 个达 ρ ≥ 0.7 ⟹ 两套数**不可混用**。
    不标出处,读者会把 `au12_smile` 与 `bs_mouthSmile` 当成同一件事的两次测量。

    ## 三件它**不做**的事

    1. **不查表、不猜**:只印载荷里已经标好的 `instruments`(来源见
       `l1_quality.build_quality_report` 那个装配点)。渲染层再查一次,
       两处口径就会分叉。
    2. **不把「未登记」画成一种仪器**:它是**披露缺口**,所以单独一个样式
       (`.prov-unknown`,虚线下划 + 警示色)——与「几何代理」并排成同款灰字,
       读者会把它当成第三套仪器读过去。
    3. **不把"载荷没带"说成"未登记"**:旧引擎(`research_mapper`)的 `evidence_chain`
       里根本没有这个键。前者是这一份载荷没披露,后者是披露了但登记表里查不到 ——
       两件事印成同一句,就是把后者说成了前者。
    """
    names = item.get("instruments")
    if not names:
        # 审计链那些行(`l1.indicators`)是**单值/多值二选一**的形态
        # (`instrument` XOR `instruments`,由 `annotate_indicators` 决定)——
        # 两种都认,读到的是同一个出处。见 `instrument_provenance` docstring「形状」。
        single = item.get("instrument")
        names = [single] if isinstance(single, str) and single else None
    if not names:
        return '<span class="prov-missing">未提供</span>'
    parts: list[str] = []
    for name in names:
        if not isinstance(name, str):
            continue
        # 没登记中文标签的就露英文键 —— 宁可露出英文,也不许吞掉这一条。
        text = (labels or {}).get(name, name)
        if name == INSTRUMENT_UNKNOWN:
            parts.append(
                f'<span class="prov-unknown" title="披露缺口：依赖列还没归到任何一路仪器上">'
                f'{text}</span>')
        else:
            parts.append(text)
    return "、".join(parts) if parts else '<span class="prov-missing">未提供</span>'


#: 模态键 → 报告里给人看的名字(键名来自 data_loader 的统一化,见那边的 key 映射)。
#:
#: ⚠️ `voice` 这一项是**模态级**名字,与上面两个文件名不同:日志按「面试 / 科研」
#:    分两个文件,而 L1 的 V 族不分。`modality_summary` 用的是模态名,所以要有它。
_MODALITY_LABELS = {
    "face": "面部",
    "gesture": "手势",
    "voice": "语音",
    "text": "转写文本",
    "voice_interview": "语音（面试）",
    "voice_research": "语音（科研）",
}

#: 日志键 → L1 的**族模态名**(`l1_quality.FAMILY_MODALITY` 用的是后者)。
#: 两者不同名是因为日志按"面试/科研"分文件,而 L1 的 V 族不分 —— 同一套指标。
#: 这张表只用于把载荷里的 `modality_summary` 对上号,不做别的判断。
_LOG_KEY_TO_MODALITY = {
    "voice_interview": "voice",
    "voice_research": "voice",
}


_STAGE_BADGE_CLASSES = {
    "stage_1_verified": "stage-verified",
    "stage_2_review": "stage-review",
    "stage_3_limited": "stage-limited",
}
_STAGE_DEFAULT_LABELS = {
    "stage_1_verified": "阶段 I · 证据充分可核验",
    "stage_2_review": "阶段 II · 证据可用需复核",
    "stage_3_limited": "阶段 III · 证据受限待补采",
    "unavailable": "未获取有效采集数据",
}
_MACRO_COLOR_CLASS = {
    "steady": "steady",
    "steady_speaking": "steady",
    "active": "active",
    "active_expressive": "active",
    "pause": "pause",
    "pause_thinking": "pause",
    "low_signal": "low",
    "low": "low",
}
_MACRO_DEFAULT_LABELS = {
    "steady": "平稳展开期",
    "active": "高互动活跃期",
    "pause": "停顿组织期",
    "low": "采集受限期",
}
_POSE_MODE_META = {
    "3d_matrix": ("三维姿态矩阵校准", "pill-ok"),
    "2d_legacy": ("二维姿态代理校准", "pill-warn"),
    "unavailable": ("无姿态参照", "pill-neutral"),
}
_AGREEMENT_PILL = {
    "高度一致": "pill-ok",
    "中度一致": "pill-tag",
    "分歧较大": "pill-warn",
    "通道未激活": "pill-neutral",
}
_ELIGIBILITY_PILL = {
    "已校准可融合": "pill-ok",
    "辅助对照": "pill-tag",
    "待3D重采验证": "pill-warn",
}
_V2_TRACK_LABELS = {
    "face_kinematics": "面部运动学",
    "voice_dynamics": "语音动力学",
    "gesture_smoothness": "上肢平滑度",
    "catch22_dynamics": "多通道时序",
}


def _finite_float(v: Any) -> Optional[float]:
    """安全转为有限浮点数，缺失或非有限值返回 None。"""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _sanitize_plain(text: Any) -> str:
    """清理载荷说明文字中的注释体记号与推理符号，保留自然中文语义。"""
    t = str(text or "")
    t = re.sub(r"\*\*|`", "", t)
    t = t.replace("⟹", "，所以").replace("⇒", "，所以")
    t = t.replace(" → ", " 变成 ").replace("→", " 变成 ")
    return re.sub(r"[，,]{2,}", "，", t)


def _clean_prose(text: Any) -> str:
    """清理渲染到交付物正文的中文说明，替换直引号并消除内部符号。"""
    t = _sanitize_plain(text)
    t = re.sub(r'"([^"]*)"', r"「\1」", t)
    return t.replace('"', "")


def _fmt_v2_highlight(v: Any, unit: str) -> str:
    """第二层动力学指标读数格式化：比例按百分比展示，其余按量级控制小数位。"""
    x = _finite_float(v)
    if x is None:
        return "—"
    if unit == "比例":
        return f"{x * 100:.1f}%"
    a = abs(x)
    if a >= 100:
        num_str = f"{x:.1f}"
    elif a >= 1:
        num_str = f"{x:.2f}"
    elif a >= 0.001 or x == 0:
        num_str = f"{x:.3f}" if x != 0 else "0"
    else:
        num_str = f"{x:.2e}"
    return f"{num_str} {escape(unit)}" if unit else num_str


def _fmt_rho(v: Any) -> str:
    """相关系数或差值格式化为三位小数，非有限值返回占位符。"""
    x = _finite_float(v)
    return f"{x:.3f}" if x is not None else "—"


#: 交付物的样式表。**与正文分开** —— 正文是 f-string,里面每个 `{` 都要转义,
#: 把 CSS 单独放一份就不用给几十个大括号加反斜杠。
#:
#: 配色按 dataviz 的规矩来:一个蓝色系列 + 墨色文字,状态色不参与系列着色;
#: 明暗两套**各自取步**(`#3987e5` / `#d95926` 是按深色底重新验过的,
#: 不是把浅色那套自动翻转)。文字只用墨色三档,不穿系列色 —— 读者不该
#: 靠颜色判断哪句话是"数据"。
_CSS = """
:root{
  color-scheme: light dark;
  --page:#f9f9f7; --surface:#fcfcfb;
  --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#898781;
  --rule:#e1e0d9; --track:#e8e7e0;
  --series-1:#2a78d6; --series-2:#eb6834;
  /* A5 仪器来源披露:「未登记」是**缺口**,不是一种仪器,所以给它自己的警示色,
     不与正文的灰字同色 —— 同色就会被当成第三套仪器读过去。 */
  --alert:#a8332b;
  --stage-steady:#2a78d6; --stage-active:#0b5d2e;
  --stage-pause:#b26b00; --stage-low:#898781;
  --ok-fg:#0b5d2e; --ok-bg:#edf7f0; --ok-bd:#bfe3cb;
  --warn-fg:#7a4a05; --warn-bg:#fdf6e9; --warn-bd:#eed39e;
  --lim-fg:#9e1f1f; --lim-bg:#fdf0f0; --lim-bd:#f2c2c2;
  --info-fg:#0a4a8f; --info-bg:#eef5fc; --info-bd:#c2dcf5;
  --attr-pos:#2a78d6; --attr-neg:#eb6834;
}
@media (prefers-color-scheme: dark){
  :root{ --page:#141413; --surface:#1b1b19;
         --ink:#f3f2ed; --ink-2:#bab8b1; --ink-3:#8c8a84;
         --rule:#302f2b; --track:#2a2926;
         --series-1:#3987e5; --series-2:#d95926;
         --alert:#ef8a80;
         --stage-steady:#3987e5; --stage-active:#34a853;
         --stage-pause:#e59a38; --stage-low:#898781;
         --ok-fg:#6ee7a0; --ok-bg:#142a1d; --ok-bd:#245337;
         --warn-fg:#f3b863; --warn-bg:#2b2112; --warn-bd:#5e431c;
         --lim-fg:#f28b82; --lim-bg:#2e1717; --lim-bd:#612927;
         --info-fg:#7eb8f7; --info-bg:#132336; --info-bd:#25466b;
         --attr-pos:#3987e5; --attr-neg:#d95926; }
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
  font-family:"Noto Sans SC","Microsoft YaHei",system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:15px;line-height:1.75;-webkit-font-smoothing:antialiased}
.wrap{max-width:860px;margin:0 auto;padding:48px 24px 72px}
header{border-bottom:1px solid var(--rule);padding-bottom:20px;margin-bottom:32px}
h1{font-size:24px;line-height:1.3;margin:0 0 10px;font-weight:650;letter-spacing:.01em}
.meta{color:var(--ink-3);font-size:13px}
/* 报告头那块「这一场是谁的」。信息比样式重要:它要在一眼扫过时被读到,
   所以给底色和边框,而不是并进 .meta 那行灰字里。 */
.subject{display:flex;flex-wrap:wrap;align-items:baseline;gap:2px 8px;
  margin:0 0 10px;padding:9px 14px;border:1px solid var(--rule);
  border-radius:8px;background:var(--surface);font-size:14px}
.subject-k{color:var(--ink-3);font-size:12.5px}
.subject-v{font-weight:600;margin-right:14px}
.subject-v:last-child{margin-right:0}
.subject-missing{display:block;color:var(--ink-2);font-size:13px;
  border-style:dashed;background:transparent}
.subject-missing code{font-size:12px}
.sources{margin-top:14px;font-size:13px;color:var(--ink-2)}
h2{font-size:17px;font-weight:650;margin:44px 0 14px}
h3{font-size:15px;font-weight:600;margin:26px 0 8px}
p{margin:0 0 12px}
.lead{font-size:16.5px;line-height:1.7;max-width:44em;margin-bottom:22px}
.hero{display:flex;align-items:baseline;gap:14px;margin:0 0 10px;flex-wrap:wrap}
.cover-number{font-size:52px;font-weight:700;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;line-height:1}
.hero-cap{color:var(--ink-2);font-size:14px}
.coverage-line{font-size:14.5px;line-height:1.6;margin:2px 0 6px;max-width:52em;
  font-variant-numeric:tabular-nums}
.coverage-why{color:var(--ink-3);font-size:12.5px;line-height:1.65;max-width:52em;
  margin:0 0 4px}
figure{margin:18px 0 6px}
figcaption{color:var(--ink-3);font-size:12.5px;margin-top:10px}
svg{display:block;width:100%;height:auto;overflow:visible}.label{fill:var(--ink-2);font-size:12.5px}
.value{fill:var(--ink);font-size:12.5px;font-weight:600;font-variant-numeric:tabular-nums}
.tick{fill:var(--ink-3);font-size:11px;font-variant-numeric:tabular-nums}
.track{fill:var(--track)}
.bar{fill:var(--series-1)}
.range{stroke:var(--ink-3);stroke-width:2;stroke-linecap:round;opacity:.5}
.median{fill:var(--ink)}
.flat-note{fill:var(--ink-3);font-size:11px}
.axis{stroke:var(--rule);stroke-width:1}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0 16px}
th,td{text-align:left;padding:7px 12px 7px 0;border-bottom:1px solid var(--rule);
  vertical-align:top}
th{color:var(--ink-3);font-weight:600;font-size:12.5px}
td.num{font-variant-numeric:tabular-nums}
.zero-note{display:block;color:var(--ink-3);font-size:12px;margin-top:2px}
.note{color:var(--ink-2);font-size:13px}
.gaps{color:var(--ink-2);font-size:13.5px}
.gaps ul,.l1-audit ul{margin:6px 0 16px;padding-left:20px}
.l1-audit{border-top:1px solid var(--rule);padding-top:14px;margin-top:22px;
  font-size:13.5px;color:var(--ink-2)}
.dim + .dim{border-top:1px solid var(--rule);padding-top:14px;margin-top:22px}
details{border-top:1px solid var(--rule);margin-top:40px;padding-top:16px;
  font-size:13.5px;color:var(--ink-2)}
summary{cursor:pointer;color:var(--ink-2);font-weight:600;font-size:14px}
footer{margin-top:52px;padding-top:16px;border-top:1px solid var(--rule);
  color:var(--ink-3);font-size:12.5px}
@media (max-width:560px){.wrap{padding:28px 16px 48px}.cover-number{font-size:40px}
  table{font-size:12.5px}}
/* 手机上整张 SVG 缩到 400px 宽,图内文字会跟着缩到 ~7px —— 看不清。
   把图放进可横向滚动的容器:图保持自己的宽度,读者横向滑动看,文字不缩水。
   只让图滚,页面主体不许横向滚。 */
figure{overflow-x:auto}
figure svg{min-width:560px}
/* ── A5 仪器来源披露(2026-11-21)──────────────────────────────────────────
   表格最后一列:每条指标由哪一路仪器产出。
   「未登记」用虚线 + 警示色,与其它来源**不同形** —— 它是披露缺口,不是第三种仪器;
   排成同款灰字,读者会把它当成一套仪器读过去。 */
.prov{color:var(--ink-2);font-size:12.5px;white-space:nowrap}
.prov-unknown{color:var(--alert);font-weight:600;
  border-bottom:1px dashed var(--alert);cursor:help}
.prov-missing{color:var(--ink-3)}
/* 页脚汇总:一枚一枚的小标签 + 一句由载荷给的汇总。 */
#instrument-disclosure .prov-chips{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 14px}
.prov-chip{font-size:12.5px;color:var(--ink-2);border:1px solid var(--rule);
  border-radius:999px;padding:2px 11px;font-variant-numeric:tabular-nums}
.prov-chip-unknown{color:var(--alert);border-color:var(--alert);border-style:dashed}
.prov-footer{font-size:13.5px;color:var(--ink);font-variant-numeric:tabular-nums}
.prov-warn{border-left:3px solid var(--alert);padding:4px 0 4px 12px;
  color:var(--ink-2);font-size:13px}
.prov-warn code{font-size:12px}
/* ── referee_ml & v2 appendices ─────────────────────────────────────────── */
.referee-card{background:var(--surface);border:1px solid var(--rule);
  border-radius:10px;padding:18px 20px;margin:14px 0 24px}
.referee-head{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:10px}
.stage-badge{display:inline-flex;align-items:center;padding:2px 10px;
  border-radius:999px;font-size:12.5px;font-weight:600;border:1px solid var(--rule);
  color:var(--ink-2);background:var(--page)}
.stage-verified{color:var(--ok-fg);background:var(--ok-bg);border-color:var(--ok-bd)}
.stage-review{color:var(--warn-fg);background:var(--warn-bg);border-color:var(--warn-bd)}
.stage-limited{color:var(--lim-fg);background:var(--lim-bg);border-color:var(--lim-bd)}
.stage-info{color:var(--info-fg);background:var(--info-bg);border-color:var(--info-bd)}
.cov-chips{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 14px}
.cov-chip{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;
  border-radius:999px;font-size:12.5px;color:var(--ink-2);background:var(--page);
  border:1px solid var(--rule);font-variant-numeric:tabular-nums}
.cov-chip-warn{color:var(--warn-fg);background:var(--warn-bg);border-color:var(--warn-bd);font-weight:600}
.cov-chip-alert{color:var(--lim-fg);background:var(--lim-bg);border-color:var(--lim-bd)}
.attr-pos{fill:var(--attr-pos);color:var(--attr-pos)}
.attr-neg{fill:var(--attr-neg);color:var(--attr-neg)}
.stage-legend{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 6px}
.stage-swatch{display:inline-block;width:9px;height:9px;border-radius:2px;flex-shrink:0}
.stage-swatch-steady{background:var(--stage-steady)}
.stage-swatch-active{background:var(--stage-active)}
.stage-swatch-pause{background:var(--stage-pause)}
.stage-swatch-low{background:var(--stage-low)}
.stage-seg-steady{fill:var(--stage-steady)}
.stage-seg-active{fill:var(--stage-active)}
.stage-seg-pause{fill:var(--stage-pause)}
.stage-seg-low{fill:var(--stage-low)}
.cp-diamond{fill:var(--alert)}
.cp-line{stroke:var(--alert);stroke-width:1.6;stroke-dasharray:3 2}
.cp-halo{stroke:var(--surface);stroke-width:3;stroke-opacity:.85}
.pill-tags{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 10px}
.pill-tag{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;
  color:var(--info-fg);background:var(--info-bg);border:1px solid var(--info-bd)}
.pill-neutral{color:var(--ink-2);background:var(--page);border-color:var(--rule)}
.pill-ok{color:var(--ok-fg);background:var(--ok-bg);border-color:var(--ok-bd)}
.pill-warn{color:var(--warn-fg);background:var(--warn-bg);border-color:var(--warn-bd)}
.diag-box{margin:10px 0 14px;padding:9px 12px;border-radius:6px;
  background:var(--page);border:1px solid var(--rule);color:var(--ink-2);font-size:12.5px}
.delta-pos{color:var(--ok-fg);font-weight:600}
"""


def _subject_block(session_id: Optional[str]) -> str:
    """报告头里的「这一场是谁的」—— 姓名 / 学号 / 院系 / 场次序号。

    ## 为什么要有它(2026-10-26,使用者原话)

    「要求显示出被测人的个人信息,不然看报告分辨不出」。报告此前只印会话号
    `YYYYMMDD_HHMMSS_xxxx` —— 一串机器号。同一批录十几个人,十几份报告在盘上
    长得一模一样,人得挨份打开才知道哪份是谁的。

    这些字段在**录制当时**就写进了会话目录的 `label.json`
    (`session_meta.upsert_label`,前端「开始录制」前填的那一屏),这里只负责读出来
    印在头里 —— 报告端**不另存一份身份**,避免两处各说各话。

    ## 三种状态都要说明白,不留白

    * 有标注   → 逐字段印出来(空字段整段不出现,不留孤零零的「院系:」)
    * 没标注   → 明说「未标注」:空白读起来像"读取失败",而"没填"与"读坏了"是两件事
    * 读坏了   → 明说「读取失败」,并指到 `label.json` —— 别猜成"没标注"

    标注是**客户端可控**的文本(走 HTTP 表单),这里一律 `escape` 后再拼 HTML。
    """
    if not session_id:
        return ""
    try:
        # 延迟导入:报告生成不依赖素材目录也能跑(测试/单文件复算时没有 recordings)。
        from session_meta import read_label
        label = read_label(session_id)
    except Exception:
        return ('<div class="subject subject-missing">被测者：'
                '<strong>读取失败</strong>（会话目录里的身份标注文件 <code>label.json</code> 读取异常，'
                '请检查原始录制目录）</div>')
    if not label:
        return ('<div class="subject subject-missing">被测者：<strong>未标注</strong>'
                '（录制时没有填写姓名 / 学号 / 院系 —— 这份报告没法按人对号）</div>')

    fields = [("被测者", label.get("name")),
              ("学号", label.get("student_id")),
              ("院系", label.get("department")),
              ("场次", label.get("serial"))]
    vals = [(k, str(v).strip()) for k, v in fields if v and str(v).strip()]
    if not vals:
        return ('<div class="subject subject-missing">被测者：<strong>未标注</strong>'
                '（录制时没有填写姓名 / 学号 / 院系 —— 这份报告没法按人对号）</div>')
    pairs = "".join(
        f'<span class="subject-k">{escape(k)}</span>'
        f'<span class="subject-v">{escape(v)}</span>'
        for k, v in vals)
    return f'<div class="subject">{pairs}</div>'


def session_started_at(session_id: Optional[str]) -> Optional[datetime]:
    """这一场**开录**的时刻 —— 会话 id 自带 `YYYYMMDD_HHMMSS`(`<日期>_<时刻>_<4位hex>`)。

    报告里要印的「测试时间」就是它:由铸号那一刻决定,与日志文件名、录制目录名同一串
    数字。**不拿报告生成时刻冒充**,也不用文件 mtime(那个会随拷贝/解压变化)。

    解析不了(旧 id、外来 id)返回 `None`,调用方据此整段不印 —— 不编一个时间。
    """
    if not session_id or len(session_id) < 15:
        return None
    try:
        return datetime.strptime(session_id[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _meta_line(session_id: Optional[str], version: Any) -> str:
    """报告头的时间行:测试时间 + 报告生成时间 + 报告口径版本。

    两个时间说的是两件事,必须都印:同一场会话隔几天重跑一次,「测试时间」不变、
    「报告生成时间」变 —— 只印后者会让人把"重跑"当成"又测了一次"
    (2026-10-26 使用者要求)。
    """
    parts = []
    started = session_started_at(session_id)
    if started is not None:
        parts.append(f"测试时间 {started.strftime('%Y-%m-%d %H:%M:%S')}")
    parts.append(f"报告生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    parts.append(f"报告口径版本 {escape(str(version))}")
    return " · ".join(parts)


def _subject_slug(session_id: Optional[str]) -> str:
    """报告文件名里的 `-姓名` 后缀(2026-10-26 使用者:列表里分不出谁是谁)。

    只取**姓名**,不放学号 —— 文件名会出现在列表、下载名、`ls` 里,学号是更敏感的那
    一半;要看学号,报告正文里有。没有标注(或读不出来)就返回空串:宁可不带,
    也不拿 `None` / 会话号充数。
    """
    if not session_id:
        return ""
    try:
        from session_meta import read_label
        label = read_label(session_id) or {}
    except Exception:
        return ""
    name = str(label.get("name") or "").strip()
    # 文件名里非法的字符(Windows 全套 + 控制字符)一律去掉;限长防止撑爆路径。
    safe = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", name).strip()
    return f"-{safe[:30]}" if safe else ""


def _unique_report_path(output_dir: str, stem: str,
                        session_id: Optional[str]) -> str:
    """`<stem>.html` 已被占用时换一个**不会盖掉另一场**的名字。

    文件名里的时刻只到秒,而「全部重跑」/批处理会在一秒内连跑好几场 —— 此前直接写
    同一个路径,后一场**静默覆盖**前一场(2026-10-26 重跑实测:三场只剩一份)。
    撞了就加会话 id 末四位(它唯一),还撞就再序号;宁可有后缀,不许覆盖。
    """
    base = os.path.join(output_dir, f"{stem}.html")
    if not os.path.exists(base):
        return base
    tail = (session_id or "")[-4:] or "x"
    alt = os.path.join(output_dir, f"{stem}-{tail}.html")
    n = 2
    while os.path.exists(alt):
        alt = os.path.join(output_dir, f"{stem}-{tail}-{n}.html")
        n += 1
    return alt


def sources_disclosure(sources: Dict[str, Dict], target: Optional[str] = None,
                       modality_summary: Optional[Dict[str, Dict]] = None) -> str:
    """披露「本报告描述的是哪一场、每个模态进来了没有」(M2 spec §5.2)。

    输入是 `LogDataLoader.selected_sessions`,值是三态之一:
    `loaded` / `unreadable`(文件在但读不出)/ `missing`(本场没有);
    另有一个特殊键 `none_bucket`(没带 session_id 的行,不进聚合但要说一声)。

    **为什么"未读到数据"必须显式出面**:I4 的实现只列真正读出来的模态,于是读不出来的
    那些**静默消失**,而报告还写着「以上模态来自同一场会话」—— 2026-09-24 实测:语音整个
    没了,读者却以为一切正常(spec §3.3)。

    同场/不同场那句话**已取消**:按 id 选之后不存在"不同场"这个状态,真正要报的是
    "哪几个模态没进来、为什么"。

    `target` 是"本报告以哪一场为准",由调用方给(`LogDataLoader.target_session` /
    实时路径的入参,第 17 条)。**为什么不从 items 里推第一个**:只有 NONE 桶、或连
    NONE 桶都没有时,items 是空的,推不出来 —— 而那正是最需要把话说清楚的情形。
    没给且 items 非空时退回"取第一个"(兼容直接调用本函数的调用方与测试)。

    ## `modality_summary`(2026-10-25 补):日志三态**不是全部事实**

    三态描述的是**日志文件**。语音还有第二个数据源:整场录音逐帧重算
    (`voice_series`),不经日志。实测场 `20260928_180131_8102` 语音日志 0 行,
    报告头说「未读到数据(…若本场确实答过题,那才是文件损坏)」,而正文里
    V 族 12 条有 10 条有值 —— 同一个模态,两个出口说了相反的话。

    载荷 (`l1.modality_summary`) 给出每个模态到底算出了几条,这里**照抄**:
    有值 ⟹ 说「N/M 条有值」,不再提"损坏";一条都没有 ⟹ 保留原措辞(那是真的)。
    渲染层不自己数 —— 它拿不到族与模态的归属(`l1_quality.FAMILY_MODALITY`)。
    """
    none_bucket = sources.get("none_bucket")
    items = {k: v for k, v in sources.items() if k != "none_bucket"}
    if target is None and items:
        target = next(iter(items.values()))["session_id"]

    summary = modality_summary or {}

    def _produced(key: str) -> Optional[Dict]:
        """这个模态**算出了几条** —— 日志键 `voice_interview` 归到 `voice` 这一族。

        没登记就返回 `None`(不猜)。判据只有这一处,两个分支共用。
        """
        fam = _LOG_KEY_TO_MODALITY.get(key, key)
        return summary.get(fam)

    def _file_status(key: str, info: Dict) -> str:
        """**一个日志文件**的三态 —— 逐文件的事实,措辞与原来一致。"""
        label = _MODALITY_LABELS.get(key, key)
        if info["status"] == "loaded":
            return f"{label} · 已读入（{info['rows']} 行）"
        if info["status"] == "unreadable":
            return f"{label} · 未读到数据（这个文件里一行数据都没有）"
        return f"{label} · 缺失"

    def _file_detail(key: str, info: Dict) -> str:
        """文件层面的**中性**描述 —— 只在"指标由第二个数据源算出来"时用。

        为什么不复用 `_file_status`:那句里的「未读到数据」会让读者以为这一维废了,
        而紧挨着的上一句刚说「算出了 10/12 条」。同一段里一冷一热,读者只会记住冷的那句。
        这里只陈述可核对的事实:这个文件几行、在不在。
        """
        label = _MODALITY_LABELS.get(key, key)
        if info["status"] == "loaded":
            return f"{label} · {info['rows']} 行"
        if info["status"] == "unreadable":
            return f"{label} · 0 行（文件在）"
        return f"{label} · 没有这个文件"

    def _produced_line(keys: list[str]) -> Optional[str]:
        """这个**模态**算出了几条 —— 按模态说**一次**,不按文件说 N 次。

        ## 为什么必须按模态去重

        日志按「面试 / 科研」分两个文件键,而 L1 的 V 族**不分** ——
        12 条指标是同一套,`modality_summary` 只有一个 `voice`。
        逐文件印就会变成「面试语音算出了 10/12」+「科研语音算出了 10/12」,
        读起来像两个模态各有 10 条,而实际是**同一批 10 条被数了两遍**。

        文件名也不能当标签:V 族不区分面试/科研,挂着文件名会让人以为两者各有 12 条。

        文件级的事实不丢 —— 附在括号里(哪份在、哪份没有),只是不再冒充模态级结论。
        """
        got = None
        for k in keys:
            got = _produced(k)
            if got:
                break
        if not got or not got.get("n_measured"):
            return None
        fam = _LOG_KEY_TO_MODALITY.get(keys[0], keys[0])
        label = _MODALITY_LABELS.get(fam, fam)
        files = "、".join(_file_detail(k, items[k]) for k in keys)
        note = f"（{got['source_note']}）" if got.get("source_note") else ""
        return (f"{label} · <strong>日志没有数据，但这个模态算出了 "
                f"{got['n_measured']} / {got['n_total']} 条指标</strong>{note}"
                f"<br><span style=\"opacity:0.8;\">　文件层面：{files}</span>")

    lines = (["本场会话：<strong>（无 —— 本场没有任何日志）</strong>"] if target is None
             else [f"本场会话：<strong>{target}</strong>"])
    if not items:
        lines.append("本报告没有装配任何模态日志。")

    # 按**模态**分组 —— 语音的两个文件键归一组(理由见 `_produced_line`)。
    groups: Dict[str, list[str]] = {}
    for key in sorted(items):
        groups.setdefault(_LOG_KEY_TO_MODALITY.get(key, key), []).append(key)

    for _fam, keys in groups.items():
        if any(items[k]["status"] == "loaded" for k in keys):
            # 这个模态的日志读进来了 ⟹ 逐文件报事实(两份文件可能一有一无)。
            lines.extend(_file_status(k, items[k]) for k in keys)
            continue

        got = _produced_line(keys)
        if got:
            # 日志全空,但指标由第二个数据源算出来了 —— 别让读者带着
            # 「文件可能损坏」的印象往下读(实测缺陷,见 docstring)。
            lines.append(got)
            continue

        # 日志全空、指标也一条没算出来 ⟹ 保留原措辞。那是真的:该去查采集。
        for k in keys:
            label = _MODALITY_LABELS.get(k, k)
            if items[k]["status"] == "unreadable":
                # ⚠️ 措辞别冤枉文件。此前这句写的是"文件在，但读不出来：空文件或损坏",
                #    而最常见的情形**根本不是损坏** —— 是这一场压根没有提交过回答
                #    (从「实时分析」页录的那些就是:那条路没有题库、不提交回答,
                #    语音模态因此结构上为空,交接日志 §8.1-4 记过)。两件事的处置完全不同:
                #    一个是"设计如此",一个是"去查文件"。分不开就会让人去追一个不存在的故障。
                lines.append(f'<strong style="color:#A23B72;">{label} · 未读到数据</strong>'
                             f'（日志文件在，但一行数据都没有。常见原因是本场'
                             f'<strong>没有提交过逐题回答</strong> —— 例如从「实时分析」页录的，'
                             f'那条路没有题库；若本场确实答过题，那才是文件损坏。）')
            else:
                lines.append(f"{label} · 缺失（本场没有这个模态的日志）")

    if none_bucket:
        # ⚠️ 2026-10-26:`NONE 桶` 是内部叫法(所有没带 session_id 的请求都落在这个
        #    `NONE` 目录里),读者不知道桶是什么。说清**发生了什么**就够:
        #    有 N 行日志没带会话编号 ⟹ 归不到任何一场 ⟹ 没算进本场。
        lines.append(f'<em>另有 {none_bucket["rows"]} 行日志没带会话编号 —— '
                     f'归不到任何一场，因此没算进本场。</em>')

    return "<br>".join(lines)


def live_sources(session_id: str, data: Dict[str, Any]) -> Dict[str, Dict]:
    """实时路径的数据来源:模态 → 三态条目(与批处理路径**同一个形状**)。

    实时路径是按**这一个 `session_id`** 去内存里取的(不像批处理要在磁盘上选文件),
    所以凡取到数据的模态都是 `loaded`。

    ⚠️ 形状必须与批处理一致。这条曾经漏改(2026-09-24 复审 C1):旧形状是
    `{k: session_id}`,而 `sources_disclosure` 会做 `info["status"]` → `TypeError`,
    **而面板照样报 success** —— 又一个静默失败,且全仓没有测试碰过这条路径。
    """
    return {k: {"session_id": session_id, "status": "loaded", "rows": len(v)}
            for k, v in data.items()}


class ReportGenerator:
    """
    行为观测报告生成器

    报告只陈述本次实际测到了什么:每个维度按 spec §5.4 的「值 + 有效样本量 + 置信度 +
    evidence_gaps」渲染,聚合只给「区间 + 置信度 + 依据」(spec §5.6)。
    **不产出对候选人的评分或评级** —— 未标定标尺上的复合点分与五档评语一律不渲染。
    不解读、不推断、不做形容词修饰。
    """

    def __init__(self, output_dir: str = "data/output"):
        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_report(
        self,
        session_id: Optional[str] = None,
        *,
        scene_profile: Optional[str] = None,
    ) -> str:
        """从磁盘 CSV 文件生成评估报告（批处理模式）。

        `session_id`:本报告要描述的那一场(M2)。**不给就取最新一场**,并在报告头写明是谁
        —— 这个参数在 M2 之前是"收下就丢"的。
        """
        print("\n" + "=" * 70)
        print(f"🚀 启动 {brand.BRAND_NAME} {brand.BRAND_SUBTITLE} (批量模式)")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = ""

        try:
            loader = LogDataLoader()
            data = loader.get_fused_latest_data(session_id)
            if not data:
                print("   ⚠️  本场没有任何可用日志,仍生成一份只含缺口的报告")

            stem = (f"{brand.REPORT_FILENAME_PREFIX}_{timestamp}"
                    f"{_subject_slug(loader.target_session)}")
            report_path = _unique_report_path(self.output_dir, stem, loader.target_session)

            if scene_profile is not None:
                result = build_l1_report(loader.target_session, logs=data, scene_profile=scene_profile)
            else:
                result = build_l1_report(loader.target_session, logs=data)
            static_images = self._scan_static_images()

            html_content = self._build_html_report(result, {}, {}, data,
                                                   static_images,
                                                   sources=loader.selected_sessions,
                                                   target=loader.target_session)

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

        except Exception as e:
            print(f"❌ 错误：{e}")
            return ""

        if os.environ.get("JX_NO_BROWSER") != "1":
            try:
                webbrowser.open('file://' + os.path.realpath(report_path))
            except Exception as e:
                print(f"   ⚠️  报告已生成,但打开浏览器失败：{e}")
        print(f"REPORT_PATH={os.path.realpath(report_path)}")
        print(f"\n✅ 报告已生成：{report_path}")
        return report_path

    def generate_report_live(self, session_id: str) -> str:
        """从运行中的 API 服务获取实时内存数据，生成证据披露报告（实时模式）"""
        print("\n" + "=" * 70)
        print(f"🚀 启动 {brand.BRAND_NAME} {brand.BRAND_SUBTITLE} (实时模式)")
        print(f"📋 会话 ID: {session_id}")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = (f"{brand.REPORT_FILENAME_PREFIX}_Live_{timestamp}"
                f"{_subject_slug(session_id)}")
        report_path = _unique_report_path(self.output_dir, stem, session_id)

        try:
            loader = LogDataLoader()
            data = loader.get_live_data(session_id)
            if not data:
                raise ValueError("无法从 API 获取实时数据，请确认三个分析服务 (8000/8001/8002) 已启动且会话存在")
            if 'face' not in data:
                print("   ⚠️  未获取到面部数据，继续使用其他模态生成报告")

            result = build_l1_report(session_id, logs=data)
            static_images = self._scan_static_images()

            html_content = self._build_html_report(result, {}, {}, data,
                                                   static_images,
                                                   sources=live_sources(session_id, data),
                                                   target=session_id)

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

        except Exception as e:
            print(f"❌ 错误：{e}")
            return ""

        if os.environ.get("JX_NO_BROWSER") != "1":
            try:
                webbrowser.open('file://' + os.path.realpath(report_path))
            except Exception as e:
                print(f"   ⚠️  实时报告已生成,但打开浏览器失败：{e}")
        print(f"REPORT_PATH={os.path.realpath(report_path)}")
        print(f"\n✅ 实时报告已生成：{report_path}")
        return report_path

    def _scan_static_images(self) -> List[str]:
        images = []
        if os.path.exists(self.output_dir):
            for f in os.listdir(self.output_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')) and 'trajectory' not in f and 'chart' not in f:
                    images.append(f)
        return images

    def _render_dimension_block(self, dim_key: str, dim: Dict[str, Any],
                                labels: Optional[Dict[str, str]] = None) -> str:
        """渲染单个维度的证据状态。不解读,不推断,不加形容词。

        形态 = spec §5.4 的「整节的正确形态」:值 + 有效样本量 + 置信度 + evidence_gaps。
        出分维度另带 §5.6 要求的提示句「该维度目前无独立效标,仅供行为描述」。
        **不再渲染"归一值"** —— 它是未标定标尺上的点分,与档位标签一样会被读成
        对该维度的评定,而本系统没有标定样本能给这个刻度背书。
        表格给出的"本场会话内观测区间"只由本场测量构成(均值 ± 会话内标准差)。

        每个维度自己带全自己的缺口 —— 出分维度也要列出未过门的槽。
        否则那些缺口只能靠顶层的汇总段兜底,而汇总段的所有条目都已被逐维列表
        覆盖过一遍(evidence_gaps 就是各维缺口的并集),同一批字符串会被打印两次。

        ★ A5(2026-11-21):表格新增「仪器来源」一列。`labels` 是载荷给的
        `l1.instruments.labels`(仪器键 → 中文标签);缺省时**露英文键** ——
        这一列是本报告的披露义务,宁可露英文也不能空着。
        """
        # ⚠️ 分支判据 2026-10-04 由 `dim["score"] is None` 改成 `not dim["evidence_chain"]`
        #    —— payload 里的点分已摘除(见 `research_mapper` 返回处那段注释)。
        #    两者**等价**:`score` 为 None 当且仅当该维一个槽都没过门,而那正是
        #    `evidence_chain` 为空。判据从"有没有分"改成"有没有证据",本来也更贴本仓的主张。
        if not dim["evidence_chain"]:
            gaps = "".join(f"<li>{g}</li>" for g in dim.get("evidence_gaps", []))
            never = "".join(f"<li>{g}</li>"
                            for g in dim.get("evidence_gaps_not_measured", []))
            never_block = (f"<p>另有几条是整套采集都不测的（不是这一场没测到）：</p>"
                           f"<ul>{never}</ul>" if never else "")
            return f"""
            <h3>{dim['display_name']}</h3>
            <p><strong>证据不足</strong> —— 本次未采集到足以评估该行为线索的有效样本。</p>
            <ul>{gaps}</ul>
            {never_block}
            """

        # ★ 「本场会话内观测区间」这一列:整维都取不到区间时**整列不印**。
        #
        # 实测(2026-10-26 那一场):L1 的每个指标都是**整场一个值**,载荷里
        # `observed_interval` 恒为空数组 ⟹ 这一列四行全是同一句
        # 「未采集到会话内变异信息」。读者看到的是"四个指标都没测出变异",
        # 而真相是"这一层本来就不产生区间"。**列在,但每格都是同一句废话**,
        # 比不印更糟:它占着位置,还暗示"本该有、这次缺了"。
        # ⟹ 一列全空就不印这一列,在表下说一句为什么。
        has_interval = any(e.get("observed_interval")
                           for e in dim["evidence_chain"])
        rows = "".join(
            f"<tr><td>{e['human_name']}</td>"
            f"<td class=\"num\">{_fmt_num(e['raw_value'])}{_zero_note(e)}</td>"
            + (f"<td>{_interval_text(e.get('observed_interval'))}</td>"
               if has_interval else "")
            + f"<td class=\"num\">{e.get('n_valid', '—')}</td>"
            # ★ A5:这一格是本报告的披露义务。它**永远**出现(哪怕整列都是
            #   「未提供」)—— 一列全缺就整列不印的规矩(见上,区间那一列)在这里
            #   不适用:区间缺了是"这一层不产生",而出处缺了是"还没披露清楚"。
            + f"<td class=\"prov\">{_instrument_cell(e, labels)}</td></tr>"
            for e in dim["evidence_chain"]
        )
        # 先给人话,再给表。表格是核对用的,而"这一维量到了哪几条"一句话就说完 ——
        # 读者不该为了知道"有没有测到东西"去数表格行数(使用者原话:
        # 「完全没有分析」;实测那版全篇只有三列表格)。
        names = "、".join(str(e["human_name"]) for e in dim["evidence_chain"])
        gaps = "".join(f"<li>{g}</li>" for g in dim.get("evidence_gaps", []))
        # ⚠️ 2026-10-26 换措辞:「过门指标」「无独立效标」都是行话。
        #    前者是证据门的内部叫法(读者不知道门是什么),后者是测量学词
        #    (效标 = 用来对照的外部标准)。改成读者能直接懂的话,意思不变:
        #    数够不够用、数有多可信、这一维有没有独立标准可以对照。
        gaps_block = (f"<p>这次没量到可用数的指标：</p><ul>{gaps}</ul>"
                      if gaps else "")
        # ⚠️ 2026-10-26:删槽行**不能**混进上面那个列表。
        #    实测那一版印的是「这次没量到可用数的指标：· 面部对称性：这一条
        #    当前这套采集不测」—— 小标题说"这次"(下一场可能就有),
        #    条目说"采集不测"(再采一百次也没有),同一段两句相反的话。
        #    载荷里这两类本来就分着(`evidence_gaps` / `evidence_gaps_not_measured`),
        #    是渲染层把它们拼回了一起。
        never = "".join(f"<li>{g}</li>"
                        for g in dim.get("evidence_gaps_not_measured", []))
        never_block = (f"<p>另有几条是整套采集都不测的（不是这一场没测到）：</p>"
                       f"<ul>{never}</ul>" if never else "")
        interval_th = "<th>本场会话内观测区间</th>" if has_interval else ""
        return f"""
        <h3>{dim['display_name']}</h3>
        <p>量到 {dim['matched_indicators']} 项 · 置信度「<strong>{dim['confidence']}</strong>」
        · 这一维没有独立的对照标准，数只能描述行为，不能当评价。</p>
        <p class="note">量到了这几条：{names}。</p>
        <table><thead><tr><th>指标</th><th>原始值</th>{interval_th}
        <th>有效样本量</th><th>仪器来源</th></tr></thead>
        <tbody>{rows}</tbody></table>
        {gaps_block}
        {never_block}
        """

    def _generate_deep_text_analysis(self, features: Dict[str, Any], result: Dict[str, Any]) -> str:
        """只陈述本次实际测到了什么,不做任何解读、推断与形容词修饰。

        每个维度按 spec §5.4 的「整节的正确形态」渲染,出分维度带 §5.6 的"无独立效标"
        提示。本函数**不再输出聚合摘要**:它曾印「综合科研潜力评分为 X 分,评级为 Y」,
        后改印「综合行为观测摘要:{档位}(置信度上限:…)」—— 两者都是把未标定标尺上的
        点分与五档评语当对人的评定(实测报告头渲染出「0.0 / 综合行为观测评分 / 待提升」)。
        聚合呈现移入报告头部的覆盖卡,只给「区间 + 置信度 + 依据」(spec §5.6)。

        参数 features 保留是为了调用方签名稳定;本函数不再从原始特征里另取默认值。

        ★ A5:仪器标签从载荷读(`l1.instruments.labels`)。旧引擎的载荷没有这一块 ⟹
        传 `None`,那一列就露英文键(见 `_instrument_cell`)—— 不编一份标签出来。
        """
        labels = ((result.get("l1") or {}).get("instruments") or {}).get("labels") or None
        parts = []
        for dim_key, dim in result["dimensions"].items():
            parts.append(self._render_dimension_block(dim_key, dim, labels))
        # 原本此处另有一段顶层「证据缺口」汇总,现已删除 —— 它的每一条都来自
        # result["evidence_gaps"](各维缺口的并集),而逐维块已经把各自的缺口列全,
        # 于是同一批字符串会在报告里出现两遍。缺口现在只有逐维这一处来源(带维度归属)。
        return "".join(parts)

    def _render_vanished_dimensions(self, result: Dict[str, Any]) -> str:
        """整维消失的那一维要**说出来** —— 前端说了,交付物也得说。

        L1 的构念对账判定 `行为表现活跃度` 的三个槽没有同构念的 L1 指标 ⟹
        它整个不进 `dimensions`,而进 `vanished_dimensions`。前端 `ReportViewer`
        为此专门加了一块卡片,注释原话是「不渲染的话报告会**悄悄少一块**,
        而少的那块恰好是『测不到的东西』—— 那是最该说出来的部分」。

        但本文件原先只遍历 `result["dimensions"]` ⟹ **HTML 报告里那一维凭空消失**。
        同一场会话:界面上说"这一维整维未出",下载的 HTML 说"只有四个维度"。

        ⚠️ 措辞必须是「本层不测」而不是「本场没测到」—— 后者让人等下一场,
        前者才是实话。逐槽理由走 `l1_mapper.drop_reason`(人话那层),
        槽名走 `slot_label`,内部键不出现在交付物里。
        钉子:`tests/test_html_report_shows_vanished_dimensions.py`。
        """
        vanished = result.get("vanished_dimensions") or []
        if not vanished:
            return ""
        items = "".join(
            f"<li><strong>{v['display_name']}</strong>"
            f"(要量 {v['n_slots']} 项,当前这套采集一项都不测)"
            f"<ul>" + "".join(
                f"<li>{l1_mapper.slot_label(d['slot'])}:这一条当前这套采集不测 —— "
                f"{l1_mapper.drop_reason(d['slot'])}</li>"
                for d in v.get("dropped", [])) + "</ul></li>"
            for v in vanished)
        return f"""
        <div id="vanished-dimensions">
            <h3>这一维整套采集都不测(不是这一场没测到)</h3>
            <p>下面这一维<strong>不是本场没测到</strong>,是它要量的东西当前这套采集
            根本没有 —— 再采一百次也不会有,别等它。</p>
            <ul>{items}</ul>
        </div>
        """

    def _coverage_heading(self, result: Dict[str, Any]) -> str:
        """「本次观测覆盖」那张图的标题 —— **维度数从载荷算,不是字面量**。

        这里原先写死 `📊 五维证据覆盖`。旧引擎给满五维,所以那句一直对;
        L1 引擎下 `行为表现活跃度` 整维消失 ⟹ 正文里只有**四个**维度,
        而标题还写着「五维」—— 同一份报告自己跟自己矛盾,读者会以为少了一维。

        ⚠️ 数的是 `dimensions`,**不含** `vanished_dimensions`:消失的那一维
        在正文里另有一块说明(「本层不测」),把它算进"本次观测"会让读者
        以为它出了值。钉子:`tests/test_the_report_stops_saying_wrong_things.py`。

        ⚠️ 2026-10-26 摘掉了开头的 📊:标题里塞 emoji 是模板腔,而且这一节
        现在是一张真图(内联 SVG),不需要用符号冒充图标。
        ⚠️ 同日摘掉括注「(4 个维度)」:标题已经说了「四维」,括注是把同一个数
        再说一遍 —— 实测渲染出来读作「四维证据覆盖（4 个维度）」。
        ⚠️ 同日把「证据覆盖」换成「各量到了多少」:「证据覆盖」是报告层的术语
        (哪个指标过了证据门),而这张图回答的问题很直白 —— 每一维量到了几条。
        """
        n = len(result.get("dimensions") or {})
        return f"{_CN_NUM.get(n, n)}维各量到了多少"

    def _lead_sentence(self, result: Dict[str, Any]) -> str:
        """报告开头那句人话结论 —— **每一个数都从载荷算**,不写固定文案。

        ## 为什么要它(2026-10-26)

        使用者原话:「完全没有分析」。实测那版报告全篇只有三列表格
        (原始值 / 未采集到会话内变异信息 / 有效样本量),读者读完不知道
        "这一场到底量到了什么"。

        结论句要回答的正是那件事,而且必须**跟着载荷变**:
        换一份覆盖数不同的载荷,这句话就得跟着换。判据是两条载荷渲染出的
        正文**逐字不同**(`tests/test_report_says_something_to_a_person.py`)——
        渲染层自己编一句固定文案,载荷一变它就是假话(本仓记过多次的形态)。

        ⚠️ 不评级、不给分:这里说的是"量到了几条",不是"这个人怎么样"。

        ## ⚠️ 2026-11-21 分母换掉了:`n_slots`(14)→ `n_measurable`

        原来这句写「量到了 8 / <strong>14</strong> 项指标」。14 里有 6 个槽是
        **永远填不上的**(3 个「这套采集不测」+ 3 个「整维没有对应指标」),把 14 当分母
        会把"这套采集缺的东西"读成"这一场没做到"。现在分母是可测量的槽数,
        另外两段单独说一句 —— 三段相加仍是 14,那个数没有消失。

        ⚠️ 旧载荷(没有 `n_measurable`,例如前端 store 里缓存的 `l1-1`)退回
        `n_slots` 做分母:那不是"另算一遍",而是**它只有这一个数可用**。
        """
        cov = result.get("coverage") or {}
        dims = result.get("dimensions") or {}
        n_passed = cov.get("n_passed", 0)
        n_slots = cov.get("n_slots", 0)
        n_dropped = cov.get("n_dropped", 0)
        # 分段数由载荷给。取不到时是 `None` —— 与"给了 0"必须分开:
        # 0 是真值(这一段真的一个槽都没有),`None` 是"这份载荷没带分段"。
        n_measurable = cov.get("n_measurable")
        n_not_measured = cov.get("n_not_measured")
        n_no_indicator = cov.get("n_no_indicator")
        denom = n_measurable if isinstance(n_measurable, int) else n_slots

        def _pair(dim: Dict[str, Any]) -> tuple[int, int]:
            got, _, total = str(dim.get("matched_indicators", "0/0")).partition("/")
            try:
                return int(got), int(total)
            except (TypeError, ValueError):
                return 0, 0

        # 最完整 / 最薄的两维 —— 用 `matched_indicators` 的分子比分母,
        # 不用绝对条数(维度声明的槽数不同,比绝对条数会把"槽多"读成"量得多")。
        ranked = sorted(dims.values(), key=lambda d: (_pair(d)[0] / (_pair(d)[1] or 1)),
                        reverse=True)
        sentences: list[str] = []

        if not dims:
            sentences.append("这一场没有任何维度产出可核对的行为量。")
        elif not n_passed:
            # 0 也要把「0 / N」印出来 —— 「一个都没量到」与「量到 0 个」是同一件事,
            # 但读者要能一眼对上下面的条形图(那里每条都是 0 / k)。
            # 分母是 0(这一场一个可测量槽都没有)时印 `0`,不印「0 / 0」。
            pair = f"0 / {denom}" if denom else "0"
            sentences.append(
                f"这一场在 {len(dims)} 个维度上量到了 <strong>{pair}</strong> 项可测量指标"
                f"—— 不是「没测」,是测到的还不足以支撑读数,逐条原因见下。")
        else:
            all_full = all(_pair(d)[1] > 0 and _pair(d)[0] >= _pair(d)[1] for d in ranked)
            if all_full:
                sentences.append(
                    f"这一场在 {len(dims)} 个维度上量到了 <strong>{n_passed} / {denom}</strong> "
                    f"项可测量指标,全部 {len(dims)} 个维度的可测量指标均已完整覆盖。")
            else:
                best = ranked[0]
                b_got, b_tot = _pair(best)
                b_ratio = b_got / (b_tot or 1)
                sentences.append(
                    f"这一场在 {len(dims)} 个维度上量到了 <strong>{n_passed} / {denom}</strong> "
                    f"项可测量指标,其中最完整的是「{best.get('display_name', '')}」"
                    f"({b_got} / {b_tot})。")
                if len(ranked) > 1:
                    worst = ranked[-1]
                    w_got, w_tot = _pair(worst)
                    w_ratio = w_got / (w_tot or 1)
                    if w_ratio < b_ratio - 1e-9:
                        sentences.append(
                            f"最薄的是「{worst.get('display_name', '')}」({w_got} / {w_tot})。")

        if isinstance(n_not_measured, int) and isinstance(n_no_indicator, int):
            # 新载荷:把剩下两段**分开点名**,并说明它们与这一场的表现无关。
            # 措辞与 `l1_report.COVERAGE_EXPLANATION` 同一套词 —— 两个出口不许各说各的。
            if n_not_measured or n_no_indicator:
                sentences.append(
                    f"分母之外还有 <strong>{n_not_measured}</strong> 项「这套采集不测」"
                    f"、<strong>{n_no_indicator}</strong> 项「整维没有对应指标」—— "
                    f"这 {n_not_measured + n_no_indicator} 项不是这一场量得好不好的问题,"
                    f"合起来才是 {n_slots} 个槽,逐条理由见下方「整套采集都不测」与各维缺口。")
        elif n_dropped:
            # 旧载荷(没有分段字段):只能照它自己带的 `n_dropped` 说一句。
            sentences.append(
                f"另有 {n_dropped} 项在当前这套采集里没有对应的测法,已整条剔除 —— "
                f"它们不是「这一场没测到」,见下方「整套采集都不测」。")
        cap = cov.get("confidence_cap")
        if cap and cap != "无":
            sentences.append(f"所有读数的置信度上限是「{cap}」。")
        return "".join(sentences)

    # ------------------------------------------------------------------
    # 覆盖数的三段式(2026-11-21)
    # ------------------------------------------------------------------
    #
    # ⚠️ 这三个方法**不计算任何数**:三段计数与那句话都由 `l1_report` 装进载荷
    #    (`coverage.summary_line` / `coverage.explanation` / `coverage.segments`)。
    #    渲染层再数一遍 = 第二份口径,而本仓为"两处各算一遍"栽过
    #    (`n_passed` 曾虚报成 7)。这里只做排版与"旧载荷没带"的兜底。

    def _coverage_line(self, cov: Dict[str, Any]) -> str:
        """三段式那一行 —— 文字与数字都照载荷印。"""
        line = cov.get("summary_line")
        if line:
            return f"<p class=\"coverage-line\">{line}</p>"
        # 旧载荷(`l1-1`,或前端 store 里的缓存结果)没有这一句。**不补算**:
        # 补算就是第二份口径。只说这一份没带,并把它自己带的两个数照印。
        return ("<p class=\"coverage-line\">这份载荷没带分段覆盖数"
                f"(旧版载荷),只有通过槽数 {cov.get('n_passed', '—')} / "
                f"{cov.get('n_slots', '—')}。</p>")

    def _coverage_explanation(self, cov: Dict[str, Any]) -> str:
        """那句"为什么不测"的解释 —— 同样照载荷印,不在渲染层另写一份。"""
        why = cov.get("explanation")
        if not why:
            return ""
        return f"<p class=\"coverage-why\">{why}</p>"

    def _coverage_hero_number(self, cov: Dict[str, Any]) -> str:
        """报告头那个大数 —— 分子 / **可测量**槽数。

        分母从 `n_slots`(14)换成 `n_measurable`(8):14 里有 6 个槽永远填不上,
        拿它当分母会把"这套采集缺的东西"读成"这一场没做到"。
        可测量槽数为 0 时印 `—`,不印「0 / 0」(那不是覆盖率,是个假分数)。
        """
        n_measurable = cov.get("n_measurable")
        if isinstance(n_measurable, int):
            return f"{cov.get('n_passed', '—')} / {n_measurable}" if n_measurable else "—"
        return f"{cov.get('n_passed', '—')} / {cov.get('n_slots', '—')}"

    def _figure_coverage(self, result: Dict[str, Any]) -> str:
        """图一:**每个维度量到了几条** —— 横向条形,一条一维。

        ## 为什么这张图该存在

        原先那张雷达图的半径是"每个维度过了几个槽",即同一个数换了张皮。
        雷达图只在"形状"本身有意义时才成立;这里的维度之间没有可比的量纲
        (维度声明的槽数不同),所以正确的形态是**条形** —— 它回答"哪一维量得全、
        哪一维薄",一眼看完。

        单系列 ⟹ **不要图例**(标题已经说了这是什么);数值直接标在条尾
        (dataviz:选择性直接标注,不给每个点都写数)。

        返回内联 SVG 字符串;**不落盘、不引外部文件** —— 报告拷走图还在。
        """
        dims = result.get("dimensions") or {}
        if not dims:
            return ""
        rows: list[tuple[str, int, int]] = []
        for dim in dims.values():
            got, _, total = str(dim.get("matched_indicators", "0/0")).partition("/")
            try:
                rows.append((str(dim.get("display_name", "")), int(got), int(total)))
            except (TypeError, ValueError):
                rows.append((str(dim.get("display_name", "")), 0, 0))

        label_w, bar_w, row_h, pad_top = 132, 380, 34, 10
        width = label_w + bar_w + 96
        height = pad_top * 2 + row_h * len(rows)
        parts: list[str] = [
            f'<svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="每个维度量到几条、一共要量几条">']
        for i, (name, got, total) in enumerate(rows):
            y = pad_top + i * row_h
            cy = y + row_h / 2
            ratio = min(1.0, max(0.0, (got / total) if total else 0.0))
            w = max(2.0, bar_w * ratio) if got else 0.0
            parts.append(f'<text class="label" x="0" y="{cy + 4:.1f}">{name}</text>')
            # 轨道画满整条 —— 读者要能看出"分母有多长",否则 3/3 与 3/14 长得一样。
            parts.append(f'<rect class="track" x="{label_w}" y="{cy - 5:.1f}" '
                         f'width="{bar_w}" height="10" rx="4"/>')
            if w:
                parts.append(f'<rect class="bar" x="{label_w}" y="{cy - 5:.1f}" '
                             f'width="{w:.1f}" height="10" rx="4"/>')
            parts.append(f'<text class="value" x="{label_w + bar_w + 10}" '
                         f'y="{cy + 4:.1f}">{got} / {total}</text>')
        parts.append("</svg>")
        return "".join(parts)

    @staticmethod
    def _prepare_dispersion_rows(result: Dict[str, Any]) -> list[dict]:
        """统一清洗并筛选逐题离散度行，供 SVG 图与平直行计数共享（防 NaN/inf/倒置/极小除零）。"""
        disp = ((result.get("l1") or {}).get("dispersion") or {})
        if not isinstance(disp, dict):
            return []
        rows: list[dict] = []
        for name, d in disp.items():
            if not isinstance(d, dict):
                continue
            try:
                lo, mid, hi = float(d["min"]), float(d["median"]), float(d["max"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(lo) and math.isfinite(mid) and math.isfinite(hi)):
                continue
            if hi < lo:
                continue
            mid = min(hi, max(lo, mid))
            eps = max(1e-12, 1e-9 * max(abs(lo), abs(hi)))
            flat = (hi - lo) <= eps
            rows.append({
                "name": name,
                "label": l1_quality.indicator_label(name),
                "lo": lo,
                "mid": mid,
                "hi": hi,
                "flat": flat,
                "n_questions": d.get("n_questions"),
            })
        return rows

    def _figure_dispersion(self, result: Dict[str, Any]) -> str:
        """图二:**逐题离散度** —— 每条指标在各题之间差多少（独立刻度 + 边界防御）。"""
        rows = self._prepare_dispersion_rows(result)
        if not rows:
            return ""

        n_win = ((result.get("l1") or {}).get("windows") or {}).get("n_questions")
        label_w, plot_w, row_h, pad_top = 136, 352, 34, 10
        width = label_w + plot_w + 96
        height = pad_top * 2 + row_h * len(rows)
        parts: list[str] = [
            f'<svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="每条指标在题与题之间的分布">']
        for i, row in enumerate(rows):
            lo, mid, hi, flat = row["lo"], row["mid"], row["hi"], row["flat"]
            y = pad_top + i * row_h
            cy = y + row_h / 2
            q_cnt = row.get("n_questions")
            suffix = (
                f" ({q_cnt}/{n_win}题)"
                if isinstance(q_cnt, int) and isinstance(n_win, int) and 0 < q_cnt < n_win
                else ""
            )
            parts.append(f'<text class="label" x="0" y="{cy + 4:.1f}">{escape(row["label"] + suffix)}</text>')
            parts.append(f'<rect class="track" x="{label_w}" y="{cy - 1:.1f}" '
                         f'width="{plot_w}" height="2" rx="1"/>')

            if flat:
                cx = label_w + plot_w / 2
                parts.append(f'<circle class="median" cx="{cx:.1f}" cy="{cy:.1f}" '
                             f'r="4.5"/>')
                parts.append(f'<text class="tick" x="{cx:.1f}" y="{cy - 8:.1f}" '
                             f'text-anchor="middle">{_fmt_compact(lo)}</text>')
                parts.append(f'<text class="flat-note" x="{width - 4}" '
                             f'y="{cy + 4:.1f}" text-anchor="end">各题相同</text>')
                continue

            span = hi - lo
            x0, x1 = lo - span * 0.06, hi + span * 0.06
            if x1 <= x0:
                cx = label_w + plot_w / 2
                parts.append(f'<circle class="median" cx="{cx:.1f}" cy="{cy:.1f}" r="4.5"/>')
                parts.append(f'<text class="flat-note" x="{width - 4}" y="{cy + 4:.1f}" text-anchor="end">各题相同</text>')
                continue
            scale = plot_w / (x1 - x0)

            def _x(v: float) -> float:
                return label_w + (v - x0) * scale

            parts.append(f'<line class="range" x1="{_x(lo):.1f}" y1="{cy:.1f}" '
                         f'x2="{_x(hi):.1f}" y2="{cy:.1f}"/>')
            parts.append(f'<circle class="median" cx="{_x(mid):.1f}" cy="{cy:.1f}" '
                         f'r="4.5"/>')
            parts.append(f'<text class="tick" x="{_x(lo):.1f}" y="{cy - 8:.1f}" '
                         f'text-anchor="middle">{_fmt_compact(lo)}</text>')
            parts.append(f'<text class="tick" x="{_x(hi):.1f}" y="{cy - 8:.1f}" '
                         f'text-anchor="middle">{_fmt_compact(hi)}</text>')
            parts.append(f'<text class="value" x="{width - 4}" y="{cy + 4:.1f}" '
                         f'text-anchor="end">{_fmt_compact(mid)}</text>')
        parts.append("</svg>")
        return "".join(parts)

    @classmethod
    def _n_flat_rows(cls, result: Dict[str, Any]) -> int:
        """离散度里有几行「各题取到同一个值」—— 与图共享同一份清洗结果。"""
        return sum(1 for r in cls._prepare_dispersion_rows(result) if r["flat"])

    def _render_l1_audit_text(self, result: Dict[str, Any],
                              only: Optional[tuple] = None) -> str:
        """把载荷里那四组「这份报告能说什么」的前提印出来。

        前端 `L1IndicatorTable` 全都渲染了(它的注释原话:放在表格上方而不是折叠里,
        因为这些是"这份报告能说什么"的前提),而 HTML 报告**一个都没印**:

        | 载荷字段 | 它回答的问题 | 不印的后果 |
        |---|---|---|
        | `l1.gaps` | 本场素材层面的限制(拼接、没有提问台账、没有 ASR) | 读者把只覆盖最后 7 分钟的数当成整场 |
        | `l1.comparability_notes` | 哪些槽名与旧报告相同但统计量换了 | 读者把两份报告的数相减 |
        | `l1.low_confidence_notes` | 置信度为什么只有"低" | 读者无从判断这个数能不能用 |
        | `l1.windows` | 整场一个窗口,还是逐回答一个窗口 | 两个口径的 F/G 值被当成同一件事 |

        ⚠️ 全部来自载荷,**渲染层不另写一份**(否则迟早分叉)。
        四组都空时不印任何标题 —— 空标题会被读成"这里该有内容但丢了"。
        钉子:`tests/test_html_report_carries_the_audit_text.py`。

        ## `only`(2026-10-26):四组不都得印在正文

        使用者原话:「科研报告上面全是更上一次比怎么这么样」。那指的是
        `comparability_notes` —— 它是**引擎迁移说明**,每份报告都有、与这一场无关。
        内容不删(复核要查),但必须收进折叠附录 ⟹ 正文只印前三维,
        附录单独印它。`only=None`(默认)仍是四组全印,直接调用的测试照旧。
        """
        l1 = result.get("l1") or {}

        def _want(key: str) -> bool:
            return only is None or key in only

        def _ul(items) -> str:
            return "<ul>" + "".join(f"<li>{_sanitize_plain(i)}</li>" for i in items) + "</ul>"

        blocks: list[str] = []

        gaps = l1.get("gaps") or []
        if gaps and _want("gaps"):
            blocks.append(
                "<div class='l1-audit'><h3>本场素材的限制（是素材的问题，不是算错）</h3>"
                + _ul(gaps) + "</div>")

        # 印**人话版**:审计版(`comparability_notes`)里是「旧槽键 → L1指标名：…」,
        # 那 6 个内部指标名不该出现在交付物正文里(前端同理由 `plain()` 之外还要换表)。
        # 载荷没发人话版时退回审计版 —— 宁可露出英文,也不许把这一条吞掉。
        notes = (l1.get("comparability_notes_plain")
                 or l1.get("comparability_notes") or [])
        if notes and _want("comparability"):
            blocks.append(
                "<div class='l1-audit'><h3>与旧引擎报告对比时注意</h3>"
                "<p>下列槽名相近，但统计量已更换，两边的数<strong>不可相减</strong>。</p>"
                + _ul(notes) + "</div>")

        # ⚠️ 印**人话版**:审计版是 `{指标名}:{理由}`,而指标名是内部键 ——
        #    实测交付物正文里印出过 `speech_rate:` / `pause_rate_per_min:` 等 5 个键,
        #    而维度表里同一个指标印的是中文槽名 ⟹ 读者会以为是两个东西。
        #    载荷没发人话版时退回审计版(宁可露英文,不许把这一段吞掉)。
        low = (l1.get("low_confidence_notes_plain")
               or l1.get("low_confidence_notes") or [])
        if low and _want("low_confidence"):
            blocks.append(
                "<div class='l1-audit'><h3>置信度为「低」的原因（素材限制，不是算错）</h3>"
                + _ul(low) + "</div>")

        w = l1.get("windows") or {}
        if w.get("granularity") and _want("windows"):
            is_answer = w["granularity"] == "answer"
            text = (f"逐回答窗口（{w.get('n_windows', '?')} 个）" if is_answer
                    else "整场一个窗口")
            outside = {k: n for k, n in (w.get("frames_outside") or {}).items()
                       if isinstance(n, (int, float)) and n > 0}
            extra = ""
            if outside:
                # ⚠️ 2026-10-26:原来写「不计入 F/G 的分母」—— F/G 是内部族名
                #    (F = 面部、G = 手势),读者不知道那是什么。说模态名即可。
                extra = ("；不在任何回答窗口内的帧：" +
                         " · ".join(f"{_MODALITY_LABELS.get(k, k)} {n} 帧" for k, n in outside.items()) +
                         "（这些帧不算进「平均到每秒」那类数的分母）")
            blocks.append(
                f"<div class='l1-audit'><h3>这一场是怎么切片的</h3><p>{text}{extra}</p></div>")

        return "".join(blocks)

    def _render_instrument_disclosure(self, result: Dict[str, Any]) -> str:
        """**仪器来源披露**那一节(A5,2026-11-21)—— 逐条标在表里,汇总放在页脚之前。

        ## 为什么必须有这一节

        "本系统同时存在两套面部仪器"这句话此前**只出现在文案里**:38 条 L1 指标
        没有一条标着出处,报告也没说 `au12_smile` 与 `bs_mouthSmile` 是两套东西。
        使用者拿报告去讲"双仪器",而报告本身一个字都没提 —— 那就是一句无法核对的声明。

        ## 三条呈现规矩

        1. **数字全部来自载荷**(`l1.instruments`,由 `l1_report._instrument_block` 装好)。
           渲染层自己数一遍 ⟹ 页脚与汇总区两个数迟早不同。
        2. **页脚那句话不在本文件拼** —— 它由 `instrument_provenance.format_footer()`
           出(`footer` 键),两处措辞才不会分叉。
        3. **「未登记」要显眼,并且说清它是缺口不是仪器**:它单独一个样式、单独一段,
           还把**缺哪几列**点出来(`unmapped_columns`)—— 只说"8 条未登记"没人能动手。
        """
        block = ((result.get("l1") or {}).get("instruments") or {})
        if not block:
            # 旧引擎(`research_mapper`)的载荷没有这一块。**不编**:说清这一份没带,
            # 而不是印一个「0 条」—— 后者会被读成"披露了,结果是零"。
            return """
            <section id="instrument-disclosure">
                <h2>仪器来源</h2>
                <p class="note">这一份报告用的载荷没有带仪器来源字段（旧引擎产物），
                所以上面每张表的「仪器来源」列都写「未提供」—— 那是"没披露"，
                不是"没有仪器"。当前默认引擎（L1）的报告一律带这一节。</p>
            </section>
            """

        labels = block.get("labels") or {}
        counts = block.get("counts") or {}
        order = block.get("order") or list(counts)
        chips = "".join(
            f'<span class="prov-chip'
            f'{" prov-chip-unknown" if k == INSTRUMENT_UNKNOWN else ""}">'
            f'{labels.get(k, k)} {counts.get(k, 0)}</span>'
            for k in order)

        n_unknown = counts.get(INSTRUMENT_UNKNOWN, 0)
        unknown_html = ""
        if n_unknown:
            cols = "、".join(f"<code>{escape(str(c))}</code>"
                             for c in (block.get("unmapped_columns") or []))
            where = f"（缺的是这几列：{cols}）" if cols else ""
            unknown_html = (
                f'<p class="prov-warn"><strong>{n_unknown} 条指标标着「未登记」</strong>。'
                f'{block.get("unknown_note", "")}{where}</p>')

        multi_html = (f'<p class="note">{block.get("multi_note", "")}</p>'
                      if block.get("n_multi") else "")

        blendshape_zero = ""
        if counts.get("blendshape", 0) == 0:
            blendshape_zero = (
                '<p class="note">注意这份报告里<strong>没有一条</strong>面部指标来自模型回归系数'
                '（52 维面部模型系数全部仅作采集记录、未参与主表任何一条行为指标）——'
                '这句话本身也是要披露的内容。</p>')

        non_inter = (block.get("non_interchangeability_disclosure") or {}).get(
            "conclusion_plain",
            "离线实测（32 场 / 389 条）表明，关键点几何代理与 52 维模型回归系数在 12 个构念中"
            "仅有 1 个达到秩一致性 0.7 以上，因此两套测量结果不可混用，本报告对每条指标明确标明出处。",
        )

        return f"""
        <section id="instrument-disclosure">
            <h2>这些数是谁量的（仪器来源）</h2>
            <p class="note">本系统的面部测量同时存在两套仪器：一套是手写的关键点几何代理（geometric，
            例如「眉内距 ÷ 眼距」这类比率），另一套是面部网格模型回归出的 52 维模型系数（blendshape）。
            {escape(non_inter)}</p>
            <div class="prov-chips">{chips}</div>
            <p class="prov-footer">{block.get("footer", "")}</p>
            {multi_html}
            {blendshape_zero}
            {unknown_html}
        </section>
        """

    def _render_evidence_chain_notices(self, result: Dict[str, Any]) -> str:
        """在正文中简洁披露跨模态时间覆盖、转写置信度与底层封停依赖提示。"""
        l1 = result.get("l1") or {}
        items: list[str] = []
        cmc = l1.get("cross_modal_coverage") or {}
        if cmc.get("summary_plain"):
            items.append(f"<li><strong>跨模态时间对齐：</strong>{escape(str(cmc['summary_plain']))}</li>")
        asr_disc = l1.get("asr_confidence_disclosure") or {}
        if asr_disc.get("note_plain"):
            items.append(f"<li><strong>语音转写置信度：</strong>{escape(str(asr_disc['note_plain']))}</li>")
        for w in (l1.get("blocked_dependency_warnings") or []):
            if w.get("note_plain"):
                items.append(f"<li><strong>2D 代理提示：</strong>{escape(str(w['note_plain']))}</li>")
        if not items:
            return ""
        return (
            "<div class='l1-audit'><h3>测量边界与跨模态对齐核查</h3>"
            f"<ul>{''.join(items)}</ul></div>"
        )

    def _render_extra_appendices(self, result: Dict[str, Any]) -> str:
        """折叠附录：参数来源三级登记（T1/T2/T3）、主动拒绝与显式封停台账、第二仪器对照。"""
        l1 = result.get("l1") or {}
        blocks: list[str] = []

        pp = l1.get("parameter_provenance") or {}
        reg = pp.get("registry") or []
        if reg:
            rows = "".join(
                f"<tr><td>{escape(str(p.get('label', p.get('param_id', ''))))}</td>"
                f"<td class=\"num\">{escape(str(p.get('value', '—')))} {escape(str(p.get('unit', '')))}</td>"
                f"<td>{escape(str(p.get('sub_tier', p.get('tier', ''))))}"
                f"{'（未标定）' if p.get('provisional') else ''}</td>"
                f"<td>{escape(str(p.get('provenance', '')))}</td></tr>"
                for p in reg
            )
            blocks.append(f"""
            <details>
                <summary>附录：参数来源三级登记表（T1 物理定义 / T2 文献与标定 / T3 经验临时）</summary>
                <p class="note">{escape(str(pp.get("summary_plain", "")))}</p>
                <table><thead><tr><th>参数</th><th>取值</th><th>来源分级</th><th>依据说明</th></tr></thead>
                <tbody>{rows}</tbody></table>
            </details>
            """)

        gl = l1.get("governance_ledger") or {}
        l0_b = gl.get("l0_blocked") or []
        ret = gl.get("retired_redundant") or []
        if l0_b or ret:
            rows_b = "".join(
                f"<tr><td>{escape(str(x.get('label', x.get('id', ''))))}</td>"
                f"<td>底层封停</td>"
                f"<td>{escape(str(x.get('reason_plain', '')))}</td>"
                f"<td>{escape(str(x.get('unblock_human', '')))}</td></tr>"
                for x in l0_b
            )
            rows_r = "".join(
                f"<tr><td>{escape(str(x.get('label', x.get('id', ''))))}</td>"
                f"<td>主动拒绝 / 退役</td>"
                f"<td>{escape(str(x.get('reason_plain', '')))}</td>"
                f"<td>{escape(str(x.get('unblock_human', '')))}</td></tr>"
                for x in ret
            )
            blocks.append(f"""
            <details>
                <summary>附录：本系统主动拒绝与显式封停的指标台账（含解封条件）</summary>
                <p class="note">{escape(str(gl.get("summary_plain", "")))}</p>
                <table><thead><tr><th>对象</th><th>治理状态</th><th>拒绝/封停理由</th><th>解封条件</th></tr></thead>
                <tbody>{rows_b}{rows_r}</tbody></table>
            </details>
            """)

        sec = l1.get("second_instrument_appendix") or {}
        if sec.get("available") and sec.get("rows"):
            srows = "".join(
                f"<tr><td>{escape(_clean_prose(r.get('label') or r.get('name', '')))}</td>"
                f"<td class=\"num\">{_fmt_num(r.get('value'))}</td>"
                f"<td class=\"num\">{escape(str(r.get('n_valid', '—')))}</td></tr>"
                for r in sec["rows"]
            )
            blocks.append(f"""
            <details>
                <summary>附录：第二仪器（52 维面部模型系数）独立对照参考</summary>
                <p class="note">{escape(_clean_prose(sec.get("disclaimer", "")))}</p>
                <table><thead><tr><th>对照项</th><th>读数</th><th>有效帧数</th></tr></thead>
                <tbody>{srows}</tbody></table>
            </details>
            """)

        dual_block = self._render_dual_calibration_appendix(l1)
        if dual_block:
            blocks.append(dual_block)

        fv2_block = self._render_feature_layer_v2_appendix(l1)
        if fv2_block:
            blocks.append(fv2_block)

        return "".join(blocks)

    def _render_stage_timeline_svg(self, seg_data: Dict[str, Any]) -> str:
        """渲染宏观行为阶段时间轴与转折点内联 SVG 及图例。"""
        raw_segs = seg_data.get("macro_stages") or seg_data.get("segments") or []
        raw_cps = seg_data.get("change_points") or seg_data.get("changepoints") or []
        if not isinstance(raw_segs, list) or not raw_segs:
            return ""

        segs: list[dict[str, Any]] = []
        max_end = 0.0
        for item in raw_segs:
            if not isinstance(item, dict):
                continue
            s0 = _finite_float(item.get("start_sec"))
            s1 = _finite_float(item.get("end_sec"))
            if s0 is None or s1 is None or s1 < s0:
                continue
            max_end = max(max_end, s1)
            raw_key = str(
                item.get("color_key")
                or item.get("stage")
                or item.get("macro_stage")
                or item.get("state")
                or ""
            )
            ckey = _MACRO_COLOR_CLASS.get(raw_key, "low")
            lbl = _clean_prose(item.get("label") or _MACRO_DEFAULT_LABELS.get(ckey, "其他阶段"))
            ann = _clean_prose(item.get("annotation") or "")
            ratio = _finite_float(item.get("ratio"))
            segs.append({
                "start": s0, "end": s1, "ckey": ckey,
                "label": lbl, "ratio": ratio, "annotation": ann,
            })
        if not segs:
            return ""

        total_raw = seg_data.get("total_duration_sec")
        if total_raw is None:
            total_raw = seg_data.get("total_sec")
        span_sec = max(_finite_float(total_raw) or 0.0, max_end, 0.001)

        pad_x, bar_w = 14.0, 692.0
        rects: list[str] = []
        legend_map: dict[tuple[str, str], float] = {}
        annotated_note = ""
        for seg in segs:
            s0 = max(0.0, min(span_sec, seg["start"]))
            s1 = max(s0, min(span_sec, seg["end"]))
            x_raw = pad_x + (s0 / span_sec) * bar_w
            w_raw = ((s1 - s0) / span_sec) * bar_w
            x = min(x_raw, pad_x + bar_w - 2.0)
            w = max(2.0, min(w_raw, pad_x + bar_w - x))
            rects.append(
                f'<rect class="stage-seg-{seg["ckey"]}" x="{x:.1f}" y="22" '
                f'width="{w:.1f}" height="24">'
                f'<title>{escape(seg["label"])} {s0:.1f}s - {s1:.1f}s</title></rect>'
            )
            ratio = seg["ratio"] if seg["ratio"] is not None else ((s1 - s0) / span_sec)
            lkey = (seg["ckey"], seg["label"])
            legend_map[lkey] = legend_map.get(lkey, 0.0) + max(0.0, ratio)
            if not annotated_note and seg["annotation"]:
                annotated_note = seg["annotation"]

        cps: list[dict[str, Any]] = []
        cp_marks: list[str] = []
        for cp in (raw_cps if isinstance(raw_cps, list) else []):
            if isinstance(cp, (int, float)) and not isinstance(cp, bool):
                t_cp = _finite_float(cp)
                cue = "节律转折点"
            elif isinstance(cp, dict):
                t_cp = _finite_float(cp.get("time_sec"))
                cue = _clean_prose(cp.get("cue_summary") or cp.get("label") or "节律转折点")
            else:
                continue
            if t_cp is None:
                continue
            t_clamped = max(0.0, min(span_sec, t_cp))
            cx = max(pad_x + 4.0, min(pad_x + bar_w - 4.0, pad_x + (t_clamped / span_sec) * bar_w))
            cps.append({"time_sec": t_clamped, "cue": cue})
            cp_marks.append(
                f'<g><title>{t_clamped:.1f}s 转折：{escape(cue)}</title>'
                f'<line class="cp-halo" x1="{cx:.1f}" y1="20" x2="{cx:.1f}" y2="48"/>'
                f'<line class="cp-line" x1="{cx:.1f}" y1="18" x2="{cx:.1f}" y2="50"/>'
                f'<polygon class="cp-diamond" points="{cx:.1f},7 {cx + 5:.1f},12 '
                f'{cx:.1f},17 {cx - 5:.1f},12"/></g>'
            )

        fracs = (0.0, 0.25, 0.5, 0.75, 1.0) if span_sec >= 8.0 else (0.0, 0.5, 1.0)
        ticks: list[str] = []
        for frac in fracs:
            val_s = frac * span_sec
            if val_s <= 0.0:
                t_txt = "0s"
            elif span_sec < 10.0:
                t_txt = f"{val_s:.1f}s".replace(".0s", "s")
            else:
                t_txt = f"{int(round(val_s))}s"
            tx = pad_x + frac * bar_w
            anchor = "start" if frac == 0.0 else ("end" if frac == 1.0 else "middle")
            ticks.append(
                f'<line class="axis" x1="{tx:.1f}" y1="50" x2="{tx:.1f}" y2="55"/>'
                f'<text class="tick" x="{tx:.1f}" y="68" text-anchor="{anchor}">{t_txt}</text>'
            )

        svg = (
            '<svg viewBox="0 0 720 76" role="img" aria-label="行为阶段时间轴与转折点">'
            '<defs><clipPath id="referee-stage-clip">'
            '<rect x="14" y="22" width="692" height="24" rx="5"/>'
            '</clipPath></defs>'
            '<g clip-path="url(#referee-stage-clip)">'
            '<rect class="track" x="14" y="22" width="692" height="24"/>'
            + "".join(rects)
            + "</g>"
            + "".join(cp_marks)
            + '<line class="axis" x1="14" y1="52" x2="706" y2="52"/>'
            + "".join(ticks)
            + "</svg>"
        )

        legend_items = "".join(
            f'<span class="cov-chip"><span class="stage-swatch stage-swatch-{ckey}" '
            f'aria-hidden="true"></span><span>{escape(lbl)}</span>'
            f'<strong>{int(round(ratio * 100))}%</strong></span>'
            for (ckey, lbl), ratio in legend_map.items()
        )
        if annotated_note:
            legend_items += f'<span class="cov-chip">首段标记：{escape(annotated_note)}</span>'

        cp_chips = ""
        if cps:
            cp_chips = '<div class="stage-legend">' + "".join(
                f'<span class="cov-chip cov-chip-alert">{cp["time_sec"]:.0f}s 转折：{escape(cp["cue"])}</span>'
                for cp in cps
            ) + "</div>"

        stat_parts = [f"共 {len(segs)} 段"]
        mean_dwell = _finite_float(seg_data.get("mean_dwell_sec"))
        if mean_dwell is not None:
            stat_parts.append(f"平均驻留 {mean_dwell:.1f} 秒")
        flicker = _finite_float(seg_data.get("flicker_reduction_pct"))
        if flicker is not None and flicker >= 0.0:
            stat_parts.append(f"较逐窗识别减少 {flicker:.0f}% 抖动切换")
        stat_html = (
            '<div class="cov-chips">'
            + "".join(f'<span class="cov-chip">{escape(p)}</span>' for p in stat_parts)
            + "</div>"
        )

        summary = _clean_prose(seg_data.get("summary_plain") or "")
        cap_html = f"<figcaption>{escape(summary)}</figcaption>" if summary else ""
        return (
            f"<h3>宏观行为阶段切分</h3>"
            f"{stat_html}"
            f"<figure>{svg}{cap_html}</figure>"
            f'<div class="stage-legend">{legend_items}</div>'
            f"{cp_chips}"
        )

    def _render_attributions_svg(self, attributions: List[Dict[str, Any]]) -> str:
        """渲染证据核验白盒加性归因发散条形图（内联 SVG）。"""
        cleaned: list[dict[str, Any]] = []
        for item in attributions:
            if not isinstance(item, dict):
                continue
            cval = _finite_float(item.get("contribution"))
            if cval is None:
                continue
            lbl = _clean_prose(item.get("label") or "")
            if not lbl:
                continue
            cleaned.append({
                "label": lbl,
                "contribution": cval,
                "direction": _clean_prose(item.get("direction") or ""),
                "value_display": _clean_prose(item.get("value_display") or ""),
            })
        if not cleaned:
            return ""

        cleaned.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        top_items = cleaned[:8]
        max_abs = max(0.001, max(abs(x["contribution"]) for x in top_items))

        top_pad, row_h = 24, 28
        height = top_pad + len(top_items) * row_h + 8
        track_left, track_w, zero_x, half_max, val_x = 190.0, 380.0, 380.0, 180.0, 716.0

        rows_svg: list[str] = [
            f'<text class="tick" x="{zero_x - 10:.1f}" y="14" text-anchor="end">待补强</text>',
            f'<text class="tick" x="{zero_x:.1f}" y="14" text-anchor="middle">0</text>',
            f'<text class="tick" x="{zero_x + 10:.1f}" y="14" text-anchor="start">正向支撑</text>',
            f'<line class="axis" x1="{zero_x:.1f}" y1="{top_pad - 4}" '
            f'x2="{zero_x:.1f}" y2="{height - 4}"/>',
        ]

        for idx, item in enumerate(top_items):
            cy = top_pad + idx * row_h + row_h / 2.0
            contrib = item["contribution"]
            is_pos = item["direction"] == "正向支撑" or (
                item["direction"] != "待补强" and contrib >= 0.0
            )
            mag = abs(contrib)
            bar_w = 2.0 if mag == 0.0 else max(6.0, (mag / max_abs) * half_max)
            bar_x = zero_x if is_pos else zero_x - bar_w
            bar_cls = "attr-pos" if is_pos else "attr-neg"
            signed_str = f"{'+' if contrib >= 0 else ''}{contrib:.2f}"
            right_txt = (
                f"{signed_str} · {item['value_display']}"
                if item["value_display"]
                else signed_str
            )
            dir_txt = item["direction"] or ("正向支撑" if is_pos else "待补强")
            rows_svg.append(
                f'<g><title>{escape(item["label"])} · {escape(dir_txt)} · '
                f'贡献 {signed_str}</title>'
                f'<text class="label" x="0" y="{cy + 4:.1f}">{escape(item["label"])}</text>'
                f'<rect class="track" x="{track_left:.1f}" y="{cy - 7:.1f}" '
                f'width="{track_w:.1f}" height="14" rx="3"/>'
                f'<rect class="{bar_cls}" x="{bar_x:.1f}" y="{cy - 7:.1f}" '
                f'width="{bar_w:.1f}" height="14" rx="3"/>'
                f'<text class="value" x="{val_x:.1f}" y="{cy + 4:.1f}" '
                f'text-anchor="end">{escape(right_txt)}</text></g>'
            )

        svg = (
            f'<svg viewBox="0 0 720 {height}" role="img" aria-label="阶段核验影响因素归因">'
            + "".join(rows_svg)
            + "</svg>"
        )
        return (
            "<h3>阶段核验影响因素归因</h3>"
            f"<figure>{svg}"
            "<figcaption>蓝色向右表示正向支撑因子，橙色向左表示待补强因子；"
            "右侧数字为归因贡献度与对应实测覆盖。</figcaption></figure>"
        )

    def _render_referee_ml_section(self, result: Dict[str, Any]) -> str:
        """渲染第三层行为阶段分段与保形证据核验区域。"""
        ref_ml = (result.get("l1") or {}).get("referee_ml") or {}
        if not isinstance(ref_ml, dict) or not ref_ml.get("available"):
            return ""

        eg = ref_ml.get("evidence_gate") or {}
        stage_code = str(eg.get("stage_code") or "")
        badge_cls = _STAGE_BADGE_CLASSES.get(stage_code, "")
        stage_label = _clean_prose(
            eg.get("stage_label") or _STAGE_DEFAULT_LABELS.get(stage_code, "阶段核验完成")
        )

        badges = [
            f'<span class="stage-badge {badge_cls}">{escape(stage_label)}</span>',
            '<span class="stage-badge stage-info">不打分核验</span>',
        ]
        escalated = bool(eg.get("uncertainty_escalated"))
        ood_flag = bool(eg.get("ood_anomaly_flag"))
        if escalated or ood_flag:
            badges.append('<span class="stage-badge stage-review">建议人工复核</span>')

        cset_labels = [
            _clean_prose(x) for x in (eg.get("conformal_set_labels") or []) if x
        ]
        if not cset_labels and eg.get("conformal_set"):
            cset_labels = [
                _STAGE_DEFAULT_LABELS.get(str(c), "待定阶段")
                for c in eg.get("conformal_set") or []
            ]
        cset_text = " / ".join(cset_labels) if cset_labels else "暂无"

        target_cov = _finite_float(eg.get("target_coverage"))
        target_pct = int(round((target_cov if target_cov is not None else 0.90) * 100))
        cov_text = f"目标覆盖率 {target_pct}%"
        emp_cov = _finite_float(eg.get("empirical_cv_coverage"))
        if emp_cov is not None:
            cov_text += f"，交叉验证实证覆盖率 {emp_cov * 100:.1f}%"

        chips = [
            f'<span class="cov-chip">保形候选集合：{escape(cset_text)}</span>',
            f'<span class="cov-chip">{escape(cov_text)}</span>',
        ]
        if escalated:
            chips.append('<span class="cov-chip cov-chip-warn">不确定性升级：建议人工复核边界项</span>')
        if "ood_anomaly_flag" in eg:
            ood_cls = " cov-chip-warn" if ood_flag else ""
            ood_txt = "已检出偏离" if ood_flag else "未见异常"
            chips.append(f'<span class="cov-chip{ood_cls}">分布偏离检测：{ood_txt}</span>')

        eg_summary = _clean_prose(eg.get("summary_plain") or "")
        summary_html = f"<p>{escape(eg_summary)}</p>" if eg_summary else ""

        seg_data = ref_ml.get("stage_segmentation") or {}
        timeline_html = ""
        if isinstance(seg_data, dict):
            if seg_data.get("available", True) is not False:
                timeline_html = self._render_stage_timeline_svg(seg_data)
            if not timeline_html:
                seg_note = _clean_prose(seg_data.get("summary_plain") or "")
                if seg_note:
                    timeline_html = f'<p class="note">{escape(seg_note)}</p>'

        attrs_html = self._render_attributions_svg(eg.get("attributions") or [])

        return (
            "<h2>行为阶段分段与证据核验</h2>"
            '<div class="referee-card">'
            f'<div class="referee-head">{"".join(badges)}</div>'
            '<p class="note">仅作行为阶段切分与证据核验，不输出分数或个体评价。</p>'
            f"{summary_html}"
            f'<div class="cov-chips">{"".join(chips)}</div>'
            f"{timeline_html}"
            f"{attrs_html}"
            "</div>"
        )

    def _render_dual_calibration_appendix(self, l1: Dict[str, Any]) -> str:
        """渲染面部双仪器一致性核对与头部姿态校准对照折叠附录。"""
        dual_inst = l1.get("face_dual_instrument") or {}
        bs_cal = l1.get("blendshape_calibration") or {}
        has_dual = bool(isinstance(dual_inst, dict) and dual_inst.get("available"))
        has_cal = bool(isinstance(bs_cal, dict) and bs_cal.get("available"))
        if not has_dual and not has_cal:
            return ""

        parts: list[str] = []
        if has_dual:
            parts.append("<h3>双仪器逐构念一致性核对</h3>")
            d_sum = _clean_prose(dual_inst.get("summary_plain") or "")
            if d_sum:
                parts.append(f'<p class="note">{escape(d_sum)}</p>')
            pairs = dual_inst.get("pairs") or []
            if pairs:
                pair_rows: list[str] = []
                for p in pairs:
                    if not isinstance(p, dict):
                        continue
                    lvl = _clean_prose(p.get("agreement_level") or "通道未激活")
                    p_cls = _AGREEMENT_PILL.get(lvl, "pill-neutral")
                    lbl = _clean_prose(p.get("label") or "")
                    nv = p.get("n_valid")
                    nv_str = str(int(nv)) if isinstance(nv, (int, float)) and math.isfinite(float(nv)) else "—"
                    pair_rows.append(
                        f"<tr><td><strong>{escape(lbl)}</strong></td>"
                        f'<td><span class="pill-tag {p_cls}">{escape(lvl)}</span></td>'
                        f'<td class="num">{_fmt_rho(p.get("spearman_rho"))}</td>'
                        f'<td class="num">{_fmt_rho(p.get("delta_spearman_rho"))}</td>'
                        f'<td class="num">{nv_str}</td></tr>'
                    )
                parts.append(
                    "<table><thead><tr><th>行为构念</th><th>一致等级</th>"
                    "<th>逐帧秩相关</th><th>变化趋势相关</th><th>有效帧</th></tr></thead>"
                    f"<tbody>{''.join(pair_rows)}</tbody></table>"
                )
            diag = dual_inst.get("au15_pitch_diagnostic") or {}
            if isinstance(diag, dict) and diag.get("available") and diag.get("diagnostic_note"):
                parts.append(
                    f'<div class="diag-box">{escape(_clean_prose(diag["diagnostic_note"]))}</div>'
                )

        if has_cal:
            parts.append("<h3>时序平滑与头部姿态校准对照</h3>")
            pose_mode = str(bs_cal.get("pose_mode") or "unavailable")
            pose_label, pose_cls = _POSE_MODE_META.get(pose_mode, _POSE_MODE_META["unavailable"])
            n_clean = bs_cal.get("n_clean_frames", 0)
            n_raw = bs_cal.get("n_raw_frames", 0)
            qpr = _finite_float(bs_cal.get("quality_pass_rate"))
            qpr_txt = f" · 通过率 {qpr * 100:.1f}%" if qpr is not None else ""
            parts.append(
                f'<div class="cov-chips">'
                f'<span class="pill-tag {pose_cls}">{pose_label}</span>'
                f'<span class="cov-chip">有效帧 {n_clean} / {n_raw}{qpr_txt}</span>'
                f"</div>"
            )
            c_sum = _clean_prose(bs_cal.get("summary_plain") or "")
            if c_sum:
                parts.append(f'<p class="note">{escape(c_sum)}</p>')
            constructs = bs_cal.get("constructs") or []
            if constructs:
                c_rows: list[str] = []
                for c in constructs:
                    if not isinstance(c, dict):
                        continue
                    lbl = _clean_prose(c.get("label") or "")
                    note = _clean_prose(c.get("note") or "")
                    note_html = f'<div class="note">{escape(note)}</div>' if note else ""
                    delta = _finite_float(c.get("rho_delta"))
                    if delta is None:
                        delta_html = "—"
                    elif delta > 0:
                        delta_html = f'<span class="delta-pos">+{delta:.3f}</span>'
                    else:
                        delta_html = f"{delta:.3f}"
                    r2 = _finite_float(c.get("pose_r2"))
                    r2_txt = f"{r2 * 100:.1f}%" if r2 is not None else "—"
                    elig = _clean_prose(c.get("eligibility") or "待3D重采验证")
                    e_cls = _ELIGIBILITY_PILL.get(elig, "pill-warn")
                    c_rows.append(
                        f"<tr><td><strong>{escape(lbl)}</strong>{note_html}</td>"
                        f'<td class="num">{_fmt_rho(c.get("raw_rho"))}</td>'
                        f'<td class="num">{_fmt_rho(c.get("calibrated_rho"))}</td>'
                        f'<td class="num">{delta_html}</td>'
                        f'<td class="num">{r2_txt}</td>'
                        f'<td><span class="pill-tag {e_cls}">{escape(elig)}</span></td></tr>'
                    )
                parts.append(
                    "<table><thead><tr><th>面部动作</th><th>校准前一致度</th>"
                    "<th>校准后一致度</th><th>变化量</th><th>头姿解释占比</th>"
                    "<th>融合判定</th></tr></thead>"
                    f"<tbody>{''.join(c_rows)}</tbody></table>"
                )
            disc = _clean_prose(bs_cal.get("disclaimer") or "")
            if disc:
                parts.append(f'<p class="note">{escape(disc)}</p>')

        return (
            "<details>"
            "<summary>附录：面部双仪器一致性与头部姿态校准对照</summary>"
            f"{''.join(parts)}"
            "</details>"
        )

    def _render_feature_layer_v2_appendix(self, l1: Dict[str, Any]) -> str:
        """渲染第二层多通道动力学特征摘要折叠附录。"""
        fv2 = l1.get("feature_layer_v2") or {}
        if not isinstance(fv2, dict) or not fv2.get("available"):
            return ""

        parts: list[str] = []
        tags = [_clean_prose(t) for t in (fv2.get("framework_tags") or []) if t]
        if tags:
            parts.append(
                '<div class="pill-tags">'
                + "".join(f'<span class="pill-tag">{escape(t)}</span>' for t in tags)
                + "</div>"
            )

        summary = _clean_prose(fv2.get("summary_plain") or "")
        if summary:
            parts.append(f'<p class="note">{escape(summary)}</p>')

        highlights = fv2.get("highlights") or []
        if highlights:
            h_rows: list[str] = []
            for h in highlights:
                if not isinstance(h, dict):
                    continue
                lbl = _clean_prose(h.get("label") or "")
                unit = _clean_prose(h.get("unit") or "")
                val_txt = _fmt_v2_highlight(h.get("value"), unit)
                track_key = str(h.get("track") or "")
                track_lbl = _V2_TRACK_LABELS.get(track_key, "其他动力学通道")
                note = _clean_prose(h.get("note") or "") or "—"
                h_rows.append(
                    f"<tr><td><strong>{escape(lbl)}</strong></td>"
                    f'<td class="num">{val_txt}</td>'
                    f'<td><span class="pill-tag pill-neutral">{escape(track_lbl)}</span></td>'
                    f"<td>{escape(note)}</td></tr>"
                )
            parts.append(
                "<table><thead><tr><th>指标名称</th><th>实测读数</th>"
                "<th>来源通道</th><th>说明</th></tr></thead>"
                f"<tbody>{''.join(h_rows)}</tbody></table>"
            )

        n_streams = (((fv2.get("tracks") or {}).get("catch22_dynamics") or {}).get("n_streams") or 0)
        infer_ms = _finite_float(fv2.get("inference_ms"))
        meta_bits: list[str] = []
        if isinstance(n_streams, (int, float)) and n_streams > 0:
            meta_bits.append(f"已提取 {int(n_streams)} 条多通道时间序列动力学特征")
        if infer_ms is not None:
            meta_bits.append(f"耗时 {infer_ms:.1f} ms")
        if meta_bits:
            parts.append(f'<p class="note">{" · ".join(meta_bits)}</p>')

        return (
            "<details>"
            "<summary>附录：第二层多通道动力学特征摘要</summary>"
            f"{''.join(parts)}"
            "</details>"
        )

    def _render_scene_profile_section(self, result: Dict[str, Any]) -> str:
        """渲染场景档案摘要与合规边界条（零术语/零符号/零禁用词）。"""
        sp = result.get("scene_profile")
        if not isinstance(sp, dict) or not sp.get("name"):
            return ""
        name = _clean_prose(sp.get("name") or "")
        weight = _clean_prose(sp.get("weight_label") or "")
        banner = _clean_prose(sp.get("compliance_banner") or "")
        ext = sp.get("extensions") or {}
        ext_html = ""
        meta_ext = ext.get("metacognition_text") if isinstance(ext, dict) else None
        if isinstance(meta_ext, dict) and isinstance(meta_ext.get("indicators"), list):
            rows_html = []
            for item in meta_ext["indicators"]:
                if not isinstance(item, dict):
                    continue
                lbl = _clean_prose(item.get("label") or "")
                val = item.get("value")
                unit = _clean_prose(item.get("unit") or "")
                val_txt = "缺口暂不可给" if val is None else f"{_fmt_num(val)} {unit}".strip()
                conf = _clean_prose(item.get("confidence") or "无")
                summary = _clean_prose(item.get("summary") or "")
                rows_html.append(
                    f"<tr><td><strong>{escape(lbl)}</strong></td>"
                    f"<td class=\"num\">{escape(val_txt)}</td>"
                    f"<td>{escape(conf)}</td>"
                    f"<td>{escape(summary)}</td></tr>"
                )
            if rows_html:
                ext_html = (
                    "<h3>审辨修正扩展观测量</h3>"
                    "<table><thead><tr><th>扩展观测量</th><th>实测读数</th>"
                    "<th>置信度</th><th>说明</th></tr></thead>"
                    f"<tbody>{''.join(rows_html)}</tbody></table>"
                )

        overrides = sp.get("explicit_blocked_overrides") or []
        ov_chips = "".join(
            f'<span class="cov-chip">主动封停：{escape(_clean_prose(o.get("label") or ""))}</span>'
            for o in overrides
            if isinstance(o, dict) and o.get("label")
        )
        head_tag = f"{name}（{weight}）" if weight else name
        banner_p = f'<p class="note">{escape(banner)}</p>' if banner else ""
        chips_div = f'<div class="cov-chips">{ov_chips}</div>' if ov_chips else ""
        return (
            f'<div class="referee-card">'
            f'<div class="referee-head"><span class="stage-badge stage-info">当前场景：{escape(head_tag)}</span></div>'
            f"{banner_p}{chips_div}{ext_html}</div>"
        )

    def _build_html_report(self, result: Dict[str, Any], chart_paths: Dict[str, Any],
                           features: Dict[str, Any], data: Dict[str, pd.DataFrame],
                           static_images: List[str],
                           sources: Optional[Dict[str, Dict]] = None,
                           target: Optional[str] = None) -> str:
        """整份交付物。**一个文件、内联 SVG、先人话后表格。**

        ## 2026-10-26 重做的四件事(使用者原话见模块顶)

        | 原来 | 现在 |
        |---|---|
        | 只有三列表格 | 开头一段**从载荷算出来的**结论句,再往下才是表 |
        | 就一张雷达图(半径 = 过了几个槽) | 两张内联 SVG:每维覆盖条形 + 逐题离散度 |
        | `<iframe>` 引 4.8 MB 的 plotly 文件 | 图内联在报告里,拷走一份就是一个文件 |
        | 引擎迁移说明印在正文(读者读成"跟上一次比") | 收进折叠附录,正文只留本场自己的限制 |

        `chart_paths` / `features` / `data` / `static_images` 四个参数**保留但不用**:
        `_build_html_report(result, {}, features, {}, [])` 有 5 个位置参数的老调用方
        (`tests/test_report_layer.py`),签名不能改。
        """
        deep_analysis_html = self._generate_deep_text_analysis(features, result)

        # 报告头第一件事:**这一场是谁的**。此前只有会话号(一串机器号),同一批
        # 十几个人录下来,十几份报告在盘上分不出谁是谁(使用者 2026-10-26)。
        subject_html = _subject_block(target)

        # 头部聚合呈现:只讲事实(spec §5.4 / §5.6)。
        # 原分数卡渲染「点分 + 五档评语」(如「0.0 / 综合行为观测评分 / 待提升」)——
        # 那是把未标定标尺上的复合点分当对候选人的评定。改为:覆盖事实 + 依据 +
        # 区间,区间只能来自本场会话自己的测量(§5.5:没有真实常模就不给位置)。
        coverage = result["coverage"]
        # ⚠️ 2026-10-26:「依据:通过证据门的指标」那张平铺列表**整段删掉**。
        #
        # 它把每一个过门槽再列一遍(维度 · 指标 · 原始值 · 区间 · 样本量),
        # 而「每一维量到了什么」那四张表逐字包含同样的五列 —— 同一批数字
        # 在一份报告里印两遍。使用者原话:「简洁易懂」。
        #
        # 它还有一处更糟:每行都带一句「本场会话内观测区间 未采集到会话内变异信息
        # (与原始值同量纲)」,读者会以为**每个指标都缺了区间**,而真相是
        # L1 每个指标整场只有一个值,本来就没有"场内变异"。
        #
        # 覆盖数(几 / 几)在报告头的 hero 里,逐维明细在维度表里,
        # 这条信息链没有断 —— 断的只是重复。
        basis_html = ""

        fig_cov = self._figure_coverage(result)
        fig_disp = self._figure_dispersion(result)
        n_disp = len(((result.get("l1") or {}).get("dispersion") or {}))
        n_win = ((result.get("l1") or {}).get("windows") or {}).get("n_questions")

        # 逐题离散度这一节**总要出现**,哪怕没有数 —— 空着消失会让人以为这一层不存在。
        # 有数:图 + 图注。没数:说清为什么没有(整场一个窗口 / 每维只测了一次),
        # 而不是留一个空标题。
        if fig_disp:
            q = f"{n_win} 道题" if n_win else "各题"
            n_flat = self._n_flat_rows(result)
            flat_note = ""
            if n_flat:
                flat_note = (f"\n            其中 {n_flat} 条在各题上是<b>同一个值</b>"
                             f"(标着「各题相同」),它们是最稳的那一批;"
                             f"值本身是 0 的,意思是每题实测均为零或未触发事件,不是没测到。")
            disp_block = f"""
            <h2>题与题之间稳不稳</h2>
            <p class="note">同一个指标,{q}里各量了一次 —— 下图每一行是一条指标,
            左端是本场最小值、右端是最大值,圆点是中位数(右侧数字)。
            <strong>每行各有自己的刻度</strong>,所以请横向看「这一行散不散」,
            不要拿两行的长短互相比较。</p>
            <figure>{fig_disp}
            <figcaption>{len(self._prepare_dispersion_rows(result))} 条指标在题与题之间有两次以上测量。
            只测了一次的指标不出现在这里 —— 单次测量没有「离散」可言,
            给个 0 会把「没测」说成「很稳」。{flat_note}</figcaption></figure>
            """
        else:
            if not n_win:
                why = "这一场是整场一个窗口，没有逐题切分"
            elif n_win == 1:
                why = "本场仅有 1 道题，单题没有题间离散度可言"
            else:
                why = f"虽然有 {n_win} 道题，但各指标有效出值的题数均不足 2 题（存在缺测或仅单题出值）"
            disp_block = f"""
            <h2>题与题之间稳不稳</h2>
            <p class="note">这一场报不出逐题离散度:{why},所以没有题与题之间的
            中位数、最小值、最大值可给。下面的所有读数都是<b>整场一个值</b>,
            不是「每题一个值取平均」—— 两者不能混着读。</p>
            """

        cov_block = ""
        if fig_cov:
            cov_block = f"""
            <h2>{self._coverage_heading(result)}</h2>
            <p class="note">每一维要量的条数不同(见右端的「量到 / 一共」),
            所以请比「自己那一维填了多少」,不要把两维的条长直接比大小。</p>
            <figure>{fig_cov}
            <figcaption>条形 = 这一维量到可用数的条数;浅色轨道 = 这一维一共要量几条。
            没量到的那些与原因逐条列在下面的维度明细里。</figcaption></figure>
            """

        # ★ 区间那一列为什么整份报告都没有 —— **整份说一次**,不是每个维度说一次。
        #
        # L1 每个指标整场只有一个值,`observed_interval` 恒为空 ⟹ 四张表都不印这一列。
        # 原先那句话写在每个维度下面,同一句话出现四遍(实测渲染出来的正文)。
        # 判据是"全部维度都没有区间",有一条有就照印、不说这句。
        all_dims = list((result.get("dimensions") or {}).values())
        no_interval_anywhere = bool(all_dims) and not any(
            e.get("observed_interval") for d in all_dims
            for e in (d.get("evidence_chain") or []))
        interval_note = ("" if not no_interval_anywhere else
                         "<p class=\"note\">下面每张表都不列「本场会话内观测区间」:"
                         "每个指标在本场只测了一次,没有「场内的变异」可报。</p>")

        # 「与旧引擎报告对比」是**引擎迁移说明**,每份报告都有、与这一场无关。
        # 印在正文里必然被读成"跟上一次比怎么样"(使用者原话)。内容不删,收进附录。
        appendix = self._render_l1_audit_text(result, only=("comparability",))
        appendix_block = ""
        if appendix:
            appendix_block = f"""
            <details>
                <summary>附录：与旧引擎报告的对照说明（给复核数据的人看，与这一场无关）</summary>
                <p class="note">这一节讲的是<b>两代引擎</b>怎么量同一件事,
                每一份报告都会印。它不描述这一场发生了什么,不必当成「跟上一次比」。</p>
                {appendix}
            </details>
            """

        # ★ A5(2026-11-21):仪器来源披露那一节,放在正文末尾、页脚之前。
        #   放在**最后**而不是报告头:读者先读到"量到了什么",再问"这些数谁量的" ——
        #   反过来会把一份交付物读成仪器说明书。但它必须在**页脚之前**,
        #   所以是"页脚那一块"而不是附录(附录会被 `<details>` 折起来)。
        instrument_block = self._render_instrument_disclosure(result)
        notices_block = self._render_evidence_chain_notices(result)
        scene_block = self._render_scene_profile_section(result)
        referee_block = self._render_referee_ml_section(result)
        extra_appendices = self._render_extra_appendices(result)

        return _zh_punct(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{brand.REPORT_HTML_TITLE}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>{brand.REPORT_HTML_TITLE}</h1>
  {subject_html}
  <div class="meta">{_meta_line(target, result['model_metadata']['version'])}</div>
  <div class="sources">{sources_disclosure(sources or {}, target,
                                            modality_summary=(result.get("l1") or {})
                                            .get("modality_summary"))}</div>
</header>

<div class="hero">
  <span class="cover-number">{self._coverage_hero_number(coverage)}</span>
  <span class="hero-cap">个可测量槽通过 · 置信度上限：{coverage['confidence_cap']}</span>
</div>
{self._coverage_line(coverage)}
{self._coverage_explanation(coverage)}
<p class="lead">{self._lead_sentence(result)}</p>
<p class="note">{result['summary_narrative']}</p>
{scene_block}
{notices_block}

{cov_block}
{disp_block}
{referee_block}

<h2>每一维量到了什么</h2>
{self._render_vanished_dimensions(result)}
{interval_note}
{deep_analysis_html}

{basis_html}

{self._render_l1_audit_text(result, only=("gaps", "low_confidence", "windows"))}

{instrument_block}

{appendix_block}
{extra_appendices}

<footer>{brand.REPORT_HTML_FOOTER}</footer>
</div>
</body>
</html>
""")


def main(argv: Optional[List[str]] = None) -> int:
    """CLI 入口:`--session-id` 定要描述的那一场(M2:可以显式指定;不给就取最新一场)。

    **没产出报告就返回非 0**(第 17 条 / 复审 I2 的额外症状):总控面板
    `app.py:run_script` 只看子进程的 returncode —— "什么都没生成但 exit 0"
    会被显示成「任务完成!」。
    """
    import argparse

    ap = argparse.ArgumentParser(description="生成行为观测报告")
    ap.add_argument("--session-id", default=None,
                    help="要描述的那场会话 id;缺省取最新一场")
    args = ap.parse_args(argv)
    return 0 if ReportGenerator().generate_report(args.session_id) else 1


if __name__ == "__main__":
    raise SystemExit(main())