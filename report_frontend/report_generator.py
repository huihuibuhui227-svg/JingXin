# report_frontend/report_generator.py

import os
import re
import webbrowser
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
from .data_loader import LogDataLoader
from . import l1_mapper, l1_quality
from .l1_report import build_l1_report


#: 维度数 → 中文数字。只到六维就够(当前最多五维,消失一维后四维);
#: 表外的数原样印阿拉伯数字 —— 宁可印 `7 个维度`,不许印一个错的中文数。
_CN_NUM = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六"}


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
    if x != x:                       # NaN
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
    if x != x:                        # NaN
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
    """一个 **0 值**旁边那句解释。非 0 返回空串。

    ## 为什么 0 必须解释(2026-10-26 实测)

    使用者那一场里 `smile_ratio` 是 `0.0`、`n_valid` 是 64 —— 表格上就印一个 `0`。
    读者看到它会有两种读法,而两种都与事实不符:

    * 「没测到」—— 不,测了 64 帧,分母是 542 帧;
    * 「测出来是零」—— 也不精确,它是**一次都没有越过阈值**。

    这两句对"这一场能说明什么"的含义完全不同,所以 0 旁边必须带一句人话。
    ⚠️ 只对**真测过**的 0 说这句话;`n_valid` 为 0 的 0 是"没测到",那由缺口行负责,
    不归这里(两处都说就成了同一件事说两遍)。
    """
    v = item.get("raw_value")
    if v is None:
        return ""
    try:
        if float(v) != 0:
            return ""
    except (TypeError, ValueError):
        return ""
    n = item.get("n_valid")
    if n is None:
        return ""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    return f"<span class=\"zero-note\">看过 {n} 个有效观测，一次也没超过阈值</span>"


#: 模态键 → 报告里给人看的名字(键名来自 data_loader 的统一化,见那边的 key 映射)。
#:
#: ⚠️ `voice` 这一项是**模态级**名字,与上面两个文件名不同:日志按「面试 / 科研」
#:    分两个文件,而 L1 的 V 族不分。`modality_summary` 用的是模态名,所以要有它。
_MODALITY_LABELS = {
    "face": "面部",
    "gesture": "手势",
    "voice": "语音",
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
}
@media (prefers-color-scheme: dark){
  :root{ --page:#141413; --surface:#1b1b19;
         --ink:#f3f2ed; --ink-2:#bab8b1; --ink-3:#8c8a84;
         --rule:#302f2b; --track:#2a2926;
         --series-1:#3987e5; --series-2:#d95926; }
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
  font-family:"Noto Sans SC","Microsoft YaHei",system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:15px;line-height:1.75;-webkit-font-smoothing:antialiased}
.wrap{max-width:860px;margin:0 auto;padding:48px 24px 72px}
header{border-bottom:1px solid var(--rule);padding-bottom:20px;margin-bottom:32px}
h1{font-size:24px;line-height:1.3;margin:0 0 10px;font-weight:650;letter-spacing:.01em}
.meta{color:var(--ink-3);font-size:13px}
.sources{margin-top:14px;font-size:13px;color:var(--ink-2)}
h2{font-size:17px;font-weight:650;margin:44px 0 14px}
h3{font-size:15px;font-weight:600;margin:26px 0 8px}
p{margin:0 0 12px}
.lead{font-size:16.5px;line-height:1.7;max-width:44em;margin-bottom:22px}
.hero{display:flex;align-items:baseline;gap:14px;margin:0 0 10px;flex-wrap:wrap}
.cover-number{font-size:52px;font-weight:700;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;line-height:1}
.hero-cap{color:var(--ink-2);font-size:14px}
figure{margin:18px 0 6px}
figcaption{color:var(--ink-3);font-size:12.5px;margin-top:10px}
svg{display:block;width:100%;height:auto;overflow:visible}
.label{fill:var(--ink-2);font-size:12.5px}
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
"""


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
        lines.append(f'<em>另有 NONE 桶 {none_bucket["rows"]} 行 —— '
                     f'那些请求没带 session_id，不属于本场，未参与计算。</em>')

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

    def generate_report(self, session_id: Optional[str] = None) -> str:
        """从磁盘 CSV 文件生成评估报告（批处理模式）。

        `session_id`:本报告要描述的那一场(M2)。**不给就取最新一场**,并在报告头写明是谁
        —— 这个参数在 M2 之前是"收下就丢"的。
        """
        print("\n" + "=" * 70)
        print("🚀 启动 JingXin 面试行为观测报告生成系统 (批量模式)")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"Research_Assessment_Report_{timestamp}.html"
        report_path = os.path.join(self.output_dir, report_filename)

        try:
            loader = LogDataLoader()
            data = loader.get_fused_latest_data(session_id)
            if not data:
                # 本场一个模态都没读到:**照样出报告**(spec §6 行 1/2 / 第 17 条)——
                # 头里逐条列出缺什么,而不是让面板显示"什么都没发生"。
                # 实测:空 data 下 L1 给 覆盖 0/14、逐维缺口照常列出、图表照常生成。
                print("   ⚠️  本场没有任何可用日志,仍生成一份只含缺口的报告")

            # ★ 与 `/api/report/structured` **同一个引擎**(2026-10-25)。
            #
            #   2026-10-18 前端那条端点已切到 L1,而这里一直留在旧引擎
            #   (`PsychologicalFeatureEngine` → `ResearchCapabilityMapper`)。
            #   后果:同一场会话在界面上给出**两个互相矛盾的数** —— 前端报告页读
            #   structured(2026-10-25 实测 7/14),而这里产出的 HTML 说 11/14。
            #   差的那 4 个槽正是旧引擎把「手掌张开度 / 肩部放松度 / 眼神接触」算成
            #   "填上了",而 L1 判定**当前管线根本不测**(见 `docs/下一步.md` 第 29 条)。
            #   两份东西在「历史报告」列表里并排,谁也看不出它们量的是两套口径。
            #
            #   对照该在**数据层**做(`?engine=legacy` 仍在),不该让交付物分叉。
            #   钉子:`tests/test_html_report_uses_the_l1_engine.py`。
            result = build_l1_report(loader.target_session, logs=data)
            static_images = self._scan_static_images()

            # ⚠️ 2026-10-26:`ReportVisualizer` 不再参与交付物。
            #
            # 它只产一张雷达图,半径是「每个维度过了几个槽」(3/2/0/0/1) ——
            # 那不是任何行为量,而且落成 `radar_chart_<epoch>.html`(单份 4.8 MB,
            # 内含整包 plotly),报告再用 `<iframe>` 引它 ⟹ **报告一拷走图就没了**。
            # 使用者原话:「就一个雷达图,完全没有分析」。
            #
            # 现在图在 `_build_html_report` 里用内联 SVG 画:一份报告就是一个文件。
            # `visualizer.py` 本身留着(它的单元测试还在,`?engine=legacy` 的对照
            # 与历史产物清理也要它),只是**交付物这条路不再经过它**。
            html_content = self._build_html_report(result, {}, {}, data,
                                                   static_images,
                                                   sources=loader.selected_sessions,
                                                   target=loader.target_session)

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

        except Exception as e:
            print(f"❌ 错误：{e}")
            return ""

        # 报告**已经落盘**。打开浏览器是"顺手"的副作用 —— 它失败不许影响返回值:
        # 退出码是本轮新引入的(第 17 条),一旦从这里抛出去,面板会把"报告其实写好了"
        # 报成「任务失败」(假警报)。
        try:
            webbrowser.open('file://' + os.path.realpath(report_path))
        except Exception as e:
            print(f"   ⚠️  报告已生成,但打开浏览器失败：{e}")
        print(f"\n✅ 报告已生成：{report_path}")
        return report_path

    def generate_report_live(self, session_id: str) -> str:
        """从运行中的 API 服务获取实时内存数据，生成评估报告（实时模式）"""
        print("\n" + "=" * 70)
        print("🚀 启动 JingXin 面试行为观测报告生成系统 (实时模式)")
        print(f"📋 会话 ID: {session_id}")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"Research_Assessment_Report_Live_{timestamp}.html"
        report_path = os.path.join(self.output_dir, report_filename)

        try:
            loader = LogDataLoader()
            data = loader.get_live_data(session_id)
            if not data:
                raise ValueError("无法从 API 获取实时数据，请确认三个分析服务 (8000/8001/8002) 已启动且会话存在")
            if 'face' not in data:
                print("   ⚠️  未获取到面部数据，继续使用其他模态生成报告")

            # 与批量路径同一个引擎(理由见 `generate_report` 那段注释)。
            # 实时路径的 `logs` 来自三个服务的 `/summary`,形状与读盘那份相同。
            result = build_l1_report(session_id, logs=data)
            static_images = self._scan_static_images()

            # 与批量路径同一条:图在报告里内联画(理由见 `generate_report` 那段)。
            # 实时路径的模态是**按这一个人的 session_id 去内存里取的**,不存在"每模态取最新"
            # 那种跨场拼接;所以来源就是这一个 id(逐个列出真正取到数据的模态)。
            html_content = self._build_html_report(result, {}, {}, data,
                                                   static_images,
                                                   sources=live_sources(session_id, data),
                                                   target=session_id)

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            webbrowser.open('file://' + os.path.realpath(report_path))
            print(f"\n✅ 实时报告已生成并打开：{report_path}")
            return report_path

        except Exception as e:
            print(f"❌ 错误：{e}")
            return ""

    def _scan_static_images(self) -> List[str]:
        images = []
        if os.path.exists(self.output_dir):
            for f in os.listdir(self.output_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')) and 'trajectory' not in f and 'chart' not in f:
                    images.append(f)
        return images

    def _render_dimension_block(self, dim_key: str, dim: Dict[str, Any]) -> str:
        """渲染单个维度的证据状态。不解读,不推断,不加形容词。

        形态 = spec §5.4 的「整节的正确形态」:值 + 有效样本量 + 置信度 + evidence_gaps。
        出分维度另带 §5.6 要求的提示句「该维度目前无独立效标,仅供行为描述」。
        **不再渲染"归一值"** —— 它是未标定标尺上的点分,与档位标签一样会被读成
        对该维度的评定,而本系统没有标定样本能给这个刻度背书。
        表格给出的"本场会话内观测区间"只由本场测量构成(均值 ± 会话内标准差)。

        每个维度自己带全自己的缺口 —— 出分维度也要列出未过门的槽。
        否则那些缺口只能靠顶层的汇总段兜底,而汇总段的所有条目都已被逐维列表
        覆盖过一遍(evidence_gaps 就是各维缺口的并集),同一批字符串会被打印两次。
        """
        # ⚠️ 分支判据 2026-10-04 由 `dim["score"] is None` 改成 `not dim["evidence_chain"]`
        #    —— payload 里的点分已摘除(见 `research_mapper` 返回处那段注释)。
        #    两者**等价**:`score` 为 None 当且仅当该维一个槽都没过门,而那正是
        #    `evidence_chain` 为空。判据从"有没有分"改成"有没有证据",本来也更贴本仓的主张。
        if not dim["evidence_chain"]:
            gaps = "".join(f"<li>{g}</li>" for g in dim.get("evidence_gaps", []))
            return f"""
            <h3>{dim['display_name']}</h3>
            <p><strong>证据不足</strong> —— 本次未采集到足以评估该行为线索的有效样本。</p>
            <ul>{gaps}</ul>
            """

        rows = "".join(
            f"<tr><td>{e['human_name']}</td>"
            f"<td class=\"num\">{_fmt_num(e['raw_value'])}{_zero_note(e)}</td>"
            f"<td>{_interval_text(e.get('observed_interval'))}</td>"
            f"<td class=\"num\">{e.get('n_valid', '—')}</td></tr>"
            for e in dim["evidence_chain"]
        )
        # 先给人话,再给表。表格是核对用的,而"这一维量到了哪几条"一句话就说完 ——
        # 读者不该为了知道"有没有测到东西"去数表格行数(使用者原话:
        # 「完全没有分析」;实测那版全篇只有三列表格)。
        names = "、".join(str(e["human_name"]) for e in dim["evidence_chain"])
        gaps = "".join(f"<li>{g}</li>" for g in dim.get("evidence_gaps", []))
        gaps_block = f"<p>未过门的指标：</p><ul>{gaps}</ul>" if gaps else ""
        return f"""
        <h3>{dim['display_name']}</h3>
        <p>过门指标 {dim['matched_indicators']}；置信度：<strong>{dim['confidence']}</strong>。
        （该维度目前无独立效标，仅供行为描述）</p>
        <p class="note">量到了这几条：{names}。</p>
        <table><thead><tr><th>指标</th><th>原始值</th><th>本场会话内观测区间</th>
        <th>有效样本量</th></tr></thead>
        <tbody>{rows}</tbody></table>
        {gaps_block}
        """

    def _generate_deep_text_analysis(self, features: Dict[str, Any], result: Dict[str, Any]) -> str:
        """只陈述本次实际测到了什么,不做任何解读、推断与形容词修饰。

        每个维度按 spec §5.4 的「整节的正确形态」渲染,出分维度带 §5.6 的"无独立效标"
        提示。本函数**不再输出聚合摘要**:它曾印「综合科研潜力评分为 X 分,评级为 Y」,
        后改印「综合行为观测摘要:{档位}(置信度上限:…)」—— 两者都是把未标定标尺上的
        点分与五档评语当对人的评定(实测报告头渲染出「0.0 / 综合行为观测评分 / 待提升」)。
        聚合呈现移入报告头部的覆盖卡,只给「区间 + 置信度 + 依据」(spec §5.6)。

        参数 features 保留是为了调用方签名稳定;本函数不再从原始特征里另取默认值。
        """
        parts = []
        for dim_key, dim in result["dimensions"].items():
            parts.append(self._render_dimension_block(dim_key, dim))
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
            f"(声明了 {v['n_slots']} 个指标槽,全部没有同构念的 L1 指标)"
            f"<ul>" + "".join(
                f"<li>{l1_mapper.slot_label(d['slot'])}:本层不测这一条 —— "
                f"{l1_mapper.drop_reason(d['slot'])}</li>"
                for d in v.get("dropped", [])) + "</ul></li>"
            for v in vanished)
        return f"""
        <div id="vanished-dimensions">
            <h3>整维未出(当前管线没有对应的可测量)</h3>
            <p>下面这一维<strong>不是本场没测到</strong>,是它要量的东西当前管线都不测 ——
            再采一百次也不会有,别等它。</p>
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
        """
        n = len(result.get("dimensions") or {})
        return f"{_CN_NUM.get(n, n)}维证据覆盖（{n} 个维度）"

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
        """
        cov = result.get("coverage") or {}
        dims = result.get("dimensions") or {}
        n_passed = cov.get("n_passed", 0)
        n_slots = cov.get("n_slots", 0)
        n_dropped = cov.get("n_dropped", 0)

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
            sentences.append(
                f"这一场在 {len(dims)} 个维度上一个指标槽都没过证据门"
                f"(共 {n_slots} 个槽)—— 不是「没测」,是测到的还不足以支撑读数,"
                f"逐条原因见下。")
        else:
            best = ranked[0]
            b_got, b_tot = _pair(best)
            sentences.append(
                f"这一场在 {len(dims)} 个维度上量到了 <strong>{n_passed} / {n_slots}</strong> "
                f"个指标槽,其中最完整的是「{best.get('display_name', '')}」"
                f"({b_got} / {b_tot})。")
            if len(ranked) > 1:
                worst = ranked[-1]
                w_got, w_tot = _pair(worst)
                if (w_got, w_tot) != (b_got, b_tot):
                    sentences.append(
                        f"最薄的是「{worst.get('display_name', '')}」({w_got} / {w_tot})。")

        if n_dropped:
            sentences.append(
                f"另有 {n_dropped} 个槽在本层没有对应的可测量,已整条剔除 —— "
                f"它们不是「这一场没测到」,见下方「整维未出」。")
        cap = cov.get("confidence_cap")
        if cap and cap != "无":
            sentences.append(f"所有读数的置信度上限是「{cap}」。")
        return "".join(sentences)

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
            f'aria-label="每个维度通过证据门的指标槽数">']
        for i, (name, got, total) in enumerate(rows):
            y = pad_top + i * row_h
            cy = y + row_h / 2
            ratio = (got / total) if total else 0.0
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

    def _figure_dispersion(self, result: Dict[str, Any]) -> str:
        """图二:**逐题离散度** —— 每条指标在八道题之间差多少。

        ## 数据早就在载荷里,报告从没印过

        `l1.dispersion` 是 20 条指标的逐题中位 / IQR / 最小 / 最大,
        `/api/report/structured` 一直在发,前端 `L1IndicatorTable` 一直在渲染 ——
        只有 HTML 交付物一个字都没提。它回答的正是读者最想问的那件事:
        **这个读数稳不稳**(同一场里,换个题目它变不变)。

        ## 为什么是"小倍数"而不是一张散点

        20 条指标的单位各不相同(占比、次/分钟、赫兹、秒)。**放在同一条 x 轴上
        就是双轴的变种** —— 位置毫无意义。所以每行各自成轴:左端是本场最小值、
        右端是最大值,刻度就标在两端。⟹ 行与行之间**只比形状,不比长短**,
        这句话必须写在图注里,否则读者会横向比。

        ⚠️ 每个数都是载荷里的原值(`min` / `median` / `max`),渲染层不重算。
        """
        disp = ((result.get("l1") or {}).get("dispersion") or {})
        if not disp:
            return ""
        rows = [(name, d) for name, d in disp.items() if isinstance(d, dict)]
        if not rows:
            return ""

        label_w, plot_w, row_h, pad_top = 128, 360, 34, 10
        width = label_w + plot_w + 96
        height = pad_top * 2 + row_h * len(rows)
        parts: list[str] = [
            f'<svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="每条指标在题与题之间的分布">']
        for i, (name, d) in enumerate(rows):
            try:
                lo, mid, hi = float(d["min"]), float(d["median"]), float(d["max"])
            except (KeyError, TypeError, ValueError):
                continue
            y = pad_top + i * row_h
            cy = y + row_h / 2
            label = l1_quality.indicator_label(name)
            parts.append(f'<text class="label" x="0" y="{cy + 4:.1f}">{label}</text>')
            parts.append(f'<rect class="track" x="{label_w}" y="{cy - 1:.1f}" '
                         f'width="{plot_w}" height="2" rx="1"/>')

            # ★ 各题取到同一个值时(`min == max`),这一行**没有左右可言**。
            #
            # 实测那一场 20 行里有 6 行如此:四条是"每道题都是 0"(一次也没超过
            # 阈值),`pose_valid_ratio` 是"每道题都是 1"。按常规画法会得到
            # 一条零长度线 + 两个**重叠在同一点**的端点标签 —— 读者看到的是
            # 一个糊掉的「00」,而真相恰恰是这一行**最稳**。
            # ⟹ 点放正中,只标一个数,并在旁边说明"各题相同"。
            flat = (hi == lo)
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
            # 两侧各留 6% 的余量,免得端点标签压在边界上被裁掉。
            x0, x1 = lo - span * 0.06, hi + span * 0.06
            scale = plot_w / (x1 - x0)

            def _x(v: float) -> float:
                return label_w + (v - x0) * scale

            parts.append(f'<line class="range" x1="{_x(lo):.1f}" y1="{cy:.1f}" '
                         f'x2="{_x(hi):.1f}" y2="{cy:.1f}"/>')
            parts.append(f'<circle class="median" cx="{_x(mid):.1f}" cy="{cy:.1f}" '
                         f'r="4.5"/>')
            # 三个数都标出来:这一行的轴是**本行自己的**,不标数就只剩形状。
            parts.append(f'<text class="tick" x="{_x(lo):.1f}" y="{cy - 8:.1f}" '
                         f'text-anchor="middle">{_fmt_compact(lo)}</text>')
            parts.append(f'<text class="tick" x="{_x(hi):.1f}" y="{cy - 8:.1f}" '
                         f'text-anchor="middle">{_fmt_compact(hi)}</text>')
            parts.append(f'<text class="value" x="{width - 4}" y="{cy + 4:.1f}" '
                         f'text-anchor="end">{_fmt_compact(mid)}</text>')
        parts.append("</svg>")
        return "".join(parts)

    @staticmethod
    def _n_flat_rows(result: Dict[str, Any]) -> int:
        """离散度里有几行「各题取到同一个值」—— 图注要说这个数。"""
        disp = ((result.get("l1") or {}).get("dispersion") or {})
        n = 0
        for d in disp.values():
            if not isinstance(d, dict):
                continue
            try:
                if float(d["min"]) == float(d["max"]):
                    n += 1
            except (KeyError, TypeError, ValueError):
                continue
        return n

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

        def _plain(t: str) -> str:
            # 载荷里的这几句是**开发者写的注释体**,带 `**` / 反引号。
            # 前端 `L1IndicatorTable.plain()` 做同一件事(纯文本界面里它们是噪音)。
            return re.sub(r"\*\*|`", "", t)

        def _ul(items) -> str:
            return "<ul>" + "".join(f"<li>{_plain(str(i))}</li>" for i in items) + "</ul>"

        blocks: list[str] = []

        gaps = l1.get("gaps") or []
        if gaps and _want("gaps"):
            blocks.append(
                "<div class='l1-audit'><h3>本场报告的限制（素材层面，不是算错）</h3>"
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
                extra = ("；不在任何回答窗口内的帧：" +
                         " · ".join(f"{k} {n} 帧" for k, n in outside.items()) +
                         "（这些帧不计入 F/G 的分母）")
            blocks.append(
                f"<div class='l1-audit'><h3>统计口径</h3><p>{text}{extra}</p></div>")

        return "".join(blocks)

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

        # 头部聚合呈现:只讲事实(spec §5.4 / §5.6)。
        # 原分数卡渲染「点分 + 五档评语」(如「0.0 / 综合行为观测评分 / 待提升」)——
        # 那是把未标定标尺上的复合点分当对候选人的评定。改为:覆盖事实 + 依据 +
        # 区间,区间只能来自本场会话自己的测量(§5.5:没有真实常模就不给位置)。
        coverage = result["coverage"]
        basis_items = "".join(
            f"<li>{slot['display_name']} · {slot['indicator']}："
            f"原始值 {_fmt_num(slot['raw_value'])}；"
            f"本场会话内观测区间 {_interval_text(slot['observed_interval'])}"
            f"（与原始值同量纲）；有效样本量 {slot['n_valid']}</li>"
            for slot in coverage["passed_slots"]
        )
        basis_html = (f"<ul>{basis_items}</ul>" if basis_items
                      else "<p>本次会话没有指标通过证据门。</p>")

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
                             f"值本身是 0 的,意思是每题都没超过阈值,不是没测到。")
            disp_block = f"""
            <h2>题与题之间稳不稳</h2>
            <p class="note">同一个指标,{q}里各量了一次 —— 下图每一行是一条指标,
            左端是本场最小值、右端是最大值,圆点是中位数(右侧数字)。
            <strong>每行各有自己的刻度</strong>,所以请横向看"这一行散不散",
            不要拿两行的长短互相比较。</p>
            <figure>{fig_disp}
            <figcaption>{n_disp} 条指标在题与题之间有两次以上测量。
            只测了一次的指标不出现在这里 —— 单次测量没有"离散"可言,
            给个 0 会把"没测"说成"很稳"。{flat_note}</figcaption></figure>
            """
        else:
            why = ("这一场是整场一个窗口,没有逐题切分" if not n_win
                   else "每条指标在每道题上都只测到一次")
            disp_block = f"""
            <h2>题与题之间稳不稳</h2>
            <p class="note">这一场报不出逐题离散度:{why},所以没有题与题之间的
            中位数、最小值、最大值可给。下面的所有读数都是<b>整场一个值</b>,
            不是"每题一个值取平均"—— 两者不能混着读。</p>
            """

        cov_block = ""
        if fig_cov:
            cov_block = f"""
            <h2>{self._coverage_heading(result)}</h2>
            <p class="note">每一维声明的槽数不同(见右端的「量到 / 声明」),
            所以请比"自己那一维填了多少",不要把两维的条长直接比大小。</p>
            <figure>{fig_cov}
            <figcaption>条形 = 通过证据门的指标槽数;浅色轨道 = 该维声明的全部槽数。
            未过门的槽与原因逐条列在下面的维度明细里。</figcaption></figure>
            """

        # 「与旧引擎报告对比」是**引擎迁移说明**,每份报告都有、与这一场无关。
        # 印在正文里必然被读成"跟上一次比怎么样"(使用者原话)。内容不删,收进附录。
        appendix = self._render_l1_audit_text(result, only=("comparability",))
        appendix_block = ""
        if appendix:
            appendix_block = f"""
            <details>
                <summary>附录：与旧引擎报告的对照说明（复核口径用，与这一场无关）</summary>
                <p class="note">这一节讲的是<b>两代引擎</b>怎么量同一件事,
                每一份报告都会印。它不描述这一场发生了什么,不必当成"跟上一次比"。</p>
                {appendix}
            </details>
            """

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JingXin 面试行为观测报告</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>JingXin 面试行为观测报告</h1>
  <div class="meta">{datetime.now().strftime("%Y-%m-%d %H:%M")} · 口径版本 {result['model_metadata']['version']}</div>
  <div class="sources">{sources_disclosure(sources or {}, target,
                                            modality_summary=(result.get("l1") or {})
                                            .get("modality_summary"))}</div>
</header>

<div class="hero">
  <span class="cover-number">{coverage['n_passed']} / {coverage['n_slots']}</span>
  <span class="hero-cap">个指标槽通过证据门 · 置信度上限：{coverage['confidence_cap']}</span>
</div>
<p class="lead">{self._lead_sentence(result)}</p>
<p class="note">{result['summary_narrative']}</p>

{cov_block}
{disp_block}

<h2>每一维量到了什么</h2>
{self._render_vanished_dimensions(result)}
{deep_analysis_html}

<h2>依据：通过证据门的指标</h2>
{basis_html}

{self._render_l1_audit_text(result, only=("gaps", "low_confidence", "windows"))}

{appendix_block}

<footer>JingXin 多模态行为观测 · 报告由系统自动生成 · 不含评分与评级</footer>
</div>
</body>
</html>
"""


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